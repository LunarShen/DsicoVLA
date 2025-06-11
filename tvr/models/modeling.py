from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
import copy
allgather = AllGather.apply
allgather2 = AllGather2.apply

logger = logging.getLogger(__name__)

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class VTRModel(nn.Module):
    def __init__(self, config):
        super(VTRModel, self).__init__()
        
        self.config = config
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        self.lora_adapter_dim = config.lora_adapter_dim
        self.alpha = config.alpha
        self.beta = config.beta
        self.H_V = config.H_V
        self.H_L = config.H_L
        logger.info(f"lora_adapter_dim {self.lora_adapter_dim} | alpha {self.alpha} | beta {self.beta} | H_V {self.H_V} | H_L {self.H_L}")
        
        assert backbone in _PT_NAME
        model_path = os.path.join(config.pretrained_path, _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, 
                         self.lora_adapter_dim, self.H_V, self.H_L)
            
        self.loss_fct = CrossEn(config)

        self.clip.load_state_dict(state_dict, strict=False)

    def forward(self, text_ids, text_mask, f_text_ids, f_text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])

        f_text_ids = f_text_ids.view(-1, f_text_ids.shape[-1])
        f_text_mask = f_text_mask.view(-1, f_text_mask.shape[-1])
        
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_f, d, h, w = video.shape
            video = video.view(b * n_f, d, h, w)
        else:
            b, pair, n_f, ts, channel, h, w = video.shape
            video = video.view(b * pair * n_f * ts, channel, h, w)
        
        ### stage 1
        video_L = self.get_text_feat(text_ids, text_mask)
        frame_L = self.get_train_text_feat(f_text_ids, f_text_mask, n_f)
        framevideo_V, frame_V = self.get_train_video_feat(video, video_mask)
        
        video_L = allgather(video_L, self.config)
        framevideo_V = allgather(framevideo_V, self.config)
        frame_L = allgather(frame_L, self.config)
        frame_V = allgather(frame_V, self.config)
        torch.distributed.barrier()
        
        logit_scale = self.clip.logit_scale.exp()
        loss = 0.
        
        video_L = video_L / video_L.norm(dim=-1, keepdim=True)  # b_t x D
        framevideo_V = framevideo_V / framevideo_V.norm(dim=-1, keepdim=True) # b_t x b_v x D
        
        t2v_logits_videoLframevideoV = torch.einsum('td,vd->tv', [video_L, framevideo_V])

        loss_t2v_videoLframevideoV = self.loss_fct(t2v_logits_videoLframevideoV * logit_scale)
        loss_v2t_videoLframevideoV = self.loss_fct(t2v_logits_videoLframevideoV.T * logit_scale)
        loss_videoLframevideoV = (loss_t2v_videoLframevideoV + loss_v2t_videoLframevideoV) / 2

        frame_L = frame_L / frame_L.norm(dim=-1, keepdim=True)
        frame_V = frame_V / frame_V.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [frame_L, frame_V])

        retrieve_t2v_logits, _ = retrieve_logits.max(dim=-1)  # abtv -> abt
        retrieve_v2t_logits, _ = retrieve_logits.max(dim=-2)  # abtv -> abv
        retrieve_t2v_logits = retrieve_t2v_logits * F.softmax(retrieve_t2v_logits * logit_scale, dim=-1)
        retrieve_t2v_logits = retrieve_t2v_logits.sum(-1)
        retrieve_v2t_logits = retrieve_v2t_logits * F.softmax(retrieve_v2t_logits * logit_scale, dim=-1)
        retrieve_v2t_logits = retrieve_v2t_logits.sum(-1)
        
        t2v_logits_frameLframeV = (retrieve_t2v_logits + retrieve_v2t_logits) / 2.0

        loss_t2v_frameLframeV = self.loss_fct(t2v_logits_frameLframeV * logit_scale)
        loss_v2t_frameLframeV = self.loss_fct(t2v_logits_frameLframeV.T * logit_scale)
        loss_frameLframeV = (loss_t2v_frameLframeV + loss_v2t_frameLframeV) / 2
        
        distillation_logits_t2v = F.kl_div(
                F.log_softmax(t2v_logits_videoLframevideoV * logit_scale, dim=-1),
                F.log_softmax(t2v_logits_frameLframeV * logit_scale, dim=-1),
                reduction='sum',
                log_target=True
            ) / t2v_logits_videoLframevideoV.numel() * 100.0
        distillation_logits_v2t = F.kl_div(
                F.log_softmax(t2v_logits_videoLframevideoV.T * logit_scale, dim=-1),
                F.log_softmax(t2v_logits_frameLframeV.T * logit_scale, dim=-1),
                reduction='sum',
                log_target=True
            ) / t2v_logits_videoLframevideoV.T.numel() * 100.0
        distillation_logits = (distillation_logits_t2v + distillation_logits_v2t) / 2

        loss = loss_videoLframevideoV + \
                loss_frameLframeV * self.alpha + \
                distillation_logits * self.beta
        
        return loss, loss_videoLframevideoV, loss_frameLframeV, distillation_logits

    def stage1_eval(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        ### stage 1
        cls = self.get_text_feat(text_ids, text_mask)
        video = self.get_video_feat(video, video_mask)

        return cls, video

    def stage2_eval(self, cls, text_mask, video_feat, video_mask):
        ### stage 2
        logit_scale = self.clip.logit_scale.exp()
        
        t_feat = cls / cls.norm(dim=-1, keepdim=True)  # b_t x D
        v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True) # b_t x b_v x D

        t2v_logits = torch.einsum('td,vd->tv', [t_feat, v_feat])
        
        return t2v_logits * logit_scale

    def get_text_feat(self, text_ids, orig_mask):
        b = text_ids.size(0)
        x = self.clip.token_embedding(text_ids)  # [batch_size, n_ctx, d_model]
        max_t_len = x.size(1)
        pos_emd = self.clip.positional_embedding[:max_t_len, :]
        x = x + pos_emd

        mask = orig_mask
        text_length = max_t_len
        attn_mask = self.clip.build_attention_mask(text_length).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((text_length, text_length)).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask>0, attn_mask, inf)
    
        for res_i, res_block in enumerate(self.clip.transformer.resblocks):
            (x, attn_mask) = res_block((x, attn_mask))

        hidden = self.clip.ln_final(x) @ self.clip.text_projection
        cls = hidden[torch.arange(hidden.shape[0]), text_ids.argmax(dim=-1)]

        cls = cls.float()
        cls = cls.view(b, -1, cls.size(-1)).squeeze(1)
        return cls

    def get_train_text_feat(self, text_ids, orig_mask, n_f):
        b = text_ids.size(0)
        x = self.clip.token_embedding(text_ids)  # [batch_size, n_ctx, d_model]
        max_t_len = x.size(1)
        pos_emd = self.clip.positional_embedding[:max_t_len, :]
        x = x + pos_emd

        mask = orig_mask
        text_length = max_t_len
        attn_mask = self.clip.build_attention_mask(text_length).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((text_length, text_length)).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask>0, attn_mask, inf)
    
        for res_i, res_block in enumerate(self.clip.transformer.resblocks):
            (x, attn_mask) = res_block((x, attn_mask), n_f, text_ids.argmax(dim=-1))
                
        hidden = self.clip.ln_final(x) @ self.clip.text_projection
        cls = hidden[torch.arange(hidden.shape[0]), text_ids.argmax(dim=-1)]

        cls = cls.float()
        cls = cls.view(b // n_f, n_f, cls.size(-1))
        frame_cls = cls.clone()

        return frame_cls

    def get_train_video_feat(self, video, video_mask):
        b, n_f = video_mask.size()
        x = video
            
        x = self.clip.visual.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        
        _, token_len, d_v = x.size()
        # assert 50 == token_len

        for res_i, res_block in enumerate(self.clip.visual.transformer.resblocks):
            x = res_block(x, x_shape=[b, n_f, token_len, d_v])
        
        x = x.view(b, n_f, token_len, d_v)
        hidden = self.clip.visual.ln_post(x) @ self.clip.visual.proj
        video_feat = hidden[:, :, 0, :].float()
    
        video_feat = video_feat.contiguous()
        frame_feat = video_feat.clone()
        
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        video_feat = self.get_video_avg_feat(video_feat, video_mask)
        
        return video_feat, frame_feat

    def get_video_feat(self, video, video_mask):
        b, n_f = video_mask.size()
        x = video
            
        x = self.clip.visual.conv1(x)  

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        
        _, token_len, d_v = x.size()
        # assert 50 == token_len

        for res_i, res_block in enumerate(self.clip.visual.transformer.resblocks):
            x = res_block(x, x_shape=[b, n_f, token_len, d_v])
        
        x = x.view(b, n_f, token_len, d_v)
        hidden = self.clip.visual.ln_post(x) @ self.clip.visual.proj
        video_feat = hidden[:, :, 0, :].float()
    
        video_feat = video_feat.contiguous()
        
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        video_feat = self.get_video_avg_feat(video_feat, video_mask)
        
        return video_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
