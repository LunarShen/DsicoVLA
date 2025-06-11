from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from timm.models.layers import drop_path
import torch
from torch import nn
import math
import torch.nn.functional as F
from .until_module import LayerNorm, ACT2FN
from collections import OrderedDict
from einops import rearrange

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention_GlobalCLS(nn.Module):
    def __init__(self, embed_dim, num_heads, lora_dim): # , lora_dim
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5
        self.lora_dim = lora_dim

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.TVPt_LoRA_a = nn.Parameter(torch.zeros(lora_dim, embed_dim))
        nn.init.kaiming_uniform_(self.TVPt_LoRA_a, a=math.sqrt(5))
        self.TVPt_LoRA_b = nn.Parameter(torch.zeros(3 * embed_dim, lora_dim))
        

        self.TVPt_down_weight = nn.Parameter(torch.empty(lora_dim, embed_dim))
        self.TVPt_down_bias = nn.Parameter(torch.empty(lora_dim))
        self.TVPt_activate = QuickGELU()
        self.TVPt_up_weight = nn.Parameter(torch.zeros(embed_dim, lora_dim))
        self.TVPt_up_bias = nn.Parameter(torch.zeros(embed_dim))
        nn.init.xavier_uniform_(self.TVPt_down_weight)
        nn.init.normal_(self.TVPt_down_bias, std=1e-6)
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x, attn_mask, n_f, top_idx):
        bsz, tgt_len, embed_dim = x.size()
        token_len = tgt_len
        d_t = embed_dim
        b = bsz // n_f
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        qkv_delta = F.linear(x, self.TVPt_LoRA_a)
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta

        q_1 = q[torch.arange(bsz), :, top_idx, :]
        q_1 = rearrange(q_1, '(b f) h d-> b h f d', b=b, f=n_f)
        
        q = q * self.scaling
        attn = (q @ k.transpose(-2, -1))
        
        attn += attn_mask[:,None,:,:]
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)

        k = rearrange(k, '(b f) h p d-> b h (f p) d', b=b, f=n_f)
        v = rearrange(v, '(b f) h p d-> b h (f p) d', b=b, f=n_f)
        attn_mask = attn_mask[:,-1,:]
        attn_mask = rearrange(attn_mask, '(b f) l-> b (f l)', b=b, f=n_f)
        
        q_1 = q_1 * self.scaling
        attn_1 = (q_1 @ k.transpose(-2, -1))

        attn_1 += attn_mask[:,None,None,:]
        
        attn_1 = attn_1.softmax(dim=-1)
        x_1 = (attn_1 @ v).transpose(1, 2).reshape(b, n_f, embed_dim)
        x_1 = F.linear(x_1, self.out_proj.weight, self.out_proj.bias)

        x_0 = x[torch.arange(bsz), top_idx, :].clone()
        x_0 = rearrange(x_0, '(b f) d-> b f d', b=b, f=n_f)
        
        x_0 = F.linear(x_0, self.TVPt_down_weight, self.TVPt_down_bias)
        x_0 = self.TVPt_activate(x_0)
        x_0 = F.linear(x_0, self.TVPt_up_weight, self.TVPt_up_bias)

        update_cls = x_0 + x_1
        update_cls = rearrange(update_cls, 'b f d -> (b f) d')
        x[torch.arange(bsz), top_idx, :] = update_cls

        return x
    
    def forward_normal(self, x, attn_mask):
        bsz, tgt_len, embed_dim = x.size()
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    
        qkv_delta = F.linear(x, self.TVPt_LoRA_a)
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta
        
        q = q * self.scaling
        attn = (q @ k.transpose(-2, -1))
        
        attn += attn_mask[:,None,:,:]
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        
        return x

class ResidualAttentionBlock_GlobalCLS(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_dim: int, attn_mask=None):
        super(ResidualAttentionBlock_GlobalCLS, self).__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention_GlobalCLS(d_model, n_head, lora_dim)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        
    def attention(self, x: torch.Tensor, attn_mask_: torch.Tensor):
        # attn_mask_ = attn_mask_.repeat_interleave(self.n_head, dim=0)
        # attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        # return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device)
        return self.attn(x, attn_mask_)

    def forward(self, para_tuple: tuple, n_f=None, top_idx=None):
        x, attn_mask = para_tuple

        if n_f is None:
            attn_mask_ = attn_mask.to(dtype=x.dtype, device=x.device)
            x = x + self.attn.forward_normal(self.ln_1(x), attn_mask_)
        else:
            attn_mask_ = attn_mask.to(dtype=x.dtype, device=x.device)
            x = x + self.attn(self.ln_1(x), attn_mask_, n_f, top_idx)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, lora_dim): # , lora_dim
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = float(self.head_dim) ** -0.5
        self.lora_dim = lora_dim

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.TVPt_LoRA_a = nn.Parameter(torch.zeros(lora_dim, embed_dim))
        nn.init.kaiming_uniform_(self.TVPt_LoRA_a, a=math.sqrt(5))
        self.TVPt_LoRA_b = nn.Parameter(torch.zeros(3 * embed_dim, lora_dim))

        # self._reset_parameters() ### lora init
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, attn_mask):
        bsz, tgt_len, embed_dim = x.size()
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    
        qkv_delta = F.linear(x, self.TVPt_LoRA_a)
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta
        
        q = q * self.scaling
        attn = (q @ k.transpose(-2, -1))
        
        attn += attn_mask[:,None,:,:]
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_dim: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Attention(d_model, n_head, lora_dim)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        
    def attention(self, x: torch.Tensor, attn_mask_: torch.Tensor):
        # attn_mask_ = attn_mask_.repeat_interleave(self.n_head, dim=0)
        # attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        # return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device)
        return self.attn(x, attn_mask_)

    def forward(self, para_tuple: tuple, n_f=None, top_idx=None):
        x, attn_mask = para_tuple 
                          
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, lora_dim: int, global_layers: int):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, lora_dim) for _ in range(layers)])
        resblocks_list = [ResidualAttentionBlock(width, heads, lora_dim) for _ in range(layers-global_layers)] + \
                            [ResidualAttentionBlock_GlobalCLS(width, heads, lora_dim) for _ in range(global_layers)]

        logger.info(len(resblocks_list))
        
        self.resblocks = nn.Sequential(*resblocks_list)
        
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]