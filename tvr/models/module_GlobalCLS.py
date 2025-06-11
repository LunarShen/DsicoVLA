from collections import OrderedDict
from typing import Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import math
from einops import rearrange
import logging

logger = logging.getLogger(__name__)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, lora_dim):
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
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)
    
    def forward(self, x, x_shape):
        b, n_f, token_len, d_v = x_shape
        bsz, tgt_len, embed_dim = x.size()
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        qkv_delta = F.linear(x, self.TVPt_LoRA_a)
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta

        q_1 = rearrange(q, '(b f) h p d-> b h f p d', b=b, f=n_f)
        q_1 = q_1[:,:,:,0,:]
        
        q = q * self.scaling
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)

        k = rearrange(k, '(b f) h p d-> b h (f p) d', b=b, f=n_f)
        v = rearrange(v, '(b f) h p d-> b h (f p) d', b=b, f=n_f)

        q_1 = q_1 * self.scaling
        attn_1 = (q_1 @ k.transpose(-2, -1))
        attn_1 = attn_1.softmax(dim=-1)
        x_1 = (attn_1 @ v).transpose(1, 2).reshape(b, n_f, embed_dim)
        x_1 = F.linear(x_1, self.out_proj.weight, self.out_proj.bias)

        x = x.reshape(b,n_f,token_len,d_v)
        # x[:,:,0,:] = (x[:,:,0,:] + x_1) / 2
        x_0 = x[:,:,0,:].clone()
        
        x_0 = F.linear(x_0, self.TVPt_down_weight, self.TVPt_down_bias)
        x_0 = self.TVPt_activate(x_0)
        x_0 = F.linear(x_0, self.TVPt_up_weight, self.TVPt_up_bias)
            
        x[:,:,0,:] = x_0 + x_1
        x = x.reshape(b * n_f, token_len, d_v)

        return x

    def forward_normal(self, x):
        bsz, tgt_len, embed_dim = x.size()
        
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        qkv_delta = F.linear(x, self.TVPt_LoRA_a)
        qkv_delta = F.linear(qkv_delta, self.TVPt_LoRA_b).reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        q,k,v = q+q_delta,k+k_delta,v+v_delta
        
        q = q * self.scaling
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        
        return x

class ResidualAttentionBlock_GlobalCLS(nn.Module):
    def __init__(self, d_model: int, n_head: int, lora_dim: int, attn_mask=None):
        super(ResidualAttentionBlock_GlobalCLS, self).__init__()

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

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))  # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
    
    def forward(self, x, x_shape=None, attn_mask=None):
        if x_shape is None:
            x = x + self.attn.forward_normal(self.ln_1(x))
        else:
            x = x + self.attn(self.ln_1(x), x_shape=x_shape)
        x = x + self.mlp(self.ln_2(x))
        return x