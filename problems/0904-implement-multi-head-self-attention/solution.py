import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "Error"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
    
        B, T, d_model = x.shape

        # Projection. Shape of Q,K,V -> (B, T, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # attention shape - (B, num_heads, num_tokens, num_tokens)
        att = (Q @ K.transpose(2, 3)) / math.sqrt(self.d_head)
        # Mask future positions with negative infinity
        if mask is not None:
            att = torch.where(mask.bool()[None, None, :, :], -torch.inf, att)

        # stable softmax
        att = torch.softmax(att, dim=-1)
        
        # result shape = (B, num_heads, num_tokens, head_dim)
        result = att @ V
        result = result.transpose(1, 2)
        result = result.reshape(B, T, d_model)

        return self.out_proj(result)