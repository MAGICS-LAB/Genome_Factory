import math
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clones


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None, dropout: Optional[nn.Dropout] = None) -> Tuple[
        torch.Tensor, Any]:
        """
        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1, 1, max_len)
            `dropout`: nn.Dropout

        Returns:
            `weighted value`: shape (batch_size, n_heads, max_len, d_v)
            `weight matrix`: shape (batch_size, n_heads, max_len, max_len)
        """
        d_k = query.size(-1)  # d_k = d_model / n_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # B*H*L*L
        if mask is not None:
            scores = scores.masked_fill(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1)  # B*H*L*L
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.sdpa = ScaledDotProductAttention()
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """
        Args: 
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
            `value`: shape (batch_size, max_len, d_model)
            `mask`: shape (batch_size, max_len)
        
        Returns:
            shape (batch_size, max_len, d_model)
        """
        if mask is not None:
            # Same mask applied to all h heads. B*1*1*L
            mask = mask.unsqueeze(1).unsqueeze(1)
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: B x H x L x D_v
        x, self.attn = self.sdpa(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)