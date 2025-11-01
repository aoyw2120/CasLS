import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


from torch import Tensor, nn


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        bias=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query, key, value,  # (len + 1) * batch_size * embedding
        attn_bias,  # batch_size * num_nodes * num_nodes * num_nodes
        key_padding_mask  # batch_size * (num_node + 1)
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        q = self.q_proj(query)  # (len + 1) * batch_size * embedding
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = (q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        k = (k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))
        v = (v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1))

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # (batch_size*num_heads)*(len+1)*(len+1)

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        '''if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask'''

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),  # batch_size * 1 * 1 * (num_node + 1)
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1)  # batch_size*num_heads * (num_node+1) * (num_node+1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, v)  # batch_size*num_heads * (num_node+1) * head_emb

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # (num_node+1) * batch_size * embed_dim
        attn = self.out_proj(attn)  # (num_node+1) * batch_size * embed_dim

        return attn
