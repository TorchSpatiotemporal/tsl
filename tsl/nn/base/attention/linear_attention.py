from typing import Optional

import torch
try:
    from fast_transformers.attention import CausalLinearAttention as CLAttention
    from fast_transformers.masking import TriangularCausalMask, LengthMask
except ModuleNotFoundError:
    CLAttention = None
from torch import Tensor
from torch import nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor


class CausalLinearAttention(nn.Module):

    def __init__(self, embed_dim, heads,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 concat: bool = True,
                 dim: int = 1) -> None:
        super(CausalLinearAttention, self).__init__()
        if CLAttention is None:
            raise RuntimeError("Install optional dependency 'fast_transformers'"
                               " to use CausalLinearAttention.")
        # store dimensions
        self.embed_dim = int(embed_dim)
        self.qdim = int(qdim) if qdim is not None else self.embed_dim
        self.kdim = int(kdim) if kdim is not None else self.embed_dim
        self.vdim = int(vdim) if vdim is not None else self.embed_dim
        self.out_channels = int(out_channels) if out_channels is not None \
            else self.embed_dim

        self.heads = heads
        self.concat = concat
        self.dim = dim

        if self.concat:
            self.head_dim = self.embed_dim // self.heads
            out_dim = self.out_channels // self.heads
            assert self.head_dim * self.heads == self.embed_dim, \
                "embed_dim must be divisible by heads"
            assert out_dim * self.heads == self.out_channels, \
                "out_channels must be divisible by heads"
        else:
            self.head_dim, out_dim = self.embed_dim, self.out_channels

        self.lin_key = Linear(self.kdim, self.heads * self.head_dim,
                              bias_initializer='zeros')
        self.lin_query = Linear(self.qdim, self.heads * self.head_dim,
                                bias_initializer='zeros')
        self.lin_value = Linear(self.vdim, self.heads * out_dim,
                                bias_initializer='zeros')

        self.attention = CLAttention(self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()

    def forward(self, query: Tensor,
                key: OptTensor = None,
                value: OptTensor = None):
        # If key and value not provided, self attention
        if key is None:
            key = query
        if value is None:
            value = key

        L, H, E = query.size(self.dim), self.heads, self.head_dim

        # move sequence dimension to penultimate dimension -> [*b s c]
        query = query.transpose(self.dim, -2)
        key = key.transpose(self.dim, -2)
        value = value.transpose(self.dim, -2)
        #
        B = value.shape[:-2]
        if not (torch.tensor(query.shape[:-2] + key.shape[:-2]) == 1).all():
            query = query.expand(*B, *query.shape[-2:])
            key = key.expand(*B, *key.shape[-2:])
        # project and split heads
        query = self.lin_query(query).view(-1, L, H, E)
        key = self.lin_key(key).view(-1, L, H, E)
        value = self.lin_value(value).view(-1, L, H, E)

        attn_mask = TriangularCausalMask(L, device=query.device)
        key_lengths = LengthMask(torch.LongTensor([1]), 1, device=query.device)

        out = self.attention(query.float(), key.float(), value.float(),
                             attn_mask, query_lengths=None,
                             key_lengths=key_lengths)

        # reshape out to [*b, s, *n, c]
        if not self.concat:
            out = out.view(*B, L, H, E).mean(-2)\
                .transpose(self.dim, -2).contiguous()
        else:
            out = out.view(*B, L, -1).transpose(self.dim, -2).contiguous()

        return out
