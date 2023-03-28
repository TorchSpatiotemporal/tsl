import math
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import OptTensor

from tsl.nn.utils import get_functional_activation


class PositionalEncoding(nn.Module):
    """The positional encoding from the paper `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017)."""

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.,
                 max_len: int = 5000,
                 affinity: bool = False,
                 batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if affinity:
            self.affinity = nn.Linear(d_model, d_model)
        else:
            self.affinity = None
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x: Tensor):
        """"""
        if self.affinity is not None:
            x = self.affinity(x)
        if self.batch_first:
            pe = self.pe[:x.size(1), :]
        else:
            pe = self.pe[:x.size(0), :]
        x = x + pe
        return self.dropout(x)


@torch.jit.script
def _get_causal_mask(seq_len: int,
                     diagonal: int = 0,
                     device: Optional[torch.device] = None):
    # mask keeping only previous steps
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    causal_mask = torch.triu(ones, diagonal)
    return causal_mask


class AttentionEncoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 add_positional_encoding: bool = False,
                 bias: bool = True,
                 activation: Optional[str] = None) -> None:
        super(AttentionEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim

        self.lin_query = Linear(qdim, self.embed_dim, bias) \
            if qdim is not None else nn.Identity()

        self.lin_key = Linear(kdim, self.embed_dim, bias) \
            if qdim is not None else nn.Identity()

        self.lin_value = Linear(vdim, self.embed_dim, bias) \
            if qdim is not None else nn.Identity()

        self.activation = get_functional_activation(activation)
        self.pe = PositionalEncoding(self.embed_dim) \
            if add_positional_encoding else nn.Identity()

    def forward(self,
                query: Tensor,
                key: OptTensor = None,
                value: OptTensor = None):
        """"""
        # inputs: [batches, steps, nodes, channels]
        if key is None:
            key = query
        if value is None:
            value = key
        query = self.pe(self.activation(self.lin_query(query)))
        key = self.pe(self.activation(self.lin_key(key)))
        value = self.activation(self.lin_value(value))
        return query, key, value


class MultiHeadAttention(nn.MultiheadAttention):
    """The multi-head attention from the paper `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017) for
    spatiotemporal data."""

    def __init__(self,
                 embed_dim,
                 heads,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 axis='steps',
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 device=None,
                 dtype=None,
                 causal=False) -> None:
        if axis in ['steps', 0]:
            shape = 's (b n) c'
        elif axis in ['nodes', 1]:
            if causal:
                raise ValueError(
                    f'Cannot use causal attention for axis "{axis}".')
            shape = 'n (b s) c'
        else:
            raise ValueError("Axis can either be 'steps' (0) or 'nodes' (1), "
                             f"not '{axis}'.")
        self._in_pattern = f'b s n c -> {shape}'
        self._out_pattern = f'{shape} -> b s n c'
        self.causal = causal
        # Impose batch dimension as the second one
        super(MultiHeadAttention, self).__init__(embed_dim,
                                                 heads,
                                                 dropout=dropout,
                                                 bias=bias,
                                                 add_bias_kv=add_bias_kv,
                                                 add_zero_attn=add_zero_attn,
                                                 kdim=kdim,
                                                 vdim=vdim,
                                                 batch_first=False,
                                                 device=device,
                                                 dtype=dtype)
        # change projections
        if qdim is not None and qdim != embed_dim:
            self.qdim = qdim
            self.q_proj = Linear(self.qdim, embed_dim)
        else:
            self.qdim = embed_dim
            self.q_proj = nn.Identity()

    def forward(self,
                query: Tensor,
                key: OptTensor = None,
                value: OptTensor = None,
                key_padding_mask: OptTensor = None,
                need_weights: bool = True,
                attn_mask: OptTensor = None):
        """"""
        # inputs: [batches, steps, nodes, channels] -> [s (b n) c]
        if key is None:
            key = query
        if value is None:
            value = key
        batch = value.shape[0]
        query, key, value = [
            rearrange(x, self._in_pattern) for x in (query, key, value)
        ]

        if self.causal:
            causal_mask = _get_causal_mask(key.size(0),
                                           diagonal=1,
                                           device=query.device)
            if attn_mask is None:
                attn_mask = causal_mask
            else:
                attn_mask = torch.logical_and(attn_mask, causal_mask)
        attn_output, attn_weights = super(MultiHeadAttention, self).forward(
            self.q_proj(query), key, value, key_padding_mask, need_weights,
            attn_mask)
        attn_output = rearrange(attn_output, self._out_pattern, b=batch) \
            .contiguous()
        if attn_weights is not None:
            attn_weights = rearrange(attn_weights,
                                     '(b d) l m -> b d l m',
                                     b=batch).contiguous()
        return attn_output, attn_weights


class TemporalSelfAttention(nn.Module):
    """Temporal Self Attention layer.

    Args:
        embed_dim (int): Size of the hidden dimension associeted with each node
            at each time step.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
        bias (bool, optional): Whther to add a learnable bias.
        device (optional): Device on which store the model.
        dtype (optional): Data Type of the parameters.

    Examples::
        >>> import torch
        >>> m = TemporalSelfAttention(32, 4, -1)
        >>> input = torch.randn(128, 24, 10, 20)
        >>> output, _ = m(input)
        >>> print(output.size())
        torch.Size([128, 24, 10, 32])
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 in_channels=None,
                 dropout=0.,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super(TemporalSelfAttention, self).__init__()

        self.embed_dim = embed_dim

        if in_channels is not None:
            self.input_encoder = Linear(in_channels, self.embed_dim)
        else:
            self.input_encoder = nn.Identity()

        self.attention = MultiHeadAttention(embed_dim,
                                            num_heads,
                                            axis='steps',
                                            dropout=dropout,
                                            bias=bias,
                                            device=device,
                                            dtype=dtype)

    def forward(self,
                x,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True):
        """"""
        # x: [batch, steps, nodes, in_channels]
        x = self.input_encoder(x)  # -> [batch, steps, nodes, embed_dim]
        return self.attention(x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=need_weights)


class SpatialSelfAttention(nn.Module):
    """Spatial Self Attention layer.

    Args:
        embed_dim (int): Size of the hidden dimension associeted with each node
            at each time step.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
        bias (bool, optional): Whther to add a learnable bias.
        device (optional): Device on which store the model.
        dtype (optional): Data Type of the parameters.

    Examples::
        >>> import torch
        >>> m = SpatialSelfAttention(32, 4, -1)
        >>> input = torch.randn(128, 24, 10, 20)
        >>> output, _ = m(input)
        >>> print(output.size())
        torch.Size([128, 24, 10, 32])
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 in_channels=None,
                 dropout=0.,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super(SpatialSelfAttention, self).__init__()

        self.embed_dim = embed_dim

        if in_channels is not None:
            self.input_encoder = Linear(in_channels, self.embed_dim)
        else:
            self.input_encoder = nn.Identity()

        self.attention = MultiHeadAttention(embed_dim,
                                            num_heads,
                                            axis='nodes',
                                            dropout=dropout,
                                            bias=bias,
                                            device=device,
                                            dtype=dtype)

    def forward(self,
                x,
                attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True):
        """"""
        # x: [batch, steps, nodes, in_channels]
        x = self.input_encoder(x)  # -> [batch, steps, nodes, embed_dim]
        return self.attention(x,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=need_weights)
