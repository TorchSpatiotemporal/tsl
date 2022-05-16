from typing import Optional

from einops import rearrange
from torch import nn, Tensor
import torch

from torch_geometric.nn.dense import Linear
from torch_geometric.typing import OptTensor

from tsl.nn.layers.positional_encoding import PositionalEncoding
from tsl.nn.utils import get_functional_activation

@torch.jit.script
def _get_causal_mask(seq_len: int, diagonal: int = 0,
                     device: Optional[torch.device] = None):
    # mask keeping only previous steps
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    causal_mask = torch.triu(ones, diagonal)
    return causal_mask


class AttentionEncoder(nn.Module):

    def __init__(self, embed_dim,
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

    def forward(self, query: Tensor,
                key: OptTensor = None,
                value: OptTensor = None):
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

    def __init__(self, embed_dim, heads,
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
                raise ValueError(f'Cannot use causal attention for axis "{axis}".')
            shape = 'n (b s) c'
        else:
            raise ValueError("Axis can either be 'steps' (0) or 'nodes' (1), "
                             f"not '{axis}'.")
        self._in_pattern = f'b s n c -> {shape}'
        self._out_pattern = f'{shape} -> b s n c'
        self.causal = causal
        # Impose batch dimension as the second one
        super(MultiHeadAttention, self).__init__(embed_dim, heads,
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

    def forward(self, query: Tensor,
                key: OptTensor = None,
                value: OptTensor = None,
                key_padding_mask: OptTensor = None,
                need_weights: bool = True,
                attn_mask: OptTensor = None):
        # inputs: [batches, steps, nodes, channels] -> [s (b n) c]
        if key is None:
            key = query
        if value is None:
            value = key
        batch = value.shape[0]
        query, key, value = [rearrange(x, self._in_pattern)
                                for x in (query, key, value)]

        if self.causal:
            causal_mask = _get_causal_mask(key.size(0), diagonal=1, device=query.device)
            if attn_mask is None:
                attn_mask = causal_mask
            else:
                attn_mask = torch.logical_and(attn_mask, causal_mask)
        attn_output, attn_weights = super(MultiHeadAttention,
                                          self).forward(self.q_proj(query),
                                                        key,
                                                        value,
                                                        key_padding_mask,
                                                        need_weights,
                                                        attn_mask)
        attn_output = rearrange(attn_output, self._out_pattern, b=batch)\
            .contiguous()
        if attn_weights is not None:
            attn_weights = rearrange(attn_weights, '(b d) l m -> b d l m',
                                     b=batch).contiguous()
        return attn_output, attn_weights


class GraphormerMHA(nn.Module):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 num_heads):
        super(GraphormerMHA, self).__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be evenly divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention
    
    def forward(self, x, mask=None, return_attention=True):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o