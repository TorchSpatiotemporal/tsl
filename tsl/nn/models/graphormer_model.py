import torch.nn as nn
from torch_geometric.typing import OptTensor
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch
import math
from einops import rearrange
from functools import partial
from typing import Optional
from tsl.nn.models.transformer_model import TransformerModel
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.base.centrality_encoding import CentralityEncoding
from tsl.nn.ops.ops import Select
from tsl.nn.blocks.encoders import TransformerLayer, SpatioTemporalTransformerLayer, Transformer
import numpy as np
from tsl.nn.base.attention.graphormer_attention import multi_head_attention_forward
from typing import Tuple
from torch import Tensor

from typing import Optional, Tuple, List
import warnings

from torch._C import _infer_size, _add_docstr
# from torch._C._nn import linear
from torch import Tensor
from torch import nn
from torch.nn.functional import pad, softmax, dropout

def discretize_values(dist):
    matrix = torch.from_numpy(dist)
    nonzero_threshold=np.partition(np.unique(dist.flatten()), 1)[1]
    matrix = matrix.double()
    matrix = torch.where(matrix < nonzero_threshold, 0., matrix)
    matrix = torch.where((matrix < 0.1) & (matrix > nonzero_threshold) , 1., matrix)
    matrix = torch.where((matrix < 0.2) & (matrix > 0.1) , 2., matrix)
    matrix = torch.where((matrix < 0.3) & (matrix > 0.2) , 3., matrix)
    matrix = torch.where((matrix < 0.4) & (matrix > 0.3) , 4., matrix)
    matrix = torch.where((matrix < 0.5) & (matrix > 0.4) , 5., matrix)
    matrix = torch.where((matrix < 0.6) & (matrix > 0.5) , 6., matrix)
    matrix = torch.where((matrix < 0.7) & (matrix > 0.6) , 7., matrix)
    matrix = torch.where((matrix < 0.8) & (matrix > 0.7) , 8., matrix)
    matrix = torch.where((matrix < 0.9) & (matrix > 0.8) , 9., matrix)
    matrix = torch.where((matrix < 1.0) & (matrix > 0.9) , 10., matrix)

    return matrix


import sys

class Graph():
    def __init__(self, graph):
        self.V = graph.shape[0]
        self.graph = graph
 
    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = sys.maxsize
        min_index = -1
 
        # Search not nearest vertex not in the
        # shortest path tree
        for u in range(self.V):
            if dist[u] < min and sptSet[u] == False:
                min = dist[u]
                min_index = u
 
        return min_index
    
    def set_disconnected_nodes_val(self, dist, val = 0):
        min = sys.maxsize
        for u in range(self.V):
            if dist[u] == min:
                dist[u] = val
        return dist 
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src, disconnected_val = 0):
 
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # x is always equal to src in first iteration
            x = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[x] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for y in range(self.V):
                if self.graph[x][y] > 0 and sptSet[y] == False and \
                dist[y] > dist[x] + self.graph[x][y]:
                        dist[y] = dist[x] + self.graph[x][y]

        dist = self.set_disconnected_nodes_val(dist, disconnected_val)
        return dist

linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor
Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
Shape:
    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
""")

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_bias: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_bias :math:
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    if attn_bias is not None:
        attn_bias = attn_bias.repeat(int(attn.shape[0] / attn_bias.shape[0]), 1, 1)
        attn += attn_bias

    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    attn_bias: Optional[Tensor] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_bias, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None

@torch.jit.script
def _get_causal_mask(seq_len: int, diagonal: int = 0,
                     device: Optional[torch.device] = None):
    # mask keeping only previous steps
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    causal_mask = torch.triu(ones, diagonal)
    return causal_mask

class GraphAttnBias(nn.Module):
    def __init__(self,
                 num_heads,
                 max_dist,
                 spd_matrix):
        super(GraphAttnBias, self).__init__()

        self.spatial_pos_encoder = nn.Embedding(max_dist, num_heads, padding_idx=0)
        # self.spatial_pos_encoder.weight = nn.Parameter(self.spatial_pos_encoder.weight.to('cuda'))
        self.spd_matrix = spd_matrix.to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        # (nodes, nodes, heads) -> (heads, nodes, nodes)
        return self.spatial_pos_encoder(self.spd_matrix).permute(2, 0, 1)

class GraphormerMHA(MultiheadAttention):
    def __init__(self, embed_dim, heads,
                 attn_bias: Optional[Tensor],
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
        super(GraphormerMHA, self).__init__(
            embed_dim, heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=False,
            device=device,
            dtype=dtype)

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

        # change projections
        if qdim is not None and qdim != embed_dim:
            self.qdim = qdim
            self.q_proj = nn.Linear(self.qdim, embed_dim)
        else:
            self.qdim = embed_dim
            self.q_proj = nn.Identity()

        self.attn_bias = attn_bias
    
    def forward(self, query: Tensor,
                key: OptTensor = None,
                value: OptTensor = None,
                key_padding_mask: OptTensor = None,
                need_weights: bool = True, attn_mask: OptTensor = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        # inputs: [batches, steps, nodes, channels] -> [s (b n) c]
        if key is None:
            key = query
        if value is None:
            value = key
        batch = value.shape[0]
        query, key, value = [rearrange(x, self._in_pattern)
                                for x in (query, key, value)]

        if self.causal:
            causal_mask = _get_causal_mask(
                key.size(0), diagonal=1, device=query.device)
            if attn_mask is None:
                attn_mask = causal_mask
            else:
                attn_mask = torch.logical_and(attn_mask, causal_mask)
        
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_weights = multi_head_attention_forward(
                self.q_proj(query), key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, attn_bias=self.attn_bias(),
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_weights = multi_head_attention_forward(
                self.q_proj(query), key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, attn_bias=self.attn_bias(),
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            attn_output  = attn_output.transpose(1, 0)

        attn_output = rearrange(attn_output, self._out_pattern, b=batch)\
            .contiguous()
        if attn_weights is not None:
            attn_weights = rearrange(attn_weights, '(b d) l m -> b d l m',
                                     b=batch).contiguous()
        return attn_output, attn_weights

class GraphormerLayer(TransformerLayer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 attn_bias,
                 ff_size=None,
                 n_heads=1,
                 axis='steps',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(GraphormerLayer, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_heads=n_heads,
            axis=axis,
            causal=causal,
            activation=activation,
            dropout=dropout
        )
        
        self.att = GraphormerMHA(embed_dim=hidden_size,
                                 attn_bias = attn_bias,
                                 qdim=input_size,
                                 kdim=input_size,
                                 vdim=input_size,
                                 heads=n_heads,
                                 axis=axis,
                                 causal=causal)
        
class SpatioTemporalGraphormerLayer(SpatioTemporalTransformerLayer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 attn_bias,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(SpatioTemporalGraphormerLayer, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_heads=n_heads,
            causal=causal,
            activation=activation,
            dropout=dropout
        )
        
        """
        self.temporal_att = GraphormerMHA(embed_dim=hidden_size,
                                          qdim=input_size,
                                          kdim=input_size,
                                          vdim=input_size,
                                          attn_bias = attn_bias,
                                          heads=n_heads,
                                          axis='steps',
                                          causal=causal)
        """
        self.spatial_att = GraphormerMHA(embed_dim=hidden_size,
                                         qdim=hidden_size,
                                         kdim=hidden_size,
                                         vdim=hidden_size,
                                         attn_bias = attn_bias,
                                         heads=n_heads,
                                         axis='nodes',
                                         causal=False)
        
class Graphormer(Transformer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 attn_bias,
                 ff_size=None,
                 output_size=None,
                 n_layers=1,
                 n_heads=1,
                 axis='steps',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(Graphormer, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            output_size=output_size,
            n_layers=n_layers,
            n_heads=n_heads,
            axis=axis,
            causal=causal,
            activation=activation,
            dropout=dropout
        )
        self.f = getattr(F, activation)

        if ff_size is None:
            ff_size = hidden_size

        if axis in ['steps', 'nodes']:
            transformer_layer = partial(GraphormerLayer, axis=axis)
        elif axis == 'both':
            transformer_layer = SpatioTemporalGraphormerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        layers = []
        for i in range(n_layers):
            layers.append(transformer_layer(input_size=input_size if i == 0 else hidden_size,
                                            hidden_size=hidden_size,
                                            attn_bias=attn_bias,
                                            ff_size=ff_size,
                                            n_heads=n_heads,
                                            causal=causal,
                                            activation=activation,
                                            dropout=dropout))

        self.net = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

class GraphormerModel(TransformerModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 ff_size,
                 exog_size,
                 horizon,
                 n_nodes,
                 n_heads,
                 n_layers,
                 max_in_degree,
                 max_out_degree,
                 max_dist,
                 in_degree_list,
                 out_degree_list,
                 spd_matrix,
                 dropout,
                 axis,
                 activation='gelu'):
        super(GraphormerModel, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            ff_size=ff_size,
            exog_size=exog_size,
            horizon=horizon,
            n_nodes=n_nodes,
            n_heads=n_heads,
            n_layers=n_layers,
            max_in_degree = max_in_degree,
            max_out_degree = max_out_degree,
            in_degree_list = in_degree_list,
            out_degree_list = out_degree_list,
            dropout=dropout,
            axis=axis,
            activation=activation)
        
        assert in_degree_list.shape[0] == out_degree_list.shape[0], \
            'the number of in_degrees and out_degrees must be equal'
        
        _attn_bias = GraphAttnBias(n_heads, max_dist, spd_matrix)

        self.transformer_encoder = nn.Sequential(
            Graphormer(input_size=hidden_size,
                       hidden_size=hidden_size,
                       attn_bias=_attn_bias,
                       ff_size=ff_size,
                       n_heads=n_heads,
                       n_layers=n_layers,
                       activation=activation,
                       dropout=dropout,
                       axis=axis),
            Select(1, -1)
        )

        self.sge = StaticGraphEmbedding(n_nodes, hidden_size)
        self.ce = CentralityEncoding(
            hidden_size, max_in_degree, max_out_degree, in_degree_list, out_degree_list)
        
    def forward(self, x, u=None, **kwargs):
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        b, *_ = x.size()
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s f -> b s 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        x = self.pe(x) + self.ce() + self.sge()
        x = self.transformer_encoder(x)

        return self.readout(x)