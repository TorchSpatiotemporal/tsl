"""TorchScript sparse functions backed by the optional torch-scatter package."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import gather_csr, scatter, segment_csr

import tsl
from tsl.nn.utils import broadcast


@torch.jit.script
def sparse_softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = -2,
) -> Tensor:
    """Compute a sparse softmax over broadcast dimensions."""
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        n_nodes = maybe_num_nodes(index, num_nodes)
        expanded_index = broadcast(index, src, dim)
        src_max = scatter(src, expanded_index, dim, dim_size=n_nodes, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, expanded_index, dim, dim_size=n_nodes, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + tsl.epsilon)


@torch.jit.script
def sparse_multi_head_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    index: Tensor,
    dim_size: Optional[int] = None,
    dropout_p: float = 0.0,
):
    """Compute sparse multi-head scaled dot-product attention."""
    dim = 0
    _, n_heads, embedding_size = q.shape
    n_nodes = maybe_num_nodes(index, dim_size)
    alpha = (q * k).sum(dim=-1) / math.sqrt(embedding_size)
    alpha = sparse_softmax(alpha, index, num_nodes=n_nodes, dim=dim)
    if dropout_p > 0.0:
        alpha = F.dropout(alpha, p=dropout_p)
    v *= alpha.view(-1, n_heads, 1)
    out = torch.zeros((n_nodes, n_heads, v.size(2)), dtype=v.dtype, device=v.device)
    add_index = broadcast(index, v, dim)
    out.scatter_add_(dim, add_index, v)
    return out, alpha
