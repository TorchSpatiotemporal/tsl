import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import gather_csr, scatter, segment_csr
from torch_scatter.utils import broadcast

import tsl

__all__ = [
    'expand_then_cat',
    'gated_tanh',
    'sparse_softmax',
    'sparse_multi_head_attention',
]


def expand_then_cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]],
                    dim: int = -1) -> Tensor:
    """Match the dimensions of tensors in the input list and then concatenate.

    Args:
        tensors (list): Tensors to concatenate.
        dim (int): Dimension along which to concatenate.
            (default: -1)
    """
    shapes = [t.shape for t in tensors]
    expand_dims = torch.max(torch.tensor(shapes), 0).values
    expand_dims[dim] = -1
    tensors = [t.expand(*expand_dims) for t in tensors]
    return torch.cat(tensors, dim=dim)


@torch.jit.script
def gated_tanh(input: Tensor, dim: int = -1) -> Tensor:
    r"""The gated tanh unite. Computes:

    .. math ::
        \text{GatedTanh}(a, b) = \text{tanh}(a) \otimes \sigma(b)

    where :attr:`input` is split in half along :attr:`dim` to form :math:`a`
    and :math:`b`, :math:`\text{tanh}` is the hyperbolic tangent function,
    :math:`\sigma` is the sigmoid function and :math:`\otimes` is the
    element-wise product between matrices.

    Args:
        input (Tensor): Input tensor.
        dim (int): Dimension on which the input is split.
            (default: -1)
    """

    out, gate = torch.tensor_split(input, 2, dim=dim)
    return torch.tanh(out) * torch.sigmoid(gate)


@torch.jit.script
def sparse_softmax(src: Tensor,
                   index: Optional[Tensor] = None,
                   ptr: Optional[Tensor] = None,
                   num_nodes: Optional[int] = None,
                   dim: int = -2) -> Tensor:
    r"""Extension of :func:`~torch_geometric.softmax` with index broadcasting
    to compute a sparsely evaluated softmax over multiple broadcast dimensions.

    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (Tensor, optional): The indices of elements for applying the
            softmax.
            (default: :obj:`None`)
        ptr (Tensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, i.e.,
            :obj:`max_val + 1` of :attr:`index`.
            (default: :obj:`None`)
        dim (int): The dimension on which to normalize, i.e., the edge
            dimension.
            (default: :obj:`-2`)
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        expanded_index = broadcast(index, src, dim)
        src_max = scatter(src, expanded_index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, expanded_index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + tsl.epsilon)


@torch.jit.script
def sparse_multi_head_attention(q: Tensor,
                                k: Tensor,
                                v: Tensor,
                                index: Tensor,
                                dim_size: Optional[int] = None,
                                dropout_p: float = 0.):
    r"""Computes multi-head, scaled, dot product attention on query, key and
    value tensors, applying dropout if a probability greater than 0 is
    specified. Index specifies for each query in q the belonging sequence in the
    original batched, dense tensor.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q (Tensor): Query tensor. See Shape section for shape details.
        k (Tensor): Key tensor. See Shape section for shape details.
        v (Tensor): Value tensor. See Shape section for shape details.
        index (Tensor): Tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dim_size (int, optional): The batched target length sequence, i.e.
            :obj:`max_val + 1` of :attr:`index`.
            (default: :obj:`None`)
        dropout_p (float): dropout probability. If greater than 0, then dropout
            is applied.
            (default: 0)

    Shapes:
        q: :math:`(S, H, E)` where S is sparsed dimension, H is the number of
            heads, and E is embedding dimension.
        k: :math:`(S, H, E)` where S is sparsed dimension, H is the number of
            heads, and E is embedding dimension.
        v: :math:`(S, H, O)` where S is sparsed dimension, H is the number of
            heads, and O is output dimension.
        index: :math:`(S)` where S is sparsed dimension.
        dim_size: must be :math:`(B \times Nt)`

        Output: attention values have shape :math:`(B, Nt, E)`; attention
            weights have shape :math:`(S, H)`
    """
    dim = 0
    B, H, E = q.shape
    N = maybe_num_nodes(index, dim_size)
    # scores
    alpha = (q * k).sum(dim=-1) / math.sqrt(E)
    alpha = sparse_softmax(alpha, index, num_nodes=N, dim=dim)
    if dropout_p > 0.0:
        alpha = F.dropout(alpha, p=dropout_p)
    v *= alpha.view(-1, H, 1)
    # out
    out = torch.zeros((N, H, v.size(2)), dtype=v.dtype, device=v.device)
    add_index = broadcast(index, v, dim)
    out.scatter_add_(dim, add_index, v)
    return out, alpha
