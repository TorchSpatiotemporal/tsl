from typing import Optional

import torch
from torch import Tensor

import tsl
from tsl.nn.utils import broadcast

__all__ = [
    'gated_tanh',
    'sparse_softmax',
    'sparse_multi_head_attention',
]


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
            (default: :obj:`-1`)
    """

    out, gate = torch.tensor_split(input, 2, dim=dim)
    return torch.tanh(out) * torch.sigmoid(gate)


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Scatter sum function.

    This implementation uses native PyTorch scatter operations.
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def sparse_softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = -2,
) -> Tensor:
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

    Raises:
        NotImplementedError: If neither :attr:`index` nor :attr:`ptr` is given.

    Notes:
        When this function is called from a :func:`torch.compile` region,
        provide :attr:`num_nodes` explicitly to avoid deriving an output shape
        from tensor data.
    """
    if index is None:
        if ptr is None:
            raise NotImplementedError
        counts = ptr.diff()
        index = torch.arange(counts.numel(), device=ptr.device)
        index = torch.repeat_interleave(index, counts)

    if num_nodes is None:
        num_nodes = int(index.max()) + 1 if index.numel() > 0 else 0

    expanded_index = broadcast(index, src, dim)
    size = list(src.size())
    size[dim] = num_nodes
    src_max = torch.full(size, -torch.inf, dtype=src.dtype, device=src.device)
    src_max.scatter_reduce_(dim, expanded_index, src, reduce='amax', include_self=True)
    src_max = torch.gather(src_max, dim, expanded_index)
    out = (src - src_max).exp()
    out_sum = torch.zeros(size, dtype=src.dtype, device=src.device)
    out_sum.scatter_add_(dim, expanded_index, out)
    out_sum = torch.gather(out_sum, dim, expanded_index)
    return out / (out_sum + tsl.epsilon)


def sparse_multi_head_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    index: Tensor,
    dim_size: Optional[int] = None,
    dropout_p: float = 0.0,
):
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

    Notes:
        When this function is called from a :func:`torch.compile` region,
        provide :attr:`dim_size` explicitly to avoid deriving an output shape
        from tensor data.

    """
    dim = 0
    _, n_heads, embedding_size = q.shape
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    alpha = (q * k).sum(dim=-1) / embedding_size**0.5
    alpha = sparse_softmax(alpha, index, num_nodes=dim_size, dim=dim)
    if dropout_p > 0.0:
        alpha = torch.nn.functional.dropout(alpha, p=dropout_p)
    weighted_v = v * alpha.view(-1, n_heads, 1)
    out = torch.zeros((dim_size, n_heads, v.size(2)), dtype=v.dtype, device=v.device)
    add_index = broadcast(index, weighted_v, dim)
    out.scatter_add_(dim, add_index, weighted_v)
    return out, alpha
