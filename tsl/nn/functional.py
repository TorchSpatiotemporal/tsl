from typing import Optional

import torch
from torch import Tensor

from tsl.imports import require_optional_dependency
from tsl.nn.utils import broadcast

__all__ = [
    'gated_tanh',
    'sparse_softmax',
    'sparse_multi_head_attention',
]


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

    This function has been adapted from `torch_scatter` to avoid the dependency on the
    `torch_scatter` package.
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


def _get_scatter_function(name):
    """Load a TorchScript scatter function after validating its dependency."""
    require_optional_dependency('torch_scatter', 'torch-scatter')
    from . import _functional_scatter

    return getattr(_functional_scatter, name)


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
        ImportError: If the optional :mod:`torch_scatter` dependency is not
            installed.
    """
    return _get_scatter_function('sparse_softmax')(src, index, ptr, num_nodes, dim)


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

    Raises:
        ImportError: If the optional :mod:`torch_scatter` dependency is not
            installed.
    """
    return _get_scatter_function('sparse_multi_head_attention')(
        q, k, v, index, dim_size, dropout_p
    )
