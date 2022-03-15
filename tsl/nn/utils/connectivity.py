from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch import Tensor
from torch_geometric.utils import dense_to_sparse, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes as pyg_num_nodes

from tsl.typing import TensArray, OptTensArray


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, np.ndarray):
        return int(edge_index.max()) + 1 if edge_index.size > 0 else 0
    else:
        return pyg_num_nodes(edge_index, num_nodes)


def adj_to_edge_index(adj: TensArray) -> Tuple[TensArray, TensArray]:
    if isinstance(adj, Tensor):
        return dense_to_sparse(adj)
    else:
        idxs = np.nonzero(adj)
        edge_index = np.stack(idxs)
        edge_weights = adj[idxs]
        return edge_index, edge_weights


def edge_index_to_adj(edge_index: TensArray,
                      edge_weights: OptTensArray = None,
                      num_nodes: Optional[int] = None) -> TensArray:
    N = maybe_num_nodes(edge_index, num_nodes)
    if isinstance(edge_index, Tensor):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32,
                                      device=edge_index.device)
        adj = torch.zeros((N, N), dtype=edge_weights.dtype,
                          device=edge_weights.device)
    else:
        if edge_weights is None:
            edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)
        adj = np.zeros((N, N), dtype=edge_weights.dtype)
    adj[edge_index[0], edge_index[1]] = edge_weights
    return adj


def transpose(edge_index: TensArray, edge_weights: OptTensArray = None) \
        -> Union[TensArray, Tuple[TensArray, TensArray]]:
    if edge_weights is not None:
        return edge_index[[1, 0]], edge_weights
    return edge_index[[1, 0]]


def weighted_degree(index: TensArray, weights: OptTensArray = None,
                    num_nodes: Optional[int] = None) -> TensArray:
    r"""Computes the weighted degree of a given one-dimensional index tensor.

    Args:
        index (LongTensor): Index tensor.
        weights (Tensor): Edge weights tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    N = maybe_num_nodes(index, num_nodes)
    if isinstance(index, Tensor):
        if weights is None:
            weights = torch.ones((index.size(0),),
                                 device=index.device, dtype=torch.int)
        out = torch.zeros((N,), dtype=weights.dtype, device=weights.device)
        out.scatter_add_(0, index, weights)
    else:
        if weights is None:
            weights = np.ones(index.shape[0], dtype=np.int)
        out = np.zeros(N, dtype=weights.dtype)
        np.add.at(out, index, weights)
    return out


def normalize(edge_index: TensArray, edge_weights: OptTensArray = None,
              dim: int = 0, num_nodes: Optional[int] = None) \
        -> Tuple[TensArray, TensArray]:
    r"""Normalize edge weights across dimension :obj:`dim`.

    .. math::
        e_{i,j} =  \frac{e_{i,j}}{deg_{i}\ \text{if dim=0 else}\ deg_{j}}

    Args:
        edge_index (LongTensor): Edge index tensor.
        edge_weights (Tensor): Edge weights tensor.
        dim (int): Dimension over which to compute normalization.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    index = edge_index[dim]
    if edge_weights is None:
        if isinstance(edge_index, Tensor):
            edge_weights = torch.ones(edge_index.size(1), dtype=torch.int,
                                      device=edge_index.device)
        else:
            edge_weights = np.ones(edge_index.shape[1], dtype=np.int)
    degree = weighted_degree(index, edge_weights, num_nodes=num_nodes)
    return edge_index, edge_weights / degree[index]


def power_series(edge_index: TensArray, edge_weights: OptTensArray = None,
                 k: int = 2, num_nodes: Optional[int] = None) \
        -> Tuple[TensArray, TensArray]:
    r"""Compute order :math:`k` power series of sparse adjacency matrix
    (:math:`A^k`).

    Args:
        edge_index (LongTensor): Edge index tensor.
        edge_weights (Tensor): Edge weights tensor.
        k (int): Order of power series.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    if isinstance(edge_index, Tensor):
        coo = to_scipy_sparse_matrix(edge_index, edge_weights, N)
        coo = coo ** k
        return from_scipy_sparse_matrix(coo)
    else:
        if edge_weights is None:
            edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)
        coo = coo_matrix((edge_weights, tuple(edge_index)), (N, N))
        coo = (coo ** k).tocoo()
        return np.stack([coo.row, coo.col], 0).astype(np.int64), coo.data
