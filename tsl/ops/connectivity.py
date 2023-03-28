from types import ModuleType
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch_sparse
from scipy.sparse import coo_matrix
from torch import Tensor
from torch_geometric.data.storage import recursive_apply
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops, subgraph
from torch_sparse import SparseTensor, fill_diag

from tsl.typing import (DataArray, OptTensArray, ScipySparseMatrix,
                        SparseTensArray, TensArray, TorchConnectivity)
from tsl.utils import casting


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, np.ndarray):
        return int(edge_index.max()) + 1 if edge_index.size > 0 else 0
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def infer_backend(obj, backend: ModuleType = None):
    if backend is not None:
        return backend
    elif isinstance(obj, Tensor):
        return torch
    elif isinstance(obj, np.ndarray):
        return np
    elif isinstance(obj, SparseTensor):
        return torch_sparse
    raise RuntimeError(f"Cannot infer valid backed from {type(obj)}.")


def convert_torch_connectivity(connectivity: TorchConnectivity,
                               target_layout: str,
                               input_layout: Optional[str] = None,
                               num_nodes: Optional[int] = None):
    formats = [None, 'dense', 'sparse', 'edge_index']
    assert input_layout in formats and target_layout in formats[1:]

    edge_attr = None

    # infer input_layout from data
    if input_layout is None:
        if isinstance(connectivity, SparseTensor):
            input_layout = 'sparse'
        elif isinstance(connectivity, (list, tuple)):
            connectivity, edge_attr = connectivity
        if isinstance(connectivity, Tensor):
            if connectivity.size(0) == connectivity.size(1):
                if connectivity.size(0) == 2 and connectivity.ndim == 2:
                    raise RuntimeError("Cannot infer input_format from [2, 2] "
                                       "connectivity matrix.")
                input_layout = 'dense'
            elif connectivity.size(0) == 2 and connectivity.ndim == 2:
                input_layout = 'edge_index'
                connectivity = (connectivity, edge_attr)
    # if input_layout is still None, it cannot be inferred
    if input_layout is None:
        raise RuntimeError("Cannot infer input_format from connectivity.")

    # check input_layout matches data
    if input_layout == 'dense':
        assert isinstance(connectivity, Tensor) \
               and connectivity.size(-2) == connectivity.size(-1)
    elif input_layout == 'sparse':
        assert isinstance(connectivity, SparseTensor) \
               and connectivity.dim() == 2
    elif input_layout == 'edge_index':
        if isinstance(connectivity, (list, tuple)):
            connectivity, edge_attr = connectivity
        assert isinstance(connectivity, Tensor) \
               and connectivity.size(0) == 2 and connectivity.ndim == 2
        connectivity = (connectivity, edge_attr)

    if input_layout == target_layout:
        return connectivity

    # edge index -> sparse tensor
    if input_layout == 'edge_index' and target_layout == 'sparse':
        edge_index, edge_attr = connectivity
        return SparseTensor.from_edge_index(edge_index, edge_attr,
                                            (num_nodes, num_nodes)).t()
    # edge index -> dense tensor
    if input_layout == 'edge_index' and target_layout == 'dense':
        edge_index, edge_weights = connectivity
        return edge_index_to_adj(edge_index, edge_weights, num_nodes=num_nodes)
    # dense tensor -> sparse tensor
    if input_layout == 'dense' and target_layout == 'sparse':
        return SparseTensor.from_dense(connectivity)
    # dense tensor -> edge index
    if input_layout == 'dense' and target_layout == 'edge_index':
        return adj_to_edge_index(connectivity)
    # sparse tensor -> dense tensor
    if input_layout == 'sparse' and target_layout == 'dense':
        return connectivity.to_dense()
    # sparse tensor -> edge index
    if input_layout == 'sparse' and target_layout == 'edge_index':
        row, col, edge_attr = connectivity.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, edge_attr


def adj_to_edge_index(adj: TensArray, backend: ModuleType = None) \
        -> Tuple[TensArray, TensArray]:
    """Convert adjacency matrix from dense layout to (:obj:`edge_index`,
    :obj:`edge_weight`) tuple. The input adjacency matrix is transposed before
    conversion.

    Args:
        adj (TensArray): dense adjacency matrix as :class:`~torch.Tensor` or
            :class:`~numpy.ndarray`.
        backend (ModuleType, optional): backend matching :obj:`adj` type (either
            :mod:`numpy` or :mod:`torch`), if :obj:`None` it is inferred from
            :obj:`adj` type.
            (default :obj:`None`)

    Returns:
        tuple: (:obj:`edge_index`, :obj:`edge_weight`) tuple of same type of
            :obj:`adj` (:class:`~torch.Tensor` or :class:`~numpy.ndarray`).
    """
    backend = infer_backend(adj, backend)

    assert backend in [torch, np]
    assert 2 <= adj.ndim <= 3
    assert adj.shape[-1] == adj.shape[-2]

    if backend is torch:
        adj = torch.transpose(adj, -2, -1)
        index = adj.nonzero(as_tuple=True)
    else:
        adj = np.swapaxes(adj, -2, -1)  # transpose
        index = adj.nonzero()

    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.shape[-1]
        index = (batch + index[1], batch + index[2])

    edge_index = backend.stack(index, 0)

    return edge_index, edge_attr


def edge_index_to_adj(edge_index: TensArray,
                      edge_weights: OptTensArray = None,
                      num_nodes: Optional[int] = None) -> TensArray:
    N = maybe_num_nodes(edge_index, num_nodes)
    if isinstance(edge_index, Tensor):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1),
                                      dtype=torch.float32,
                                      device=edge_index.device)
        adj = torch.zeros((N, N),
                          dtype=edge_weights.dtype,
                          device=edge_weights.device)
    else:
        if edge_weights is None:
            edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)
        adj = np.zeros((N, N), dtype=edge_weights.dtype)
    adj[edge_index[0], edge_index[1]] = edge_weights
    return adj.T


def transpose(edge_index: SparseTensArray, edge_weights: OptTensArray = None) \
        -> Union[TensArray, Tuple[TensArray, TensArray]]:
    if isinstance(edge_index, SparseTensor):
        return edge_index.t()
    if edge_weights is not None:
        return edge_index[[1, 0]], edge_weights
    return edge_index[[1, 0]]


def reduce_graph(subset: Union[Tensor, List[int]],
                 edge_index: SparseTensArray,
                 num_nodes: Optional[int] = None,
                 backend: ModuleType = None) \
        -> Tuple[TensArray, OptTensArray]:
    """Returns the subgraph with all nodes in :attr:`subset` and only the edges
    between them.

    Args:
        subset: The index of the nodes in the output subgraph.
        edge_index: Adjacency matrix as COO :obj:`edge_index` or
            :class:`torch_sparse.SparseTensor`.
        num_nodes: The number of nodes.
            (default: :obj:`None`)
        backend (ModuleType, optional): Backend matching :obj:`edge_index` type
            (either :mod:`numpy` or :mod:`torch`), if :obj:`None` it is inferred
            from :obj:`edge_index` type.
            (default :obj:`None`)

    Returns:
        tuple: edge_index, edge_mask
    """
    backend = infer_backend(edge_index, backend)
    if backend is torch:
        edge_index, _, edge_mask = subgraph(subset,
                                            edge_index,
                                            return_edge_mask=True,
                                            relabel_nodes=True,
                                            num_nodes=num_nodes)
        return edge_index, edge_mask
    else:
        edge_index = edge_index[subset, subset]
        return edge_index, None


def weighted_degree(index: TensArray,
                    weights: OptTensArray = None,
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
            weights = torch.ones((index.size(0), ),
                                 device=index.device,
                                 dtype=torch.int)
        out = torch.zeros((N, ), dtype=weights.dtype, device=weights.device)
        out.scatter_add_(0, index, weights)
    else:
        if weights is None:
            weights = np.ones(index.shape[0], dtype=np.int)
        out = np.zeros(N, dtype=weights.dtype)
        np.add.at(out, index, weights)
    return out


def asymmetric_norm(edge_index: SparseTensArray,
                    edge_weight: OptTensArray = None,
                    dim: int = 0, num_nodes: Optional[int] = None,
                    add_self_loops: bool = False) \
        -> Tuple[SparseTensArray, OptTensArray]:
    r"""Normalize edge weights across dimension :obj:`dim`.

    .. math::
        e_{i,j} =  \frac{e_{i,j}}{deg_{i}\ \text{if dim=0 else}\ deg_{j}}

    Args:
        edge_index (LongTensor): Edge index tensor.
        edge_weight (Tensor): Edge weights tensor.
        dim (int): Dimension over which to compute normalization.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        add_self_loops: Whether to add self loops to the adjacency matrix.
    """
    backend = infer_backend(edge_index)

    if backend is torch_sparse:
        assert edge_weight is None
        if add_self_loops:
            edge_index = fill_diag(edge_index, 1)
        deg = edge_index.sum(dim=dim).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        edge_index = deg_inv.view(-1, 1) * edge_index
        return edge_index, None

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    index = edge_index[dim]
    degree = weighted_degree(index, edge_weight, num_nodes=num_nodes)
    norm_weight = (1 if edge_weight is None else edge_weight) / degree[index]
    return edge_index, norm_weight


def normalize_connectivity(edge_index, edge_weight, norm,
                           num_nodes) -> Tuple[Adj, OptTensor]:
    if norm == 'sym':
        norm_edge_index = gcn_norm(edge_index,
                                   edge_weight,
                                   num_nodes,
                                   add_self_loops=False)
        if isinstance(edge_index, SparseTensor):
            return norm_edge_index, None
        return norm_edge_index
    elif (norm == 'asym') or (norm == 'mean'):
        return asymmetric_norm(edge_index,
                               edge_weight,
                               dim=1,
                               num_nodes=num_nodes,
                               add_self_loops=False)
    elif norm == 'gcn':
        norm_edge_index = gcn_norm(edge_index,
                                   edge_weight,
                                   num_nodes,
                                   add_self_loops=True)
        if isinstance(edge_index, SparseTensor):
            return norm_edge_index, None
        return norm_edge_index
    elif (norm == 'none') or (norm is None):
        if (edge_weight is None) and not isinstance(edge_index, SparseTensor):
            edge_weight = torch.ones((edge_index.size(1), ),
                                     device=edge_index.device)
        return edge_index, edge_weight
    else:
        raise NotImplementedError(f'Normalization {norm} not implemented.')


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
        from torch_geometric.utils import (from_scipy_sparse_matrix,
                                           to_scipy_sparse_matrix)
        coo = to_scipy_sparse_matrix(edge_index, edge_weights, N)
        coo = coo**k
        return from_scipy_sparse_matrix(coo)
    else:
        if edge_weights is None:
            edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)
        coo = coo_matrix((edge_weights, tuple(edge_index)), (N, N))
        coo = (coo**k).tocoo()
        return np.stack([coo.row, coo.col], 0).astype(np.int64), coo.data


def get_dummy_edge_index(dummy: str,
                         num_nodes: int,
                         edge_prob: float = 0.1,
                         directed: bool = True,
                         device=None):
    r"""Create an edge index corresponding to a certain dummy connectivity
    (e.g., full graph).

    Args:
        dummy (str): The dummy connectivity, can be one of :obj:`"identity"`
            (`A`=`I`), :obj:`"full"` (`A = np.ones(N, N)`), :obj:`"random"` or
            `:obj:`"none"` (returns :obj:`None`).
        num_nodes (int): Number of nodes in the graph.
        edge_prob (float): Edge probability for the random graph.
            (default :obj:`0.1`)
        directed (bool): Whether to generate a directed/undirected graph.
            (default :obj:`True`)
        device (optional): Device for the created tensor.
            (default :obj:`None`)
    """
    if dummy == 'identity':
        nodes = torch.arange(num_nodes, device=device)
        edge_index = torch.stack([nodes, nodes])
    elif dummy == 'random':
        from torch_geometric.utils import erdos_renyi_graph
        edge_index = erdos_renyi_graph(num_nodes, edge_prob,
                                       directed=directed).to(device)
    elif dummy == 'full':
        nodes = torch.arange(num_nodes, device=device)
        edge_index = torch.cartesian_prod(nodes, nodes).T
    elif dummy == 'none':
        edge_index = None
    else:
        raise NotImplementedError
    return edge_index


def parse_connectivity(
        connectivity: Union[SparseTensArray, Tuple[DataArray]],
        target_layout: Optional[str] = None,
        num_nodes: Optional[int] = None
) -> Tuple[Optional[Adj], Optional[Tensor]]:
    # Convert to torch
    # from np.ndarray, pd.DataFrame or torch.Tensor
    if isinstance(connectivity, (pd.DataFrame, np.ndarray, Tensor)):
        connectivity = casting.copy_to_tensor(connectivity)
    elif isinstance(connectivity, (list, tuple)):
        connectivity = recursive_apply(connectivity, casting.copy_to_tensor)
    # from scipy sparse matrix
    elif isinstance(connectivity, ScipySparseMatrix.__args__):
        connectivity = SparseTensor.from_scipy(connectivity)
    elif not isinstance(connectivity, SparseTensor):
        raise TypeError("`connectivity` must be a dense matrix or in "
                        "COO format (i.e., an `edge_index`).")

    if target_layout is not None:
        connectivity = convert_torch_connectivity(connectivity,
                                                  target_layout,
                                                  num_nodes=num_nodes)
    return connectivity
