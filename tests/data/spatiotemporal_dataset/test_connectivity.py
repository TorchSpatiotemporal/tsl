"""Test connectivity (layouts, n_edges, presence in the item).
"""
import numpy as np
import torch
from torch_sparse import SparseTensor

from tsl.data import SpatioTemporalDataset

from .helpers import _grid_target


N_NODES = 4


def _dataset(connectivity=None):
    target = _grid_target(30, N_NODES, 1)
    return SpatioTemporalDataset(target=target, connectivity=connectivity,
                                 window=4, horizon=2)


def _dense_adj():
    adj = np.zeros((N_NODES, N_NODES), dtype='float32')
    adj[0, 1] = adj[1, 2] = adj[2, 3] = 1.0  # 3 directed edges
    return adj


def test_no_connectivity():
    ds = _dataset(None)
    assert not ds.has_connectivity
    assert ds.n_edges is None
    assert 'edge_index' not in ds.patterns
    assert 'edge_index' not in ds[0].input


def test_edge_index_tuple_layout():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # 4 edges
    edge_weight = torch.tensor([0.5, 0.6, 0.7, 0.8])
    ds = _dataset((edge_index, edge_weight))
    assert ds.has_connectivity
    assert isinstance(ds.edge_index, torch.Tensor)
    assert ds.n_edges == 4
    assert ds.patterns['edge_index'] == '2 e'
    assert ds.patterns['edge_weight'] == 'e'
    # both appear in the item input, unchanged
    item = ds[0]
    np.testing.assert_array_equal(item.input['edge_index'].numpy(),
                                  edge_index.numpy())
    np.testing.assert_array_equal(item.input['edge_weight'].numpy(),
                                  edge_weight.numpy())


def test_dense_adjacency_layout():
    adj = _dense_adj()
    ds = _dataset(adj)
    assert ds.has_connectivity
    # preserved as a dense [N, N] tensor (weights baked in, no separate weight)
    assert isinstance(ds.edge_index, torch.Tensor)
    assert tuple(ds.edge_index.shape) == (N_NODES, N_NODES)
    assert ds.edge_weight is None
    # metadata reflects the dense layout (NOT a COO '2 e')
    assert ds.patterns['edge_index'] == 'n n'
    assert ds.batch_patterns['edge_index'] == 'n n'
    assert ds.n_edges == 3  # nonzero entries, not size(1) == N
    # the dense matrix is delivered to the item intact
    np.testing.assert_array_equal(ds[0].input['edge_index'].numpy(), adj)


def test_sparse_tensor_layout():
    adj = _dense_adj()
    st = SparseTensor.from_dense(torch.as_tensor(adj))
    ds = _dataset(st)
    assert ds.has_connectivity
    assert isinstance(ds.edge_index, SparseTensor)
    assert ds.patterns['edge_index'] == 'n n'
    assert ds.n_edges == 3  # nnz
    assert 'edge_index' in ds[0].input


def test_two_node_corner_dense_vs_coo():
    # the dtype check disambiguates the [2, 2] shape ambiguity on 2-node graphs
    target = _grid_target(30, 2, 1)
    dense = SpatioTemporalDataset(target=target, window=4, horizon=2,
                                  connectivity=np.array([[0., 1.], [1., 0.]],
                                                        dtype='float32'))
    assert dense.patterns['edge_index'] == 'n n' and dense.n_edges == 2
    coo = SpatioTemporalDataset(target=target, window=4, horizon=2,
                                connectivity=(torch.tensor([[0, 1, 1],
                                                            [1, 0, 1]]), None))
    assert coo.patterns['edge_index'] == '2 e' and coo.n_edges == 3
