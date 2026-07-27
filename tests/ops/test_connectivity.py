# %%
import numpy as np
import torch
import torch_sparse
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor

import pandas as pd
import pytest
from scipy.sparse import coo_matrix

from tsl.ops.connectivity import (adj_to_edge_index, asymmetric_norm,
                                  convert_torch_connectivity,
                                  edge_index_to_adj, get_dummy_edge_index,
                                  infer_backend, maybe_num_nodes,
                                  normalize_connectivity, parse_connectivity,
                                  power_series, reduce_graph, transpose,
                                  weighted_degree)
from tsl.ops.graph_generators import build_circle_graph

num_nodes = 30
_, edge_index, edge_weight = build_circle_graph(num_nodes)
# %%
skip_edges = np.copy(edge_index[:, :-1:2])
skip_edges[1, :] += 1
num_nodes += 1
edge_index = np.concatenate([edge_index, skip_edges], 1)
edge_index, edge_weight = parse_connectivity((edge_index, edge_weight),
                                             'edge_index',
                                             num_nodes=num_nodes)
adj_t = parse_connectivity((edge_index, edge_weight),
                           'sparse',
                           num_nodes=num_nodes)

assert not is_undirected(edge_index)


def _test_normalize_connectivity(norm):
    ei, ew = normalize_connectivity(edge_index, edge_weight, norm, num_nodes)
    a_ = convert_torch_connectivity((ei, ew), 'sparse', num_nodes=num_nodes)
    a, _ = normalize_connectivity(adj_t, None, norm, num_nodes)
    assert torch.allclose(a.to_dense(), a_.to_dense())


def test_normalize_dense():
    a_ = convert_torch_connectivity((edge_index, edge_weight),
                                    'dense',
                                    num_nodes=num_nodes)
    deg_inv = 1. / a_.sum(1)
    deg_inv[deg_inv == float('inf')] = 0.
    a_ = deg_inv.view(-1, 1) * a_
    a, _ = normalize_connectivity(adj_t, None, 'mean', num_nodes)
    assert torch.allclose(a.to_dense(), a_)


def test_normalize_connectivity():
    norms = ['mean', 'sym', 'asym', 'none', 'gcn', None]
    for n in norms:
        _test_normalize_connectivity(n)


def test_maybe_num_nodes():
    # Test for numpy input
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert maybe_num_nodes(edge_index) == 3

    # Test for torch.Tensor input
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert maybe_num_nodes(edge_index) == 3

    # Test for empty input
    edge_index = np.array([], dtype=np.int64)
    assert maybe_num_nodes(edge_index) == 0

    # Test for num_nodes not None
    assert maybe_num_nodes(edge_index, num_nodes=10) == 10

    # Test for invalid input
    try:
        maybe_num_nodes(None)
    except AttributeError:
        pass


def test_infer_backend():
    # Test for torch.Tensor input
    x = torch.tensor([1, 2, 3])
    assert infer_backend(x) == torch

    # Test for np.ndarray input
    x = np.array([1, 2, 3])
    assert infer_backend(x) == np

    # Test for SparseTensor input
    x = SparseTensor(row=torch.tensor([0, 1]),
                     col=torch.tensor([1, 0]),
                     value=torch.tensor([1, 2]))
    assert infer_backend(x) == torch_sparse

    # Test for invalid input
    try:
        infer_backend(None)
    except RuntimeError:
        pass


def test_convert_torch_connectivity():
    dense = torch.eye(4) / 2.
    sparse = SparseTensor.from_dense(dense)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    edge_attr = torch.ones(4) / 2
    edge_index_attr = (edge_index, edge_attr)

    # Test dense -> sparse
    converted = convert_torch_connectivity(dense, 'sparse')
    assert isinstance(converted, SparseTensor)
    assert torch.allclose(converted.to_dense(), dense)

    # Test dense -> edge_index
    _, converted = convert_torch_connectivity(dense, 'edge_index')
    assert torch.allclose(converted, edge_attr)

    # Test sparse -> dense
    converted = convert_torch_connectivity(sparse, 'dense')
    assert torch.allclose(converted, dense)

    # Test sparse -> edge_index
    _, converted = convert_torch_connectivity(sparse, 'edge_index')
    assert torch.allclose(converted, edge_attr)

    # Test edge_index -> sparse
    converted = convert_torch_connectivity(edge_index_attr,
                                           'sparse',
                                           input_layout='edge_index',
                                           num_nodes=4)
    assert isinstance(converted, SparseTensor)
    assert torch.allclose(converted.to_dense(), dense)

    # Test edge_index -> dense
    converted = convert_torch_connectivity(edge_index_attr,
                                           'dense',
                                           input_layout='edge_index',
                                           num_nodes=4)
    assert torch.allclose(converted, dense)

    # Test invalid input_layout
    try:
        convert_torch_connectivity(edge_index_attr,
                                   'sparse',
                                   input_layout='invalid',
                                   num_nodes=5)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected an AssertionError")

    # Test invalid target_layout
    try:
        convert_torch_connectivity(edge_index_attr,
                                   'invalid',
                                   input_layout='edge_index',
                                   num_nodes=5)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected an AssertionError")

    # Test unable to infer input_layout
    try:
        convert_torch_connectivity(torch.eye(2, 2), 'sparse')
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected a RuntimeError")


def test_adj_to_edge_index():
    # Test for converting adjacency matrix to edge index
    adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float)
    expected_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]],
                                  dtype=torch.long)
    expected_attr = torch.tensor([1., 1., 1., 1.], dtype=torch.float)

    index, attr = adj_to_edge_index(adj)

    assert torch.allclose(index, expected_index)
    assert torch.allclose(attr, expected_attr)

    # Test for converting a batch of adjacency matrices to edge index
    batch_adj = torch.stack([adj, adj], dim=0)
    expected_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long)
    expected_attr = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1.],
                                 dtype=torch.float)

    index, attr = adj_to_edge_index(batch_adj)

    assert torch.allclose(index, expected_index)
    assert torch.allclose(attr, expected_attr)

    # Test for converting a batch of adjacency matrices to edge index with
    # backend set to numpy
    batch_adj_np = batch_adj.numpy()
    expected_index_np = expected_index.numpy()
    expected_attr_np = expected_attr.numpy()

    index_np, attr_np = adj_to_edge_index(batch_adj_np, backend=np)

    assert np.allclose(index_np, expected_index_np)
    assert np.allclose(attr_np, expected_attr_np)


def test_edge_index_to_adj():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weights = torch.tensor([0.5, 1, 0.3])
    adj = edge_index_to_adj(edge_index, edge_weights, 3)
    expected_adj = torch.tensor([[0, 0.5, 0], [0, 0, 1], [0.3, 0, 0]]).T
    assert torch.allclose(adj, expected_adj)


def test_transpose():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    expected_edge_index = torch.tensor([[1, 2, 0], [0, 1, 2]])
    assert torch.allclose(transpose(edge_index), expected_edge_index)
    # Test for sparse tensor
    adj = SparseTensor(row=torch.tensor([0, 1, 2]),
                       col=torch.tensor([1, 2, 0]))
    expected_adj = SparseTensor(row=torch.tensor([1, 2, 0]),
                                col=torch.tensor([0, 1, 2]))
    assert torch.allclose(transpose(adj).to_dense(), expected_adj.to_dense())
    # Test for np.ndarray input
    edge_index_np = edge_index.numpy()
    expected_edge_index_np = expected_edge_index.numpy()
    assert np.allclose(transpose(edge_index_np), expected_edge_index_np)


def test_transpose_with_weights():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    weights = torch.tensor([1., 2., 3.])
    ei, ew = transpose(edge_index, weights)
    assert torch.allclose(ei, torch.tensor([[1, 2, 0], [0, 1, 2]]))
    # weights are passed through unchanged
    assert torch.allclose(ew, weights)


def test_maybe_num_nodes_sparse_tensor():
    adj = SparseTensor(row=torch.tensor([0, 1, 2]),
                       col=torch.tensor([1, 2, 0]),
                       sparse_sizes=(3, 3))
    assert maybe_num_nodes(adj) == 3


def test_edge_index_to_adj_numpy_default_weights():
    edge_index = np.array([[0, 1], [1, 0]])
    adj = edge_index_to_adj(edge_index)
    assert isinstance(adj, np.ndarray)
    np.testing.assert_array_equal(adj, np.array([[0., 1.], [1., 0.]]))


def test_convert_torch_connectivity_same_layout_passthrough():
    adj = SparseTensor.from_dense(torch.eye(3))
    out = convert_torch_connectivity(adj, 'sparse', input_layout='sparse')
    assert out is adj


def test_weighted_degree_torch():
    index = torch.tensor([0, 1, 1, 2])
    weights = torch.tensor([1., 2., 3., 4.])
    out = weighted_degree(index, weights, num_nodes=3)
    assert torch.allclose(out, torch.tensor([1., 5., 4.]))


def test_weighted_degree_torch_default_weights():
    index = torch.tensor([0, 1, 1, 2])
    out = weighted_degree(index, num_nodes=3)
    # unit weights -> plain node degree
    assert torch.allclose(out, torch.tensor([1, 2, 1], dtype=out.dtype))


def test_weighted_degree_numpy():
    index = np.array([0, 1, 1, 2])
    weights = np.array([1., 2., 3., 4.])
    out = weighted_degree(index, weights, num_nodes=3)
    np.testing.assert_array_equal(out, np.array([1., 5., 4.]))


def test_weighted_degree_numpy_default_weights():
    index = np.array([0, 1, 1, 2])
    out = weighted_degree(index, num_nodes=3)
    # unit weights -> plain node degree
    np.testing.assert_array_equal(out, np.array([1, 2, 1]))


def test_reduce_graph_torch():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    sub_edge_index, edge_mask = reduce_graph(torch.tensor([0, 1]), edge_index,
                                             num_nodes=3)
    # only edge (0, 1) survives and nodes are relabeled
    assert torch.allclose(sub_edge_index, torch.tensor([[0], [1]]))
    assert edge_mask.tolist() == [True, False, False]


def test_power_series_torch():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weights = torch.ones(3)
    ei, ew = power_series(edge_index, edge_weights, k=2, num_nodes=3)
    # A^2 of a directed 3-cycle is again a 3-cycle (shifted by two)
    assert ei.shape == (2, 3)
    assert ew.shape[0] == 3


def test_power_series_numpy():
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    ei, ew = power_series(edge_index, None, k=2, num_nodes=3)
    assert isinstance(ei, np.ndarray)
    assert ei.shape == (2, 3)


@pytest.mark.parametrize('dummy,num_nodes,expected_shape', [
    ('identity', 4, (2, 4)),
    ('full', 3, (2, 9)),
])
def test_get_dummy_edge_index_deterministic(dummy, num_nodes, expected_shape):
    edge_index = get_dummy_edge_index(dummy, num_nodes)
    assert tuple(edge_index.shape) == expected_shape


def test_get_dummy_edge_index_identity_values():
    edge_index = get_dummy_edge_index('identity', 3)
    assert torch.allclose(edge_index,
                          torch.tensor([[0, 1, 2], [0, 1, 2]]))


def test_get_dummy_edge_index_none():
    assert get_dummy_edge_index('none', 5) is None


def test_get_dummy_edge_index_random_shape():
    edge_index = get_dummy_edge_index('random', 10, edge_prob=0.5)
    assert edge_index.size(0) == 2


def test_get_dummy_edge_index_invalid():
    with pytest.raises(NotImplementedError):
        get_dummy_edge_index('not_a_dummy', 3)


def test_asymmetric_norm_numpy():
    edge_index = np.array([[0, 0, 1], [1, 2, 2]])
    edge_weight = np.array([1., 1., 1.])
    _, norm_weight = asymmetric_norm(edge_index, edge_weight, dim=1,
                                     num_nodes=3)
    # dim=1: divide by in-degree of target nodes (node 2 has in-degree 2)
    np.testing.assert_allclose(norm_weight, [1., 0.5, 0.5])


def test_asymmetric_norm_sparse():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    adj_t = SparseTensor.from_edge_index(edge_index, torch.ones(3), (3, 3))
    out, ew = asymmetric_norm(adj_t, None, dim=1)
    assert isinstance(out, SparseTensor)
    assert ew is None


def test_parse_connectivity_dataframe():
    df = pd.DataFrame(np.eye(3))
    edge_index, edge_weight = parse_connectivity(df, 'edge_index', num_nodes=3)
    assert isinstance(edge_index, torch.Tensor)
    assert edge_index.shape == (2, 3)


def test_parse_connectivity_scipy_sparse():
    out = parse_connectivity(coo_matrix(np.eye(3)), 'sparse')
    assert isinstance(out, SparseTensor)


def test_parse_connectivity_sparse_tensor_passthrough():
    adj = SparseTensor.from_dense(torch.eye(3))
    out = parse_connectivity(adj, 'sparse')
    assert isinstance(out, SparseTensor)


def test_parse_connectivity_invalid_type_raises():
    with pytest.raises(TypeError):
        parse_connectivity('not connectivity')
