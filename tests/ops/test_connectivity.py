# %%
import numpy as np
import torch
import torch_sparse
from torch_geometric.utils import is_undirected
from torch_sparse import SparseTensor

from tsl.ops.connectivity import (adj_to_edge_index,
                                  convert_torch_connectivity,
                                  edge_index_to_adj, infer_backend,
                                  maybe_num_nodes, normalize_connectivity,
                                  parse_connectivity, transpose)
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
