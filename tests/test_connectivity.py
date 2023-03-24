#%%
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.utils import is_undirected

import tsl.ops.connectivity as C
from tsl.ops.graph_generators import build_circle_graph

num_nodes = 30
_, edge_index, edge_weight = build_circle_graph(num_nodes)
#%%
skip_edges = np.copy(edge_index[:, :-1:2])
skip_edges[1, :] += 1
num_nodes += 1
edge_index = np.concatenate([edge_index, skip_edges], 1)
edge_index, edge_weight = C.parse_connectivity((edge_index, edge_weight), 'edge_index', num_nodes=num_nodes)
adj_t = C.parse_connectivity((edge_index, edge_weight), 'sparse', num_nodes=num_nodes)

assert not is_undirected(edge_index)

#%%
def test_convert_connectivity():
    ei, ew = C.convert_torch_connectivity(adj_t, 'edge_index', num_nodes=num_nodes)
    a_ = C.convert_torch_connectivity((ei, ew), 'sparse', num_nodes=num_nodes)
    assert ew is None
    assert torch.allclose(adj_t.to_dense(), a_.to_dense())


def _test_normalize_connectivity(norm):
    ei, ew = C.normalize_connectivity(edge_index, edge_weight, norm, num_nodes)
    a_ = C.convert_torch_connectivity((ei, ew), 'sparse', num_nodes=num_nodes)
    a, _ = C.normalize_connectivity(adj_t, None, norm, num_nodes)
    assert torch.allclose(a.to_dense(), a_.to_dense())


def test_normalize_connectivity():
    norms = ['mean', 'sym', 'asym', 'none', 'gcn', None]
    for n in norms:
        _test_normalize_connectivity(n)