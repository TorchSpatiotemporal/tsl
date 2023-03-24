#%%
import numpy as np
import torch
from torch_geometric.utils import is_undirected

import tsl.ops.connectivity as C
from tsl.ops.graph_generators import build_circle_graph

num_nodes = 30
_, edge_index, edge_weight = build_circle_graph(num_nodes)
edge_index, edge_weight = C.parse_connectivity((edge_index, edge_weight), 'edge_index')
adj_t = C.parse_connectivity((edge_index, edge_weight), 'sparse')

assert not is_undirected(edge_index)
#%%

def test_convert_connectivity():
    ei, ew = C.convert_torch_connectivity(adj_t, 'edge_index')
    a_ = C.convert_torch_connectivity((ei, ew), 'sparse')
    assert ew is None
    assert torch.allclose(adj_t.to_dense(), a_.to_dense())


def _test_normalize_connectivity(norm):
    ei, ew = C.normalize_connectivity(edge_index, edge_weight, norm, edge_index.max() + 1)
    a, _ = C.normalize_connectivity(adj_t, None, norm, edge_index.max() + 1)
    ei_, ew_ = C.convert_torch_connectivity(a, 'edge_index')
    a_ = C.convert_torch_connectivity((ei, ew), 'sparse')
    if ew_ is not None:
        assert torch.allclose(ew, ew_)
    else:
        torch.allclose(ew, torch.ones_like(ew))
    assert torch.allclose(a.to_dense(), a_.to_dense())


def test_normalize_connectivity():
    norms = ['mean', 'sym', 'asym', 'none', None]
    for n in norms:
        _test_normalize_connectivity(n)