"""Unit tests for :mod:`tsl.data.data` (StorageView, Data, module helpers).

Tensors use the *grid* discipline -- every cell holds a unique value -- so that
view filtering, rearranging and subgraph results can be matched element-wise.
"""
import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from tsl.data import Data
from tsl.data.data import get_size, pattern_size_repr
from tsl.data.preprocessing.scalers import ScalerModule


# -- builders ---------------------------------------------------------------

def _grid(*shape):
    return torch.arange(int(np.prod(shape))).float().reshape(*shape)


def _graph_data(n_nodes=3, t=4, c=2, scaler=None):
    """A small static graph: window 'x' and horizon 'y' on the nodes plus a
    triangular connectivity."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.tensor([1., 2., 3.])
    transform = {'x': scaler} if scaler is not None else None
    return Data(input={'x': _grid(t, n_nodes, c)},
                target={'y': _grid(2, n_nodes, c)},
                edge_index=edge_index,
                edge_weight=edge_weight,
                transform=transform,
                pattern={'x': 't n c', 'y': 't n c',
                         'edge_index': '2 e', 'edge_weight': 'e'})


# -- get_size / pattern_size_repr -------------------------------------------

def test_get_size_tensor_and_sparse():
    assert get_size(torch.zeros(4, 3)) == (4, 3)
    st = SparseTensor.from_edge_index(torch.tensor([[0, 1], [1, 2]]),
                                      sparse_sizes=(3, 3))
    assert get_size(st) == (3, 3)


def test_pattern_size_repr():
    x = torch.zeros(4, 3)
    assert pattern_size_repr('x', x, 't n') == 'x=[t=4, n=3]'
    assert pattern_size_repr('x', x) == 'x=[4, 3]'


# -- StorageView ------------------------------------------------------------

def test_views_filter_keys():
    d = Data(input={'x': _grid(2), 'u': _grid(2)}, target={'y': _grid(2)})
    assert set(d.input.keys()) == {'x', 'u'}
    assert set(d.target.keys()) == {'y'}


def test_view_getitem_outside_view_raises_keyerror():
    d = Data(input={'x': _grid(2)}, target={'y': _grid(2)})
    # 'y' exists in the shared store but not in the input view
    with pytest.raises(KeyError):
        _ = d.input['y']


def test_view_filter_keys_with_args():
    d = Data(input={'x': _grid(2), 'u': _grid(2)})
    assert set(d.input.keys('x')) == {'x'}
    # an argument outside the view is filtered out
    assert set(d.input.keys('y')) == set()


def test_view_setitem_and_delitem_track_keys():
    d = Data(input={'x': _grid(2)})
    d.input['u'] = _grid(2)
    assert 'u' in set(d.input.keys())
    del d.input['u']
    assert 'u' not in set(d.input.keys())


def test_view_keys_property_drops_missing_mapping_entries():
    d = Data(input={'x': _grid(2), 'u': _grid(2)})
    # drop 'u' from the underlying store directly
    del d._store['u']
    assert set(d.input.keys()) == {'x'}


def test_view_to_dict_is_shallow_copy():
    d = Data(input={'x': _grid(2)})
    dd = d.input.to_dict()
    assert set(dd) == {'x'}
    assert dd['x'] is d.input['x']  # shallow: same tensor object


def test_view_apply_returns_self_and_applies_to_view_keys_only():
    # `apply` over a view returns that view and applies `func` only to the keys
    # the view exposes ('x'); keys not in the view ('y') are left as-is.
    d = Data(input={'x': _grid(2)}, target={'y': _grid(2)})
    ret = d.input.apply(lambda t: t + 100)
    assert ret is d.input
    assert torch.equal(d.input['x'], _grid(2) + 100)
    assert torch.equal(d.target['y'], _grid(2))


def test_views_share_underlying_store():
    # the 'input' and 'target' views are live windows onto a single shared store:
    # the same tensor handed to both keys is the same object read back through
    # either view, and an in-place write through one view is seen through the other
    data = _grid(2)
    d = Data(input={'x': data}, target={'y': data})
    assert d.input['x'] is data
    assert d.target['y'] is data
    assert d.input['x'] is d.target['y']
    d.input.apply_(lambda t: t.add_(100))
    assert torch.equal(d.target['y'], _grid(2) + 100)
    assert torch.equal(d['y'], _grid(2) + 100)  # and the store itself


# -- Data construction ------------------------------------------------------

def test_construction_defaults():
    d = Data()
    assert d.mask is None
    assert d.transform == {}
    assert d.has_mask is False
    assert d.has_transform is False
    assert d.edge_weight is None


def test_mask_and_transform_flags():
    d = Data(input={'x': _grid(2)}, mask=torch.ones(2, dtype=torch.bool),
             transform={'x': ScalerModule(bias=0., scale=1.)})
    assert d.has_mask is True
    assert d.has_transform is True


# -- __cat_dim__ ------------------------------------------------------------

def test_cat_dim_node_dimension():
    d = Data(input={'a': torch.zeros(3, 2)}, pattern={'a': 'n c'})
    assert d.__cat_dim__('a', d['a']) == 0


def test_cat_dim_edge_dimension():
    d = Data(input={'a': torch.zeros(3, 2)}, pattern={'a': 'e c'})
    assert d.__cat_dim__('a', d['a']) == 1 - 1  # index of 'e' == 0


def test_cat_dim_stack_when_no_node_or_edge():
    d = Data(input={'a': torch.zeros(4, 2)}, pattern={'a': 't c'})
    assert d.__cat_dim__('a', d['a']) is None


def test_cat_dim_sparse_double_node():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    st = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))
    d = Data(input={'adj': st}, pattern={'adj': 'n n'})
    assert d.__cat_dim__('adj', st) == (0, 1)


def test_cat_dim_falls_back_when_no_pattern():
    d = Data()
    d['foo'] = torch.zeros(3)
    # no pattern => PyG default (concatenate node-like attrs on dim 0)
    assert d.__cat_dim__('foo', d['foo']) == 0


# -- stores_as --------------------------------------------------------------

def test_stores_as_copies_keys_and_pattern():
    src = _graph_data()
    dst = Data()
    dst.stores_as(src)
    # pattern is copied, not shared
    assert dst.pattern == src.pattern
    dst.pattern['x'] = 'changed'
    assert src.pattern['x'] == 't n c'
    # 'stores_as' primes the view key-tracking; once the store is populated with
    # the same keys the views expose exactly the source's input/target split
    for k in src.keys():
        dst[k] = src[k]
    assert set(dst.input.keys()) == set(src.input.keys())
    assert set(dst.target.keys()) == set(src.target.keys())


# -- rearrange --------------------------------------------------------------

def test_rearrange_element_updates_value_and_pattern():
    d = Data(input={'x': _grid(4, 3)}, pattern={'x': 't n'})
    d.rearrange_element('x', 'n t')
    assert d.pattern['x'] == 'n t'
    assert tuple(d['x'].shape) == (3, 4)


def test_rearrange_explicit_start_mismatch_raises():
    d = Data(input={'x': _grid(4, 3)}, pattern={'x': 't n'})
    with pytest.raises(RuntimeError):
        d.rearrange_element('x', 'c n -> n c')


def test_rearrange_updates_transform_in_lockstep():
    scaler = ScalerModule(bias=torch.zeros(4, 3),
                          scale=_grid(4, 3) + 1, pattern='t n')
    d = Data(input={'x': _grid(4, 3)}, transform={'x': scaler},
             pattern={'x': 't n'})
    d.rearrange_element('x', 'n t')
    assert d.transform['x'].pattern == 'n t'
    assert tuple(d.transform['x'].scale.shape) == (3, 4)


def test_rearrange_multiple():
    d = Data(input={'x': _grid(4, 3), 'u': _grid(4, 3)},
             pattern={'x': 't n', 'u': 't n'})
    d.rearrange({'x': 'n t', 'u': 'n t'})
    assert d.pattern['x'] == 'n t' and d.pattern['u'] == 'n t'


# -- subgraph ---------------------------------------------------------------

def test_subgraph_inplace_reduces_graph():
    d = _graph_data()
    d.subgraph_(torch.tensor([0, 1]))
    assert d.num_nodes == 2
    assert tuple(d['x'].shape) == (4, 2, 2)
    # only the 0->1 edge survives among {0->1, 1->2, 2->0}
    assert d.edge_index.tolist() == [[0], [1]]
    assert d.edge_weight.tolist() == [1.]


def test_subgraph_bool_subset():
    d = _graph_data()
    d.subgraph_(torch.tensor([True, False, True]))
    assert d.num_nodes == 2
    assert tuple(d['x'].shape) == (4, 2, 2)


def test_subgraph_slices_scaler():
    scaler = ScalerModule(bias=torch.zeros(1, 3, 1),
                          scale=_grid(1, 3, 1) + 1, pattern='t n c')
    d = _graph_data(scaler=scaler)
    d.subgraph_(torch.tensor([0, 1]))
    assert tuple(d.transform['x'].scale.shape) == (1, 2, 1)


def test_subgraph_does_not_mutate_original():
    scaler = ScalerModule(bias=torch.zeros(1, 3, 1),
                          scale=_grid(1, 3, 1) + 1, pattern='t n c')
    d = _graph_data(scaler=scaler)
    sub = d.subgraph(torch.tensor([0, 1]))
    # the returned subgraph is reduced ...
    assert tuple(sub['x'].shape) == (4, 2, 2)
    assert tuple(sub.transform['x'].scale.shape) == (1, 2, 1)
    # ... while the original is left fully intact (tensors *and* scaler)
    assert tuple(d['x'].shape) == (4, 3, 2)
    assert tuple(d.transform['x'].scale.shape) == (1, 3, 1)
    assert d.num_nodes == 3
    # a second call must reproduce the first (no shared-state corruption)
    sub2 = d.subgraph(torch.tensor([0, 1]))
    assert torch.equal(sub2['x'], sub['x'])


# -- numpy ------------------------------------------------------------------

def test_numpy_converts_tensors():
    d = Data(input={'x': _grid(2, 3)}, pattern={'x': 't n'})
    d.numpy()
    assert isinstance(d['x'], np.ndarray)
