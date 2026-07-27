"""Unit tests for :mod:`tsl.data.batch`.

Covers :class:`StaticBatch`, :class:`DisjointBatch` and the scaler/graph collate
helpers, including full round-trips through ``ScalerModule`` transforms.
"""
import numpy as np
import pytest
import torch

from tsl.data import Data, DisjointBatch, StaticBatch
from tsl.data.batch import (collate_scaler_params, get_static_scaler,
                            separate_scaler_params, static_scaler_collate)
from tsl.data.preprocessing.scalers import ScalerModule


# -- builders ---------------------------------------------------------------

def _grid(*shape):
    return torch.arange(int(np.prod(shape))).float().reshape(*shape)


def _data(b=0, n_nodes=3, t=4, c=2, scaler=None, with_u=False):
    """A static graph sample whose tensors are offset by ``b`` so that distinct
    samples never collide element-wise."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.tensor([1., 2., 3.])
    x = _grid(t, n_nodes, c) + b * 1000
    inp = {'x': x}
    pattern = {'x': 't n c', 'y': 't n c', 'edge_index': '2 e',
               'edge_weight': 'e'}
    if with_u:  # node-invariant, time-varying covariate
        inp['u'] = _grid(t, c) + b * 1000
        pattern['u'] = 't c'
    transform = {'x': scaler} if scaler is not None else None
    return Data(input=inp, target={'y': _grid(2, n_nodes, c) + b * 1000},
                edge_index=edge_index, edge_weight=edge_weight,
                transform=transform, pattern=pattern)


def _example_ids(data_list):
    """Recover the originating sample index of each :class:`Data` from the
    ``b * 1000`` offset baked into its tensors (the first 'x' value is
    ``0 + b * 1000``)."""
    return [round(d.x.flatten()[0].item() / 1000) for d in data_list]


def _tv_scaler(t=4, n_nodes=3, c=2, offset=0.):
    """A time-varying scaler (its params carry a leading time dimension).

    ``offset`` shifts both params so that distinct samples get distinct,
    non-trivial scalers (used to prove per-graph separation on reconstruction).
    """
    return ScalerModule(bias=torch.zeros(t, n_nodes, c) + offset,
                        scale=_grid(t, n_nodes, c) + 1 + offset, pattern='t n c')


# ===========================================================================
# StaticBatch
# ===========================================================================

def test_static_collate_shapes_and_patterns():
    batch = StaticBatch.from_data_list([_data(i) for i in range(3)])
    # time-varying tensors gain a leading batch dim and a 'b ' pattern prefix
    assert tuple(batch.x.shape) == (3, 4, 3, 2)
    assert batch.pattern['x'] == 'b t n c'
    assert tuple(batch.y.shape) == (3, 2, 3, 2)
    # the shared static graph is kept from the first element, unchanged
    assert tuple(batch.edge_index.shape) == (2, 3)
    assert batch.pattern['edge_index'] == '2 e'
    assert batch.batch_size == 3 and batch.num_graphs == 3


def test_static_get_example_round_trip():
    data_list = [_data(i) for i in range(3)]
    batch = StaticBatch.from_data_list(data_list)
    for i, original in enumerate(data_list):
        ex = batch.get_example(i)
        assert ex.pattern['x'] == 't n c'  # 'b ' stripped back off
        assert torch.equal(ex.x, original.x)
        assert torch.equal(ex.y, original.y)
        assert torch.equal(ex.edge_index, original.edge_index)


def test_static_to_data_list():
    data_list = [_data(i) for i in range(3)]
    batch = StaticBatch.from_data_list(data_list)
    out = batch.to_data_list()
    assert len(out) == 3
    for original, restored in zip(data_list, out):
        assert torch.equal(restored.x, original.x)


def test_static_batch_size_inferred_from_pattern():
    # no explicit size: inferred from the first 'b'-prefixed tensor
    batch = StaticBatch(input={'x': torch.zeros(5, 4, 3)},
                        pattern={'x': 'b t n'})
    assert batch.batch_size == 5


def test_static_getitem_dispatch():
    data_list = [_data(i) for i in range(4)]
    batch = StaticBatch.from_data_list(data_list)
    # int -> single example
    assert torch.equal(batch[1].x, data_list[1].x)
    # 0-dim tensor -> single example
    assert torch.equal(batch[torch.tensor(2)].x, data_list[2].x)
    # str -> attribute access (the raw batched tensor)
    assert torch.equal(batch['x'], batch.x)
    # slice -> list of the *right* examples (1 and 2), not just the right count
    assert _example_ids(batch[1:3]) == [1, 2]


@pytest.mark.parametrize('index,expected', [
    (slice(1, 3), [1, 2]),
    ([0, 2], [0, 2]),
    ((0, 3), [0, 3]),
])
def test_static_index_select_sequences(index, expected):
    data_list = [_data(i) for i in range(4)]
    batch = StaticBatch.from_data_list(data_list)
    out = batch.index_select(index)
    assert _example_ids(out) == expected


def test_static_index_select_tensor_and_ndarray():
    batch = StaticBatch.from_data_list([_data(i) for i in range(4)])
    # every index form must select exactly examples 0 and 2 (by value, not count)
    assert _example_ids(batch.index_select(torch.tensor([0, 2]))) == [0, 2]
    assert _example_ids(
        batch.index_select(np.array([0, 2], dtype=np.int64))) == [0, 2]
    # bool tensor / bool ndarray pick the True positions (0 and 2)
    mask = [True, False, True, False]
    assert _example_ids(batch.index_select(torch.tensor(mask))) == [0, 2]
    assert _example_ids(batch.index_select(np.array(mask))) == [0, 2]


def test_static_index_select_invalid_type():
    batch = StaticBatch.from_data_list([_data(i) for i in range(2)])
    with pytest.raises(IndexError):
        batch.index_select(1.5)


# -- static scaler round-trip ----------------------------------------------

def test_static_scaler_collate_time_varying():
    # distinct scaler per element so recovery is checked by value, not just shape
    scalers = [{'x': _tv_scaler(offset=i * 1000.)} for i in range(3)]
    out = static_scaler_collate(scalers)
    # time-varying params are stacked on a new batch axis and tagged 'b '
    assert out['x'].pattern == 'b t n c'
    assert tuple(out['x'].scale.shape) == (3, 4, 3, 2)
    # round-trip recovery pulls back exactly element 1's params (not 0's or 2's)
    recovered = get_static_scaler(out, 1)
    assert recovered['x'].pattern == 't n c'
    assert tuple(recovered['x'].scale.shape) == (4, 3, 2)
    assert torch.equal(recovered['x'].scale, scalers[1]['x'].scale)
    assert torch.equal(recovered['x'].bias, scalers[1]['x'].bias)


def test_static_scaler_collate_time_invariant_keeps_first():
    # distinct statics so we can verify the *first* is the one kept
    statics = [ScalerModule(bias=torch.zeros(3, 2) + i * 1000.,
                            scale=_grid(3, 2) + 1 + i * 1000., pattern='n c')
               for i in range(3)]
    out = static_scaler_collate([{'x': s} for s in statics])
    # no batch dimension is added for a time-invariant scaler ...
    assert out['x'].pattern == 'n c'
    assert tuple(out['x'].scale.shape) == (3, 2)
    # ... and the first element's params are kept verbatim (not collated)
    assert torch.equal(out['x'].scale, statics[0].scale)
    assert torch.equal(out['x'].bias, statics[0].bias)


def test_static_batch_carries_transform_through_round_trip():
    # distinct scaler per sample so the round-trip is checked by value
    data_list = [_data(i, scaler=_tv_scaler(offset=i * 1000.)) for i in range(3)]
    batch = StaticBatch.from_data_list(data_list)
    assert batch.transform['x'].pattern == 'b t n c'
    ex = batch.get_example(2)
    assert ex.transform['x'].pattern == 't n c'
    assert torch.equal(ex.transform['x'].scale, data_list[2].transform['x'].scale)
    assert torch.equal(ex.transform['x'].bias, data_list[2].transform['x'].bias)


# ===========================================================================
# scaler-param collate / separate helpers
# ===========================================================================

def test_collate_scaler_params_stack():
    value, is_repeated = collate_scaler_params(
        [torch.zeros(2, 2), torch.ones(2, 2)], cat_dim=None)
    assert tuple(value.shape) == (2, 2, 2)
    assert is_repeated is False


def test_collate_scaler_params_concat():
    value, is_repeated = collate_scaler_params(
        [torch.tensor([1., 2.]), torch.tensor([3., 4., 5.])], cat_dim=0)
    assert value.tolist() == [1., 2., 3., 4., 5.]
    assert is_repeated is False


def test_collate_scaler_params_repeated_with_batch_index():
    batch_index = torch.tensor([0, 0, 1])
    value, is_repeated = collate_scaler_params(
        [torch.tensor([[5.]]), torch.tensor([[6.]])],
        cat_dim=0, batch_index=batch_index)
    assert value.flatten().tolist() == [5., 5., 6.]
    assert is_repeated is True


def test_separate_scaler_params_inverts_both_branches():
    # non-repeated: narrow according to slices
    value = torch.tensor([1., 2., 3., 4., 5.])
    slices = [0, 2, 5]
    out = separate_scaler_params(value, slices, idx=1, is_repeated=False,
                                 cat_dim=0)
    assert out.tolist() == [3., 4., 5.]
    # repeated: index_select using the stored index tensor
    slices_rep = [torch.tensor([0, 1]), torch.tensor([2])]
    out_rep = separate_scaler_params(value, slices_rep, idx=0,
                                     is_repeated=True, cat_dim=0)
    assert out_rep.tolist() == [1., 2.]


# ===========================================================================
# DisjointBatch
# ===========================================================================

def test_disjoint_assignment_vector_and_sizes():
    batch = DisjointBatch.from_data_list([_data(i) for i in range(3)])
    assert batch.batch.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert batch.ptr.tolist() == [0, 3, 6, 9]
    assert batch.num_graphs == 3 and batch.batch_size == 3
    # nodes are concatenated along the node dimension (one big graph)
    assert tuple(batch.x.shape) == (4, 9, 2)
    assert tuple(batch.edge_index.shape) == (2, 9)


def test_disjoint_get_example_round_trip():
    data_list = [_data(i) for i in range(3)]
    batch = DisjointBatch.from_data_list(data_list)
    for i, original in enumerate(data_list):
        ex = batch.get_example(i)
        assert torch.equal(ex.x, original.x)
        assert torch.equal(ex.edge_index, original.edge_index)
        assert ex.pattern['x'] == 't n c'


def test_disjoint_node_invariant_covariate_repeated_along_nodes():
    data_list = [_data(i, with_u=True) for i in range(2)]
    batch = DisjointBatch.from_data_list(data_list)
    # 'u' (pattern 't c') is repeated along a new node dimension and the
    # time/node axes are swapped to follow the time-then-node convention
    assert batch.pattern['u'] == 't n c'
    assert tuple(batch['u'].shape) == (4, 6, 2)
    # and it is correctly restored on reconstruction
    ex = batch.get_example(1)
    assert ex.pattern['u'] == 't c'
    assert torch.equal(ex['u'], data_list[1]['u'])


def test_disjoint_graph_attributes_stacked_on_batch_dim():
    data_list = [_data(i, with_u=True) for i in range(2)]
    batch = DisjointBatch.from_data_list(data_list, graph_attributes=['u'])
    # as a graph attribute, 'u' is stacked on a new leading (batch) dim instead
    # of being repeated along nodes
    assert tuple(batch['u'].shape) == (2, 4, 2)
    assert batch.pattern['u'].startswith('b')
    # each stacked row corresponds, in order, to the sample it came from
    for i, original in enumerate(data_list):
        assert torch.equal(batch['u'][i], original['u'])


def test_disjoint_force_batch_adds_dummy_dimension():
    batch = DisjointBatch.from_data_list([_data(i) for i in range(2)],
                                         force_batch=True)
    assert tuple(batch.x.shape) == (1, 4, 6, 2)
    assert batch.pattern['x'] == 'b t n c'


def test_disjoint_exclude_keys():
    batch = DisjointBatch.from_data_list([_data(i, with_u=True)
                                          for i in range(2)],
                                         exclude_keys=['u'])
    assert 'u' not in batch._store


def test_disjoint_transform_round_trip():
    # a *distinct* scaler per sample, so that recovering values equal to
    # 'data_list[i]' genuinely proves each example is rebuilt from its own node
    # chunk -- with identical scalers a wrong-graph pick would pass unnoticed
    data_list = [_data(i, scaler=_tv_scaler(offset=i * 1000.)) for i in range(3)]
    batch = DisjointBatch.from_data_list(data_list)
    # scaler params are concatenated along the node dimension (3 graphs x 3 nodes)
    assert tuple(batch.transform['x'].scale.shape) == (4, 9, 2)
    for i, original in enumerate(data_list):
        ex = batch.get_example(i)
        assert tuple(ex.transform['x'].scale.shape) == (4, 3, 2)
        assert ex.transform['x'].pattern == 't n c'
        assert torch.equal(ex.transform['x'].scale, original.transform['x'].scale)
        assert torch.equal(ex.transform['x'].bias, original.transform['x'].bias)
