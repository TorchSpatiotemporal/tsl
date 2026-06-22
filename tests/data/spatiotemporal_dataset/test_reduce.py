""" reduce / reduce_ (copy vs inplace, time/node, edge pruning)."""
import numpy as np
import pandas as pd
import torch

from tsl.data import SpatioTemporalDataset

from .helpers import _grid_target


def _dataset():
    target = _grid_target(40, 4, 1)
    index = pd.date_range('2020-01-01', periods=40, freq='h')
    mask = (np.arange(40 * 4).reshape(40, 4, 1) % 2).astype(bool)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_weight = torch.tensor([10., 20., 30., 40.])
    ds = SpatioTemporalDataset(target=target, index=index, mask=mask,
                               connectivity=(edge_index, edge_weight),
                               window=4, horizon=2)
    ds.add_covariate('u', _grid_target(40, 4, 1), 't n f')
    ds.set_trend(np.ones((40, 4, 1), dtype='float32'))
    return ds


def test_reduce_is_a_copy_reduce_inplace_mutates():
    ds = _dataset()
    reduced = ds.reduce(node_index=np.array([0, 1]))
    # the original is untouched...
    assert ds.n_nodes == 4
    # ...and the returned copy is reduced
    assert reduced.n_nodes == 2

    ds2 = _dataset()
    ret = ds2.reduce_(node_index=np.array([0, 1]))
    assert ret is ds2  # reduce_ mutates and returns self
    assert ds2.n_nodes == 2


def test_reduce_time_and_node_aligns_all_tensors():
    ds = _dataset()
    red = ds.reduce(time_index=np.arange(0, 20), node_index=np.array([0, 1, 2]))
    # every per-(t, n) tensor is reduced consistently
    assert red.shape == (20, 3, 1)
    assert tuple(red.mask.shape) == (20, 3, 1)
    assert tuple(red.trend.shape) == (20, 3, 1)
    assert tuple(red.u.shape) == (20, 3, 1)
    assert len(red.index) == 20
    assert red.n_steps == 20 and red.n_nodes == 3


def test_reduce_node_only_keeps_full_time_axis():
    # node-only reduction must leave the temporal index untouched (regression
    # guard: it used to crash dereferencing a None time_index)
    ds = _dataset()
    red = ds.reduce(node_index=np.array([0, 1]))
    assert red.shape == (40, 2, 1)
    assert len(red.index) == 40


def test_reduce_prunes_edges_and_weights():
    ds = _dataset()
    red = ds.reduce(node_index=np.array([0, 1, 2]))  # drop node 3
    # edges touching node 3 (3->0) are removed; their weights go with them
    assert red.n_edges == 2
    np.testing.assert_array_equal(red.edge_weight.numpy(), [10., 20.])
