""" Test covariates (synch-mode routing, prefixes, partitioning, removal).
"""
import numpy as np
import pytest
import torch

from tsl.data.batch_map import BatchMap, BatchMapItem
from tsl.data.preprocessing.scalers import StandardScaler
from tsl.data.synch_mode import HORIZON, WINDOW

from .helpers import _make_grid_dataset, ref_horizon_steps, ref_window_steps


def _node_cov(n_steps, n_nodes, n_channels=1, offset=1000):
    """Node-level covariate with a unique value per (t, n, f) cell, in a range
    disjoint from the target grid so a misrouted slice is detectable."""
    size = n_steps * n_nodes * n_channels
    return (offset + np.arange(size)).reshape(
        n_steps, n_nodes, n_channels).astype('float32')


def _graph_cov(n_steps, n_channels=2, offset=5000):
    """Graph-level (no node dim) covariate, pattern ``t f``."""
    return (offset + np.arange(n_steps * n_channels)).reshape(
        n_steps, n_channels).astype('float32')


# -- windowing ---------------------------------------------

INPUT_CONFIGS = [
    (4, 2, 1, 0),
    (6, 3, 2, 1),    # stride + delay
    (12, 12, 1, 0),  # horizon == window
    (4, 3, 1, -2),   # delay < 0: window/horizon overlap
]


@pytest.mark.parametrize('window,horizon,stride,delay', INPUT_CONFIGS)
def test_covariate_synch_mode_routing(window, horizon, stride, delay):
    n_steps, n_nodes = 60, 3
    ds, _ = _make_grid_dataset(window, horizon, stride, delay, n_steps=n_steps,
                               n_nodes=n_nodes, n_channels=2)
    cov = _node_cov(n_steps, n_nodes)
    ds.add_covariate('u_win', cov, 't n f', synch_mode=WINDOW)
    ds.add_covariate('u_hor', cov, 't n f', synch_mode=HORIZON)
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        idx = ds.indices[p].item()
        item = ds[p]
        w_steps = ref_window_steps(idx, window, 1)
        h_steps = ref_horizon_steps(idx, window, horizon, delay, 1)
        # WINDOW covariate is sliced at the window steps...
        np.testing.assert_array_equal(item.input['u_win'].numpy(), cov[w_steps])
        # ...and HORIZON covariate strictly at the horizon steps.
        np.testing.assert_array_equal(item.input['u_hor'].numpy(), cov[h_steps])


# -- add covariate (node / graph level) -------------------------------------

def test_add_covariate_node_and_graph_level():
    ds, _ = _make_grid_dataset(window=4, horizon=2, n_nodes=3, n_channels=2)
    n_steps = ds.n_steps
    node = _node_cov(n_steps, 3)
    graph = _graph_cov(n_steps)
    ds.add_covariate('un', node, 't n f')
    ds.add_covariate('ug', graph, 't f')
    # accessible as an attribute of the dataset
    np.testing.assert_array_equal(ds.un.numpy(), node)
    np.testing.assert_array_equal(ds.ug.numpy(), graph)
    # present in the item's input, with the declared patterns
    item = ds[0]
    assert 'un' in item.input and 'ug' in item.input
    assert ds.patterns['un'] == 't n f'
    assert ds.patterns['ug'] == 't f'


def test_add_exogenous_global_prefix_forces_graph_level():
    ds, _ = _make_grid_dataset(window=4, horizon=2, n_nodes=3)
    g = _graph_cov(ds.n_steps)
    ds.add_exogenous('global_g', g)  # 'global_' is stripped, forces graph-level
    assert 'g' in ds.exogenous
    assert 'global_g' not in ds.covariates
    assert ds.patterns['g'] == 't f'  # no node dimension


def test_add_exogenous_node_level_default():
    ds, _ = _make_grid_dataset(window=4, horizon=2, n_nodes=3)
    node = _node_cov(ds.n_steps, 3)
    ds.add_exogenous('u', node)  # node_level=True by default
    assert ds.patterns['u'] == 't n f'
    assert 'u' in ds.exogenous


# -- static attributes ------------------------------------------------------

def test_static_attribute_not_time_sliced():
    ds, _ = _make_grid_dataset(window=4, horizon=3, delay=1, n_nodes=3)
    attr = np.arange(3 * 5).reshape(3, 5).astype('float32')  # 'n f', no time
    ds.add_covariate('meta', attr, 'n f')
    assert 'meta' in ds.attributes and 'meta' not in ds.exogenous
    # a static attribute is delivered whole to every sample, never time-sliced
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        np.testing.assert_array_equal(ds[p].input['meta'].numpy(), attr)


# -- promoting a covariate to the target ------------------------------------

def test_covariate_promoted_to_target_via_set_target_map():
    # add a covariate, then make it the prediction target by swapping the
    # target map for one that points at the covariate's horizon slice.
    window, horizon, delay = 4, 2, 1
    n_steps, n_nodes = 60, 3
    ds, target = _make_grid_dataset(window, horizon, delay=delay,
                                    n_steps=n_steps, n_nodes=n_nodes,
                                    n_channels=2)
    cov = _node_cov(n_steps, n_nodes)
    ds.add_covariate('u', cov, 't n f')

    ds.set_target_map(y=BatchMapItem('u', HORIZON))

    # the target map now references the covariate, not the original target
    assert set(ds.target_map) == {'y'}
    assert ds.target_map['y'].keys == ['u']
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        idx = ds.indices[p].item()
        item = ds[p]
        h_steps = ref_horizon_steps(idx, window, horizon, delay, 1)
        # y is now the covariate sliced at the horizon steps, not the grid target
        np.testing.assert_array_equal(item.y.numpy(), cov[h_steps])
        assert not np.array_equal(item.y.numpy(), target[h_steps])


def test_covariate_promoted_to_target_only_keeps_x_as_input():
    # a covariate added input-side and then promoted to be the sole target:
    # the window still feeds the original target as 'x', the horizon yields the
    # covariate as 'y'.
    window, horizon = 4, 2
    n_steps, n_nodes = 60, 3
    ds, target = _make_grid_dataset(window, horizon, n_steps=n_steps,
                                    n_nodes=n_nodes, n_channels=2)
    cov = _node_cov(n_steps, n_nodes)
    # keep the covariate out of the input map: it is purely the new target
    ds.add_covariate('u', cov, 't n f', add_to_input_map=False)
    assert 'u' not in ds.input_map

    ds.set_target_map(y=('u', HORIZON))  # tuple form, cast to a BatchMapItem

    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        idx = ds.indices[p].item()
        item = ds[p]
        w_steps = ref_window_steps(idx, window, 1)
        h_steps = ref_horizon_steps(idx, window, horizon, 0, 1)
        np.testing.assert_array_equal(item.x.numpy(), target[w_steps])
        np.testing.assert_array_equal(item.y.numpy(), cov[h_steps])
        assert 'u' not in item.input


# -- promoting a covariate to the input -------------------------------------

def test_covariate_promoted_to_only_input_via_set_input_map():
    # add a covariate, then make it the sole model input by swapping the input
    # map for a freshly built one that points at the covariate's window slice.
    window, horizon = 4, 2
    n_steps, n_nodes = 60, 3
    ds, target = _make_grid_dataset(window, horizon, n_steps=n_steps,
                                    n_nodes=n_nodes, n_channels=2)
    cov = _node_cov(n_steps, n_nodes)
    ds.add_covariate('u', cov, 't n f')

    new_x = BatchMap()
    new_x['x'] = BatchMapItem('u', WINDOW)
    ds.set_input_map(new_x)

    # the input map now references only the covariate, not the original target
    assert set(ds.input_map) == {'x'}
    assert ds.input_map['x'].keys == ['u']
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        idx = ds.indices[p].item()
        item = ds[p]
        w_steps = ref_window_steps(idx, window, 1)
        h_steps = ref_horizon_steps(idx, window, horizon, 0, 1)
        # x is now the covariate sliced at the window steps, not the grid target
        np.testing.assert_array_equal(item.x.numpy(), cov[w_steps])
        assert not np.array_equal(item.x.numpy(), target[w_steps])
        assert 'u' not in item.input  # no longer mapped under its own key
        # the original target still drives y at the horizon
        np.testing.assert_array_equal(item.y.numpy(), target[h_steps])


# -- partitioning / flags ---------------------------------------------------

def test_covariate_partitioning_and_flags():
    ds, _ = _make_grid_dataset(window=4, horizon=2, n_nodes=3)
    assert not ds.has_covariates and ds.n_covariates == 0
    ds.add_covariate('u', _node_cov(ds.n_steps, 3), 't n f')        # exogenous
    ds.add_covariate('meta', np.arange(3 * 4).reshape(3, 4).astype('float32'),
                     'n f')                                          # attribute
    assert ds.has_covariates and ds.n_covariates == 2
    # 't' in pattern -> exogenous; otherwise -> attribute
    assert set(ds.exogenous) == {'u'}
    assert set(ds.attributes) == {'meta'}
    assert set(ds.covariates) == {'u', 'meta'}


# -- removal --------------------------

def test_remove_covariate_clears_everywhere():
    ds, _ = _make_grid_dataset(window=4, horizon=3, n_nodes=3)
    u = _node_cov(ds.n_steps, 3)
    ds.add_covariate('u', u, 't n f')
    sc = StandardScaler(axis=0)
    sc.fit(torch.as_tensor(u))
    ds.add_scaler('u', sc)
    assert 'u' in ds.covariates and 'u' in ds.input_map and 'u' in ds.scalers

    ds.remove_covariate('u')

    # gone from the covariate store, the input map, and the scalers
    assert 'u' not in ds.covariates
    assert 'u' not in ds.input_map
    assert 'u' not in ds.scalers
    assert ds.n_covariates == 0
    # and the dataset is still usable
    item = ds[0]
    assert 'u' not in item.input
    assert 'x' in item.input


def test_remove_covariate_drops_composite_items():
    ds, _ = _make_grid_dataset(window=4, horizon=2, n_nodes=3)
    ds.add_covariate('u', _node_cov(ds.n_steps, 3), 't n f')
    # composite item that references the covariate under a different map key
    ds.input_map['xu'] = (['target', 'u'], 'window', True, -1)
    assert 'xu' in ds.input_map

    ds.remove_covariate('u')

    # the composite item can no longer be collated without 'u', so it is dropped
    assert 'xu' not in ds.input_map
    assert 'xu' not in ds[0].input
    assert 'u' not in ds.input_map
    assert 'u' not in ds[0].input
    # the remaining default item still collates
    assert 'x' in ds.input_map
    assert 'x' in ds[0].input
