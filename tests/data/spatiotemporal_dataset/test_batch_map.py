"""Test BatchMap (default maps, reset/set/update, routing, composites)."""
import numpy as np

from tsl.data import SpatioTemporalDataset
from tsl.data.synch_mode import HORIZON, WINDOW

from .helpers import _grid_target


def _dataset():
    return SpatioTemporalDataset(target=_grid_target(60, 3, 2), window=4,
                                 horizon=2)


def test_default_input_and_target_maps():
    ds = _dataset()
    x, y = ds.input_map['x'], ds.target_map['y']
    # x: the target, window-synchronized and preprocessed (scaled if a scaler)
    assert x.keys == ['target']
    assert x.synch_mode is WINDOW and x.preprocess is True
    # y: the target, horizon-synchronized and left in raw units
    assert y.keys == ['target']
    assert y.synch_mode is HORIZON and y.preprocess is False


def test_by_synch_mode_filters():
    ds = _dataset()
    ds.add_covariate('u_h', np.zeros((60, 3, 1), dtype='float32'), 't n f',
                     synch_mode=HORIZON)
    assert set(ds.input_map.by_synch_mode(WINDOW)) == {'x'}
    assert set(ds.input_map.by_synch_mode(HORIZON)) == {'u_h'}
    assert set(ds.target_map.by_synch_mode(HORIZON)) == {'y'}


def test_set_replaces_update_merges():
    ds = _dataset()
    ds.set_input_map(only=('target', 'window'))  # set -> replace
    assert list(ds.input_map) == ['only']

    ds2 = _dataset()
    ds2.update_input_map(extra=('target', 'horizon'))  # update -> merge
    assert list(ds2.input_map) == ['x', 'extra']


def test_reset_restores_defaults_with_covariates():
    ds = _dataset()
    ds.add_covariate('u', np.zeros((60, 3, 1), dtype='float32'), 't n f')
    ds.set_input_map(only=('target', 'window'))  # clobber the map
    assert list(ds.input_map) == ['only']
    ds.reset_input_map()
    # the default map is x plus every covariate (with default routing)
    assert set(ds.input_map) == {'x', 'u'}


def test_multi_key_composite_pattern_and_shape():
    ds = _dataset()
    ds.add_covariate('u', np.zeros((60, 3, 1), dtype='float32'), 't n f')
    # concatenate target (2 ch) and u (1 ch) along the channel dim
    ds.update_input_map(xu=(['target', 'u'], 'window', True, -1))
    item = ds.input_map['xu']
    assert item.keys == ['target', 'u']
    assert item.pattern == 't n f'
    assert item.shape == (60, 3, 3)  # 2 + 1 channels
    # and it collates to that channel count in a batch
    batch = ds[0:3]
    assert tuple(batch.input['xu'].shape) == (3, 4, 3, 3)
