"""Scalers."""
import numpy as np
import pytest
import torch

from tsl.data import SpatioTemporalDataset

from .helpers import (_grid_target, fitted_standard_scaler, ref_horizon_steps,
                      ref_window_steps)


def test_scaler_transforms_x_target_y_unscaled():
    n_steps, n_nodes, n_channels = 60, 2, 1
    target = _grid_target(n_steps, n_nodes, n_channels)
    sc = fitted_standard_scaler(target)
    ds = SpatioTemporalDataset(target=target, window=4, horizon=2, delay=1,
                               scalers={'target': sc})
    item = ds[0]
    idx = ds.indices[0].item()
    w_steps = ref_window_steps(idx, 4, 1)
    h_steps = ref_horizon_steps(idx, 4, 2, 1, 1)
    # x (input window) is scaled with the fitted params...
    expected_x = sc.transform(torch.as_tensor(target[w_steps]))
    np.testing.assert_allclose(item.x.numpy(), expected_x.numpy(), rtol=1e-6)
    assert not np.allclose(item.x.numpy(), target[w_steps])
    # ...while y (target) is left in raw units by default
    np.testing.assert_array_equal(item.y.numpy(), target[h_steps])


def test_scaler_attached_to_item_transform():
    target = _grid_target(60, 2, 1)
    sc = fitted_standard_scaler(target)
    ds = SpatioTemporalDataset(target=target, window=4, horizon=2,
                               scalers={'target': sc})
    item = ds[0]
    assert 'x' in item.transform
    # the attached scaler round-trips the (scaled) x back to raw units
    raw = item.transform['x'].inverse_transform(item.x)
    idx = ds.indices[0].item()
    np.testing.assert_allclose(raw.numpy(),
                               target[ref_window_steps(idx, 4, 1)], rtol=1e-6)


def test_scaler_params_constant_across_samples():
    # Scaling must use the stored fit params identically for every sample; a
    # per-batch refit would leak each batch's statistics into its inputs.
    target = _grid_target(80, 2, 1)
    sc = fitted_standard_scaler(target)
    ds = SpatioTemporalDataset(target=target, window=4, horizon=2,
                               scalers={'target': sc})
    first = ds[0].transform['x']
    for p in (1, 10, ds.n_samples - 1):
        other = ds[p].transform['x']
        assert torch.allclose(first.bias, other.bias)
        assert torch.allclose(first.scale, other.scale)
    # and they equal the originally fitted params
    assert torch.allclose(first.bias.squeeze(), sc.bias.squeeze())
    assert torch.allclose(first.scale.squeeze(), sc.scale.squeeze())


def test_add_scaler_invalid_key_raises():
    target = _grid_target(40, 2, 1)
    ds = SpatioTemporalDataset(target=target, window=4, horizon=2)
    with pytest.raises(KeyError):
        ds.add_scaler('does_not_exist', fitted_standard_scaler(target))


def test_scaler_on_covariate_is_applied():
    # a scaler attached to a covariate (not just the target) scales that
    # covariate in the item and rides along in item.transform
    target = _grid_target(60, 3, 2)
    u = _grid_target(60, 3, 1)
    ds = SpatioTemporalDataset(target=target, window=4, horizon=2)
    ds.add_covariate('u', u, 't n f', preprocess=True)
    ds.add_scaler('u', fitted_standard_scaler(u))
    item = ds[0]
    idx = ds.indices[0].item()
    w_steps = ref_window_steps(idx, 4, 1)
    assert 'u' in item.transform
    expected = item.transform['u'].transform(torch.as_tensor(u[w_steps]))
    np.testing.assert_allclose(item.input['u'].numpy(), expected.numpy(),
                               rtol=1e-6)
    assert not np.allclose(item.input['u'].numpy(), u[w_steps])
