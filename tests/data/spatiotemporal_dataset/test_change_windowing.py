"""Test change_windowing context manager.
"""
import numpy as np
import pytest

from .helpers import _make_grid_dataset, ref_horizon_steps, ref_window_steps


def test_change_windowing_active_inside_and_restored_after():
    ds, target = _make_grid_dataset(window=4, horizon=2, delay=0, stride=1,
                                    n_steps=60, n_nodes=2, n_channels=1)
    # snapshot the full windowing state before entering the context
    before = dict(window=ds.window, horizon=ds.horizon, delay=ds.delay,
                  stride=ds.stride, window_lag=ds.window_lag,
                  horizon_lag=ds.horizon_lag, n_samples=ds.n_samples)
    before_indices = ds.indices.clone()

    with ds.change_windowing(window=10, horizon=5, delay=1) as d:
        # the override is active: arithmetic and retrieved tensors follow it
        assert d.window == 10 and d.horizon == 5 and d.delay == 1
        idx = d.indices[0].item()
        item = d[0]
        np.testing.assert_array_equal(
            item.x.numpy(), target[ref_window_steps(idx, 10, 1)])
        np.testing.assert_array_equal(
            item.y.numpy(), target[ref_horizon_steps(idx, 10, 5, 1, 1)])

    # on exit every windowing attribute and the cached indices are restored
    after = dict(window=ds.window, horizon=ds.horizon, delay=ds.delay,
                 stride=ds.stride, window_lag=ds.window_lag,
                 horizon_lag=ds.horizon_lag, n_samples=ds.n_samples)
    assert after == before
    np.testing.assert_array_equal(ds.indices.numpy(), before_indices.numpy())
    # and retrieval is back to the original windowing
    idx0 = ds.indices[0].item()
    np.testing.assert_array_equal(
        ds[0].x.numpy(), target[ref_window_steps(idx0, 4, 1)])


def test_change_windowing_restores_after_exception():
    # even if the body raises, the finally-clause must roll back the windowing
    ds, _ = _make_grid_dataset(window=4, horizon=2, n_steps=60, n_nodes=2)
    before = (ds.window, ds.horizon, ds.n_samples)
    with pytest.raises(RuntimeError):
        with ds.change_windowing(window=8):
            assert ds.window == 8  # active inside
            raise RuntimeError('boom')
    assert (ds.window, ds.horizon, ds.n_samples) == before


def test_change_windowing_rejects_invalid_key():
    ds, _ = _make_grid_dataset(window=4, horizon=2, n_steps=60, n_nodes=2)
    before = (ds.window, ds.horizon, ds.n_samples)
    with pytest.raises(AssertionError):
        with ds.change_windowing(not_a_windowing_key=3):
            pass
    # the rejected override must not have touched the dataset
    assert (ds.window, ds.horizon, ds.n_samples) == before
