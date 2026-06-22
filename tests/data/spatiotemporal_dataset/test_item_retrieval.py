"""Item retrieval (x/y exact values & shapes vs reference)."""
import numpy as np
import pytest

from tsl.data import StaticBatch

from .helpers import _make_grid_dataset, ref_horizon_steps, ref_window_steps


ITEM_CONFIGS = [
    (4, 2, 1, 0, 1, 1),
    (12, 1, 1, 0, 1, 1),
    (6, 3, 2, 1, 1, 1),   # stride + delay
    (8, 4, 1, 0, 2, 2),   # lags subsample both ends
    (4, 2, 1, -2, 1, 1),  # negative delay (window/horizon overlap)
    (0, 3, 1, 0, 1, 1),   # horizon-only (no window)
]


@pytest.mark.parametrize('window,horizon,stride,delay,window_lag,horizon_lag',
                         ITEM_CONFIGS)
def test_item_x_y_exact_values(window, horizon, stride, delay, window_lag,
                               horizon_lag):
    n_steps, n_nodes, n_channels = 60, 3, 2
    ds, target = _make_grid_dataset(window, horizon, stride, delay, window_lag,
                                    horizon_lag, n_steps, n_nodes, n_channels)
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        idx = ds.indices[p].item()
        item = ds[p]
        # y is the target at the horizon steps, exactly
        h_steps = ref_horizon_steps(idx, window, horizon, delay, horizon_lag)
        np.testing.assert_array_equal(item.y.numpy(), target[h_steps])
        assert tuple(item.y.shape) == (len(h_steps), n_nodes, n_channels)
        # x is the target at the window steps (when there is a window)
        if window > 0:
            w_steps = ref_window_steps(idx, window, window_lag)
            np.testing.assert_array_equal(item.x.numpy(), target[w_steps])
            assert tuple(item.x.shape) == (len(w_steps), n_nodes, n_channels)
        else:
            assert 'x' not in item.input


def test_negative_index_matches_positive():
    ds, target = _make_grid_dataset(window=4, horizon=2)
    last = ds[ds.n_samples - 1]
    neg = ds[-1]
    np.testing.assert_array_equal(neg.x.numpy(), last.x.numpy())
    np.testing.assert_array_equal(neg.y.numpy(), last.y.numpy())
    # -1 must address the last sample, whose window ends at the very last step
    idx = ds.indices[-1].item()
    np.testing.assert_array_equal(
        neg.x.numpy(), target[ref_window_steps(idx, 4, 1)])


def test_slice_returns_static_batch_with_exact_values():
    ds, target = _make_grid_dataset(window=4, horizon=2, n_nodes=3,
                                    n_channels=2)
    batch = ds[0:5]
    assert isinstance(batch, StaticBatch)
    # batch adds a leading 'b' dim to time-varying tensors
    assert tuple(batch.x.shape) == (5, 4, 3, 2)
    assert tuple(batch.y.shape) == (5, 2, 3, 2)
    # each row matches the corresponding single-item retrieval
    for p in range(5):
        np.testing.assert_array_equal(batch.x[p].numpy(), ds[p].x.numpy())
        np.testing.assert_array_equal(batch.y[p].numpy(), ds[p].y.numpy())


def test_delay_shifts_horizon_only():
    # increasing delay shifts the horizon forward but leaves the window fixed
    ds0, target = _make_grid_dataset(window=4, horizon=2, delay=0)
    ds3, _ = _make_grid_dataset(window=4, horizon=2, delay=3)
    np.testing.assert_array_equal(ds0[0].x.numpy(), ds3[0].x.numpy())
    np.testing.assert_array_equal(ds0[0].y.numpy(), target[4:6])
    np.testing.assert_array_equal(ds3[0].y.numpy(), target[7:9])
