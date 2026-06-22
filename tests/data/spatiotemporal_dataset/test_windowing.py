"""Group 2 -- Windowing arithmetic."""
import numpy as np
import pytest

from .helpers import (_make_dataset, _make_grid_dataset, ref_horizon_range,
                      ref_horizon_steps, ref_sample_span, ref_window_range,
                      windowing_cases)


@windowing_cases
def test_windowing_properties(window, horizon, stride, delay, window_lag,
                              horizon_lag):
    ds = _make_dataset(window, horizon, stride, delay, window_lag, horizon_lag)
    assert ds.horizon_offset == window + delay
    assert ds.sample_span == ref_sample_span(window, horizon, delay)
    assert ds.samples_offset == int(np.ceil(window / stride))


@windowing_cases
def test_lag_subsampling_counts(window, horizon, stride, delay, window_lag,
                                horizon_lag):
    ds = _make_dataset(window, horizon, stride, delay, window_lag, horizon_lag)
    # number of steps actually taken per role respects the lag subsampling
    assert len(ds._horizon_range) == len(
        ref_horizon_range(window, horizon, delay, horizon_lag))
    if window > 0:
        assert len(ds._window_range) == len(
            ref_window_range(window, window_lag))


@windowing_cases
def test_window_horizon_disjoint_when_delay_nonneg(window, horizon, stride,
                                                   delay, window_lag,
                                                   horizon_lag):
    # window steps and horizon steps must never share a time step when
    # delay >= 0: a shared step would feed a target into the input window.
    if window == 0 or delay < 0:
        pytest.skip('overlap only relevant for window>0, delay>=0')
    ds = _make_dataset(window, horizon, stride, delay, window_lag, horizon_lag)
    expanded = ds.expand_indices()
    win = expanded['window'].numpy()
    hrz = expanded['horizon'].numpy()
    for w_row, h_row in zip(win, hrz):
        assert set(w_row).isdisjoint(set(h_row))


@windowing_cases
def test_window_horizon_value_disjoint_when_delay_nonneg(window, horizon,
                                                         stride, delay,
                                                         window_lag,
                                                         horizon_lag):
    # same disjointness guarantee as test_window_horizon_disjoint_when_delay_
    # nonneg, but verified on the *retrieved values* rather than the indices.
    # The grid target gives every (t, n, f) cell a globally unique value, so a
    # value shared between x and y would mean a target leaked into the window.
    if window == 0 or delay < 0:
        pytest.skip('overlap only relevant for window>0, delay>=0')
    ds, _ = _make_grid_dataset(window, horizon, stride, delay, window_lag,
                               horizon_lag)
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        item = ds[p]
        x_vals = set(item.x.numpy().ravel().tolist())
        y_vals = set(item.y.numpy().ravel().tolist())
        assert x_vals.isdisjoint(y_vals)


# (window, horizon, delay). Spans both overlap regimes: horizon >= -delay,
# where the overlap is the trailing |delay| window steps, and horizon < -delay,
# where the horizon ends before the window does so the overlap is all `horizon`
# steps. Includes window == horizon cases, one of which has window == horizon ==
# -delay (horizon_offset 0): the horizon then covers the whole window.
NEGATIVE_DELAY_OVERLAP = [
    (6, 4, -1),
    (6, 4, -2),
    (6, 4, -4),   # horizon == -delay (overlap spans the whole horizon/tail)
    (6, 4, -5),   # horizon < -delay
    (8, 2, -5),   # horizon < -delay
    (3, 5, -2),
    (12, 6, -3),
    (4, 1, -3),   # horizon < -delay
    (1, 3, -1),
    (4, 4, -2),   # window == horizon
    (5, 5, -1),   # window == horizon
    (4, 4, -4),   # window == horizon == -delay -> horizon covers whole window
]


@pytest.mark.parametrize('window,horizon,delay', NEGATIVE_DELAY_OVERLAP)
def test_window_horizon_overlap_allowed_when_delay_negative(window, horizon,
                                                            delay):
    # delay < 0 (with unit lags) makes the horizon start inside the window;
    # the overlap is intentional and must be present, not silently dropped.
    ds = _make_dataset(window, horizon, delay=delay)
    expanded = ds.expand_indices()
    overlap_seen = False
    for w_row, h_row in zip(expanded['window'].numpy(),
                            expanded['horizon'].numpy()):
        if not set(w_row).isdisjoint(set(h_row)):
            overlap_seen = True
            break
    assert overlap_seen


@pytest.mark.parametrize('window,horizon,delay', NEGATIVE_DELAY_OVERLAP)
def test_window_horizon_overlap_values_are_expected_when_delay_negative(
        window, horizon, delay):
    # with delay < 0 the horizon starts |delay| steps before the window ends,
    # so x and y share their boundary steps. On a grid target (globally unique
    # cells) the values common to x and y must be exactly the target at the
    # overlapping time steps. The overlap is window steps [window+delay, window)
    # clipped to the horizon's reach (window+delay+horizon), so its size is
    # min(-delay, horizon).
    ds, target = _make_grid_dataset(window, horizon, delay=delay)
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        idx = ds.indices[p].item()
        item = ds[p]
        # expected overlap.
        start = window + delay
        end = min(window, window + delay + horizon)
        expected_steps = idx + np.arange(start, end)
        expected = set(target[expected_steps].ravel().tolist())
        # the empirical overlap, taken from the actually retrieved tensors
        x_vals = set(item.x.numpy().ravel().tolist())
        y_vals = set(item.y.numpy().ravel().tolist())
        # exact equality: the shared values are the expected ones and no others,
        # so a spurious extra overlap would fail here rather than be absorbed.
        assert (x_vals & y_vals) == expected


@pytest.mark.parametrize('window,delay', [(0, -1), (1, -2), (4, -5),
                                          (12, -13)])
def test_negative_horizon_offset_rejected(window, delay):
    # horizon_offset = window + delay < 0 would place the horizon before the
    # sample's first step (target step < 0): a nonsensical config that the
    # dataset must refuse rather than silently read wrapped/out-of-bounds data.
    with pytest.raises(ValueError):
        _make_dataset(window=window, horizon=2, delay=delay)


def test_zero_horizon_offset_allowed():
    # horizon_offset == 0 (delay == -window) is the supported boundary: the
    # horizon starts exactly at the sample's first step.
    ds = _make_dataset(window=4, horizon=2, delay=-4)
    assert ds.horizon_offset == 0
    np.testing.assert_array_equal(
        ds.expand_indices(np.array([0]))['horizon'].numpy().ravel(),
        ref_horizon_steps(0, 4, 2, -4, 1))
