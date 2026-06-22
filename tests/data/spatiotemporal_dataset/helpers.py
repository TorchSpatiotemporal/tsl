import numpy as np
import pytest
import torch

from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing.scalers import StandardScaler


# -- independent reference --------------------------------------------------

def ref_window_range(window, window_lag):
    return np.arange(0, window, window_lag)


def ref_horizon_range(window, horizon, delay, horizon_lag):
    offset = window + delay
    return np.arange(offset, offset + horizon, horizon_lag)


def ref_sample_span(window, horizon, delay):
    return max(window + delay + horizon, window)


def ref_index_values(n_steps, window, horizon, delay, stride):
    span = ref_sample_span(window, horizon, delay)
    return np.arange(0, n_steps - span + 1, stride)


def ref_n_samples(n_steps, window, horizon, delay, stride):
    return len(ref_index_values(n_steps, window, horizon, delay, stride))


def ref_window_steps(idx, window, window_lag):
    """Absolute window time steps for a sample whose index value is ``idx``."""
    return idx + ref_window_range(window, window_lag)


def ref_horizon_steps(idx, window, horizon, delay, horizon_lag):
    """Absolute horizon time steps for a sample whose index value is ``idx``."""
    return idx + ref_horizon_range(window, horizon, delay, horizon_lag)


# -- builders ---------------------------------------------------------------

def _make_dataset(window, horizon, stride=1, delay=0,
                  window_lag=1, horizon_lag=1, n_steps=200, **kwargs):
    return SpatioTemporalDataset(target=np.arange(n_steps).astype('float32'),
                                 window=window,
                                 horizon=horizon,
                                 stride=stride,
                                 delay=delay,
                                 window_lag=window_lag,
                                 horizon_lag=horizon_lag,
                                 **kwargs)


def _grid_target(n_steps, n_nodes, n_channels):
    """A target whose every (t, n, f) cell holds a unique value, so a retrieved
    tensor can be matched element-wise against the source by time step."""
    size = n_steps * n_nodes * n_channels
    return np.arange(size).reshape(n_steps, n_nodes, n_channels).astype(
        'float32')


def _make_grid_dataset(window, horizon, stride=1, delay=0, window_lag=1,
                       horizon_lag=1, n_steps=60, n_nodes=3, n_channels=2,
                       **kwargs):
    target = _grid_target(n_steps, n_nodes, n_channels)
    return SpatioTemporalDataset(target=target, window=window, horizon=horizon,
                                 stride=stride, delay=delay,
                                 window_lag=window_lag, horizon_lag=horizon_lag,
                                 **kwargs), target


def fitted_standard_scaler(target, axis=0):
    """A ``StandardScaler`` fitted on ``target`` (shared by the scaler and trend
    tests)."""
    scaler = StandardScaler(axis=axis)
    scaler.fit(torch.as_tensor(np.asarray(target)))
    return scaler


# Windowing grid for the foundation sweeps. ``window == 0`` exercises the
# horizon-only path (no 'window' key in expand_indices); large stride exercises
# the ceil rounding of samples_offset; lags exercise subsampling. Only configs
# with horizon_offset = window + delay >= 0 are kept (the supported range).
def _windowing_grid():
    configs = []
    for window in (0, 1, 4, 12):
        for horizon in (1, 2, 3, 12):
            for stride in (1, 2, 3):
                for delay in sorted({0, 1, 3, -1, -window}):
                    if window + delay < 0:
                        continue
                    for window_lag in (1, 2):
                        if window == 0 and window_lag != 1:
                            continue  # window_lag is irrelevant without a window
                        for horizon_lag in (1, 3):
                            configs.append((window, horizon, stride, delay,
                                            window_lag, horizon_lag))
    return configs


WINDOWING = _windowing_grid()


# Shared parametrization over the full windowing grid, so every sweep test
# unpacks the same (window, horizon, stride, delay, window_lag, horizon_lag)
# tuple and there is a single place to evolve the grid signature.
windowing_cases = pytest.mark.parametrize(
    'window,horizon,stride,delay,window_lag,horizon_lag', WINDOWING)
