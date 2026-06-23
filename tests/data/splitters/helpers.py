import numpy as np
import pandas as pd

from tsl.data import SpatioTemporalDataset


# -- footprint helpers ------------------------------------------------------
# Footprints are computed from the dataset's own indexing.

def _steps(dataset, indices, which):
    if len(indices) == 0:
        return set()
    expanded = dataset.expand_indices(np.asarray(indices))
    return set(expanded[which].numpy().ravel().tolist())


def _targets(dataset, indices):
    """Time steps used as prediction targets (horizon) by the given samples."""
    return _steps(dataset, indices, 'horizon')


def _footprint(dataset, indices):
    """All time steps touched (input window + target horizon) by the samples."""
    steps = _steps(dataset, indices, 'horizon')
    if dataset.window > 0:
        steps |= _steps(dataset, indices, 'window')
    return steps


# -- dataset builders -------------------------------------------------------

def _make_dataset(window, horizon, stride=1, delay=0,
                  window_lag=1, horizon_lag=1, n_steps=1500):
    return SpatioTemporalDataset(target=np.arange(n_steps).astype('float32'),
                                 window=window,
                                 horizon=horizon,
                                 stride=stride,
                                 delay=delay,
                                 window_lag=window_lag,
                                 horizon_lag=horizon_lag)


def _make_dt_dataset(window, horizon, periods, freq='D', start='2019-01-01',
                     stride=1, delay=0):
    """Dataset backed by a tz-naive :class:`~pandas.DatetimeIndex`, as required
    by the time-based splitters (:class:`AtTimeStepSplitter`) and the
    ``indices_between`` / ``disjoint_months`` helpers."""
    index = pd.date_range(start, periods=periods, freq=freq)
    target = pd.DataFrame(np.arange(periods, dtype='float32'), index=index)
    return SpatioTemporalDataset(target=target,
                                 window=window,
                                 horizon=horizon,
                                 stride=stride,
                                 delay=delay)


# window/horizon/stride/delay grid for the leakage sweeps. ``window=0`` covers
# horizon-only datasets; large ``stride`` exercises the ceil rounding of the
# offset; delays span negative (down to -window, i.e. horizon_offset == 0) to
# large positive. Keep only horizon_offset = window + delay >= 0: a negative
# horizon_offset puts the horizon before the window (target steps < 0 for the
# first sample), which the dataset cannot represent and is outside the
# splitter's supported range.
CONFIGS = [(w, h, s, d)
           for w in (0, 1, 4, 12)
           for h in (1, 2, 3, 12, 24)
           for s in (1, 2, 3, 5)
           for d in sorted({-1, 0, 3, w, h, s, -w})
           if w + d >= 0]
