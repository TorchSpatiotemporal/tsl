import numpy as np
import pytest

from tsl.data import SpatioTemporalDataset, TemporalSplitter


# -- helpers ----------------------------------------------------------------
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


def _make_dataset(window, horizon, stride=1, delay=0,
                  window_lag=1, horizon_lag=1, n_steps=1500):
    return SpatioTemporalDataset(target=np.arange(n_steps).astype('float32'),
                                 window=window,
                                 horizon=horizon,
                                 stride=stride,
                                 delay=delay,
                                 window_lag=window_lag,
                                 horizon_lag=horizon_lag)


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


# -- exact-count test (independent reference computation) -------------------

def test_temporal_splitter():
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2, offset='window')
    # create a dummy sequence
    seq = np.arange(100)
    # create dummy SpatiotemporalDataset
    window = 3
    horizon = 3

    dataset = SpatioTemporalDataset(
        target=seq,
        window=window,
        horizon=horizon,
    )
    n_samples = 100 - window - horizon + 1
    idxs = splitter.split(dataset)
    # With the default offset='window', the closest train/val samples are
    # samples_offset = ceil(window / stride) = 3 positions apart, so the slice
    # drops samples_offset - 1 = 2 samples between splits.
    gap = window - 1
    # check that the split is correct
    assert len(dataset) == n_samples
    # val_len = int(0.1 * (95 - 19)) = 7, test_len = int(0.2 * 95) = 19
    # test_start = 76, val_start = 69
    assert len(idxs['train']) == 69 - gap
    assert len(idxs['val']) == 7 - gap
    assert len(idxs['test']) == 19
    # repeat using integer val/test len
    splitter = TemporalSplitter(val_len=8, test_len=20)
    idxs = splitter.split(dataset)
    # test_start = 75, val_start = 67
    assert len(idxs['train']) == 67 - gap
    assert len(idxs['val']) == 8 - gap
    assert len(idxs['test']) == 20


# -- offset='sample': fully disjoint footprints -----------------------------

@pytest.mark.parametrize('window,horizon,stride,delay', CONFIGS)
def test_sample_offset_fully_disjoint(window, horizon, stride, delay):
    dataset = _make_dataset(window, horizon, stride, delay)
    idxs = TemporalSplitter(0.1, 0.2, offset='sample').split(dataset)
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    if not (len(tr) and len(va) and len(te)):
        pytest.skip('degenerate split for this config')
    f_tr, f_va, f_te = (_footprint(dataset, tr),
                        _footprint(dataset, va),
                        _footprint(dataset, te))
    # no time step shared between splits in any role (input or target)
    assert f_tr.isdisjoint(f_va)
    assert f_va.isdisjoint(f_te)
    assert f_tr.isdisjoint(f_te)


@pytest.mark.parametrize('window,horizon,stride,delay',
                         [(12, 12, 1, 0), (12, 6, 2, 0),
                          (6, 12, 1, 3), (4, 4, 3, -4)])  # last: horizon_offset==0
@pytest.mark.parametrize('window_lag,horizon_lag', [(2, 1), (1, 3), (2, 3)])
def test_sample_offset_disjoint_with_lags(window, horizon, stride, delay,
                                          window_lag, horizon_lag):
    dataset = _make_dataset(window, horizon, stride, delay,
                            window_lag=window_lag, horizon_lag=horizon_lag)
    idxs = TemporalSplitter(0.1, 0.2, offset='sample').split(dataset)
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    if not (len(tr) and len(va) and len(te)):
        pytest.skip('degenerate split for this config')
    f_tr, f_va, f_te = (_footprint(dataset, tr),
                        _footprint(dataset, va),
                        _footprint(dataset, te))
    assert f_tr.isdisjoint(f_va)
    assert f_va.isdisjoint(f_te)
    assert f_tr.isdisjoint(f_te)


# -- offset='window': never leaks targets when it accepts a config ----------

@pytest.mark.parametrize('window,horizon,stride,delay', CONFIGS)
def test_window_offset_never_leaks_targets(window, horizon, stride, delay):
    dataset = _make_dataset(window, horizon, stride, delay)
    try:
        idxs = TemporalSplitter(0.1, 0.2, offset='window').split(dataset)
    except AssertionError:
        return  # window-sized gap can't cover the horizon -> correctly refused
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    if not (len(tr) and len(va) and len(te)):
        pytest.skip('degenerate split for this config')
    # an evaluation target must never appear in training, as target OR input
    train_fp = _footprint(dataset, tr)
    assert _targets(dataset, va).isdisjoint(train_fp)
    assert _targets(dataset, te).isdisjoint(train_fp | _footprint(dataset, va))


def test_window_offset_rejects_leaky_config():
    # horizon (24) > window (6) at stride 1: the window gap cannot separate the
    # horizons, so it must refuse rather than silently leak.
    dataset = _make_dataset(window=6, horizon=24)
    with pytest.raises(AssertionError):
        TemporalSplitter(0.1, 0.2, offset='window').split(dataset)


def test_window_offset_accepts_safe_config():
    # horizon == window at stride 1 is exactly the safe boundary.
    dataset = _make_dataset(window=12, horizon=12)
    idxs = TemporalSplitter(0.1, 0.2, offset='window').split(dataset)
    assert len(idxs['train']) and len(idxs['val']) and len(idxs['test'])


@pytest.mark.parametrize('horizon', [2, 4, 12])
def test_window_offset_boundary_is_horizon_eq_window(horizon):
    # stride 1: safe iff horizon <= window. window == horizon passes,
    # window == horizon - 1 fails.
    TemporalSplitter(0.1, 0.2, offset='window').split(
        _make_dataset(window=horizon, horizon=horizon))
    with pytest.raises(AssertionError):
        TemporalSplitter(0.1, 0.2, offset='window').split(
            _make_dataset(window=horizon - 1, horizon=horizon))


# -- edge cases -------------------------------------------------------------

def test_unknown_offset_raises():
    with pytest.raises(ValueError):
        TemporalSplitter(0.1, 0.2, offset='nope').split(_make_dataset(4, 2))


def test_window_zero_handled():
    # window=0: 'window' mode cannot separate anything -> must refuse (and never
    # reach the would-be-negative slice); 'sample' mode still works.
    dataset = _make_dataset(window=0, horizon=2)
    with pytest.raises(AssertionError):
        TemporalSplitter(0.1, 0.2, offset='window').split(dataset)
    idxs = TemporalSplitter(0.1, 0.2, offset='sample').split(dataset)
    f_tr = _footprint(dataset, idxs['train'])
    f_va = _footprint(dataset, idxs['val'])
    f_te = _footprint(dataset, idxs['test'])
    assert f_tr.isdisjoint(f_va) and f_va.isdisjoint(f_te)
