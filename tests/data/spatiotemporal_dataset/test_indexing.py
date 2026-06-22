"""Dataset indexing utils."""
import numpy as np
import pandas as pd
import pytest
import torch

from tsl.data import SpatioTemporalDataset
from tsl.data.synch_mode import HORIZON, WINDOW

from .helpers import (_make_dataset, ref_horizon_steps, ref_index_values,
                      ref_n_samples, ref_window_steps, windowing_cases)


@windowing_cases
def test_n_samples_matches_reference(window, horizon, stride, delay,
                                     window_lag, horizon_lag):
    n_steps = 200
    ds = _make_dataset(window, horizon, stride, delay, window_lag, horizon_lag,
                       n_steps=n_steps)
    expected = ref_n_samples(n_steps, window, horizon, delay, stride)
    assert ds.n_samples == expected
    assert len(ds) == expected
    # the stored index values must be exactly idx = p * stride
    np.testing.assert_array_equal(
        ds.indices.numpy(),
        ref_index_values(n_steps, window, horizon, delay, stride))


@windowing_cases
def test_expand_indices_matches_reference(window, horizon, stride, delay,
                                          window_lag, horizon_lag):
    ds = _make_dataset(window, horizon, stride, delay, window_lag, horizon_lag)
    expanded = ds.expand_indices()  # all samples
    idx_values = ref_index_values(200, window, horizon, delay, stride)

    # horizon is always present
    expected_horizon = np.stack([
        ref_horizon_steps(idx, window, horizon, delay, horizon_lag)
        for idx in idx_values
    ])
    np.testing.assert_array_equal(expanded['horizon'].numpy(),
                                  expected_horizon)

    if window > 0:
        expected_window = np.stack([
            ref_window_steps(idx, window, window_lag) for idx in idx_values
        ])
        np.testing.assert_array_equal(expanded['window'].numpy(),
                                      expected_window)
    else:
        # horizon-only datasets must not expose a window key
        assert 'window' not in expanded


@windowing_cases
def test_get_window_horizon_indices_single_and_batch(window, horizon, stride,
                                                     delay, window_lag,
                                                     horizon_lag):
    ds = _make_dataset(window, horizon, stride, delay, window_lag, horizon_lag)
    idx_values = ref_index_values(200, window, horizon, delay, stride)

    # single-item retrieval (scalar index -> 1-D result)
    for p in (0, ds.n_samples // 2, ds.n_samples - 1):
        idx = idx_values[p]
        np.testing.assert_array_equal(
            ds.get_horizon_indices(torch.tensor(p)).numpy(),
            ref_horizon_steps(idx, window, horizon, delay, horizon_lag))
        if window > 0:
            np.testing.assert_array_equal(
                ds.get_window_indices(torch.tensor(p)).numpy(),
                ref_window_steps(idx, window, window_lag))

    # batch retrieval (vector index -> 2-D result)
    batch = torch.arange(ds.n_samples)
    expected_h = np.stack([
        ref_horizon_steps(idx, window, horizon, delay, horizon_lag)
        for idx in idx_values
    ])
    np.testing.assert_array_equal(ds.get_horizon_indices(batch).numpy(),
                                  expected_h)


def test_expand_indices_merge_and_unique():
    ds = _make_dataset(window=4, horizon=2, delay=1)
    idx = np.array([0, 5])
    expanded = ds.expand_indices(idx)
    # merge=True returns the sorted unique union of window+horizon steps
    merged = ds.expand_indices(idx, merge=True).numpy()
    union = np.union1d(expanded['window'].numpy().ravel(),
                       expanded['horizon'].numpy().ravel())
    np.testing.assert_array_equal(merged, union)
    # unique=True returns sorted unique steps per role
    uniq = ds.expand_indices(idx, unique=True)
    np.testing.assert_array_equal(
        uniq['window'].numpy(),
        np.unique(expanded['window'].numpy()))
    np.testing.assert_array_equal(
        uniq['horizon'].numpy(),
        np.unique(expanded['horizon'].numpy()))


def _ref_overlapping(ds, idxs1, idxs2, which):
    """Independent reference for overlapping_indices: a sample overlaps iff any
    of its time steps (in the given role) is shared with the other set."""
    e1 = ds.expand_indices(np.asarray(idxs1))[which].numpy()
    e2 = ds.expand_indices(np.asarray(idxs2))[which].numpy()
    common = np.intersect1d(e1, e2)
    m1 = np.array([np.any(np.isin(row, common)) for row in e1])
    m2 = np.array([np.any(np.isin(row, common)) for row in e2])
    return m1, m2


@pytest.mark.parametrize('synch', [WINDOW, HORIZON])
def test_overlapping_indices(synch):
    ds = _make_dataset(window=6, horizon=3, delay=0)
    # two overlapping ranges of samples
    idxs1 = np.arange(0, 20)
    idxs2 = np.arange(15, 35)
    which = synch.name.lower()
    exp_m1, exp_m2 = _ref_overlapping(ds, idxs1, idxs2, which)

    o1, o2 = ds.overlapping_indices(idxs1, idxs2, synch_mode=synch,
                                    as_mask=False)
    np.testing.assert_array_equal(o1, idxs1[exp_m1])
    np.testing.assert_array_equal(o2, idxs2[exp_m2])

    m1, m2 = ds.overlapping_indices(idxs1, idxs2, synch_mode=synch,
                                    as_mask=True)
    np.testing.assert_array_equal(m1, exp_m1)
    np.testing.assert_array_equal(m2, exp_m2)


def test_data_timestamps_aligns_to_index():
    n_steps = 200
    index = pd.date_range('2020-01-01', periods=n_steps, freq='h')
    ds = SpatioTemporalDataset(target=np.arange(n_steps).astype('float32'),
                               index=index, window=4, horizon=2, delay=1)
    idx = np.array([0, 3, 10])
    ts = ds.data_timestamps(idx)
    expanded = ds.expand_indices(idx)
    # timestamps must be the index sampled at exactly the expanded steps
    for role in ('window', 'horizon'):
        expected = index.to_numpy()[expanded[role].numpy()]
        np.testing.assert_array_equal(ts[role], expected)


def test_data_timestamps_unique():
    n_steps = 200
    index = pd.date_range('2020-01-01', periods=n_steps, freq='h')
    ds = SpatioTemporalDataset(target=np.arange(n_steps).astype('float32'),
                               index=index, window=4, horizon=2, delay=1)
    idx = np.array([0, 1, 2])  # overlapping samples -> repeated steps
    ts = ds.data_timestamps(idx, unique=True)
    expanded = ds.expand_indices(idx, unique=True)
    # unique=True collapses the repeated steps to the sorted unique timestamps
    for role in ('window', 'horizon'):
        np.testing.assert_array_equal(ts[role],
                                      index[expanded[role].numpy()])


def test_data_timestamps_none_without_index():
    ds = _make_dataset(window=4, horizon=2)
    assert ds.data_timestamps() is None


def test_set_indices_bounds_reject():
    ds = _make_dataset(window=4, horizon=2, delay=0)  # sample_span = 6
    max_index = ds.n_steps - ds.sample_span
    # in-range indices are accepted and stored verbatim
    ds.set_indices([0, 5, max_index])
    np.testing.assert_array_equal(ds.indices.numpy(), [0, 5, max_index])
    # out-of-range (would read past the end of the target) must be rejected
    with pytest.raises(AssertionError):
        ds.set_indices([0, max_index + 1])
    with pytest.raises(AssertionError):
        ds.set_indices([-1, 0])
