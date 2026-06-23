from datetime import datetime

import numpy as np
import pytest

from tsl.data import AtTimeStepSplitter, TemporalSplitter
from tsl.data.datamodule.splitters import (at_ts, disjoint_months,
                                            indices_between, temporal)
from tsl.data.synch_mode import SynchMode

from .helpers import _make_dt_dataset, _steps


# -- indices_between --------------------------------------------------------

def test_indices_between_bounded_range():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idx = indices_between(dataset, first_ts=(2019, 3, 1), last_ts=(2019, 5, 1))
    assert len(idx) == 61
    assert idx.min() == 55
    assert idx.max() == 115


def test_indices_between_open_bounds_are_supersets():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    bounded = set(indices_between(dataset, first_ts=(2019, 3, 1),
                                  last_ts=(2019, 5, 1)).tolist())
    first_only = set(indices_between(dataset, first_ts=(2019, 3, 1)).tolist())
    last_only = set(indices_between(dataset, last_ts=(2019, 5, 1)).tolist())
    assert bounded <= first_only
    assert bounded <= last_only


def test_indices_between_no_bounds_returns_all():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idx = indices_between(dataset)
    assert len(idx) == len(dataset)


def test_indices_between_accepts_datetime_objects():
    # tuples and datetime objects must map to the same index range.
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    as_tuples = indices_between(dataset, first_ts=(2019, 3, 1),
                                last_ts=(2019, 5, 1))
    as_dt = indices_between(dataset, first_ts=datetime(2019, 3, 1),
                            last_ts=datetime(2019, 5, 1))
    assert np.array_equal(np.asarray(as_tuples), np.asarray(as_dt))


# -- disjoint_months --------------------------------------------------------

def _ref_disjoint_months(dataset, months, synch_mode):
    """Independent reference: a sample is ``after`` when the first AND last step
    of its synch span fall in ``months``, ``prev`` when both fall outside, and
    is dropped otherwise."""
    idx = np.asarray(dataset._indices)
    if synch_mode is SynchMode.WINDOW:
        start, end = 0, dataset.window - 1
    else:
        start = dataset.horizon_offset
        end = dataset.horizon_offset + dataset.horizon - 1
    in_start = np.isin(dataset.index[idx + start].month, months)
    in_end = np.isin(dataset.index[idx + end].month, months)
    pos = np.arange(len(dataset))
    return pos[~in_start & ~in_end], pos[in_start & in_end]


@pytest.mark.parametrize('window,horizon', [(4, 4), (12, 4), (4, 12), (12, 1)])
@pytest.mark.parametrize('months,synch_mode',
                         [([1, 2], SynchMode.WINDOW),
                          ([7], SynchMode.HORIZON),
                          ([6, 12], SynchMode.WINDOW)])
def test_disjoint_months_partition_is_exact(months, synch_mode, window,
                                            horizon):
    dataset = _make_dt_dataset(window=window, horizon=horizon, periods=400)
    prev, after = disjoint_months(dataset, months=months, synch_mode=synch_mode)
    ref_prev, ref_after = _ref_disjoint_months(dataset, months, synch_mode)
    # the two groups never share a sample
    assert set(prev).isdisjoint(set(after))
    # exact match: every in-month sample is returned (completeness) and only
    # in-month samples are returned (soundness); same for the complement
    assert set(after) == set(ref_after.tolist())
    assert set(prev) == set(ref_prev.tolist())
    # and they share no time step in the synch role they are split on: the
    # window (resp. horizon) footprints of the two groups are disjoint
    role = 'window' if synch_mode is SynchMode.WINDOW else 'horizon'
    assert _steps(dataset, prev, role).isdisjoint(_steps(dataset, after, role))


def test_disjoint_months_accepts_single_int_month():
    # ``months`` is run through ``ensure_list``, so a bare int is accepted.
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    prev, after = disjoint_months(dataset, months=6)
    assert set(prev).isdisjoint(set(after))
    indices = np.asarray(dataset._indices)[after]
    assert (dataset.index[indices].month == 6).all()


def test_disjoint_months_invalid_synch_mode_raises():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    with pytest.raises(ValueError):
        disjoint_months(dataset, months=[1], synch_mode='nope')


# -- aliases ----------------------------------------------------------------

def test_aliases():
    # ``Dataset.get_splitter`` resolves a method string via ``getattr`` on this
    # module, so these aliases are the registry entries for 'temporal'/'at_ts'.
    assert temporal is TemporalSplitter
    assert at_ts is AtTimeStepSplitter
