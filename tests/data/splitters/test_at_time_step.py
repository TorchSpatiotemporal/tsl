import numpy as np
import pytest

from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule.splitters import AtTimeStepSplitter

from .helpers import CONFIGS, _footprint, _make_dt_dataset, _targets


def _assert_pairwise_disjoint(dataset, idxs):
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    f_tr, f_va, f_te = (_footprint(dataset, tr), _footprint(dataset, va),
                        _footprint(dataset, te))
    assert f_tr.isdisjoint(f_va)
    assert f_va.isdisjoint(f_te)
    assert f_tr.isdisjoint(f_te)


# -- auto-defaulted training range is leak-free -----------------------------

def test_auto_default_is_completely_disjoint():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    splitter = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                                  last_val_ts=(2019, 10, 1),
                                  first_test_ts=(2019, 11, 1))
    idxs = splitter.split(dataset)
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    assert len(tr) and len(va) and len(te)
    _assert_pairwise_disjoint(dataset, idxs)
    # train ends exactly ``offset`` positions before validation starts
    offset = int(np.ceil(dataset.sample_span / dataset.stride))
    assert int(va.min()) - int(tr.max()) == offset


# Reuse the same window/horizon/stride/delay grid as the TemporalSplitter
# leakage sweep (see helpers.CONFIGS), run under both separation modes.
@pytest.mark.parametrize('min_offset', ['sample', 'window'])
@pytest.mark.parametrize('window,horizon,stride,delay', CONFIGS)
def test_separation_invariant_across_configs(window, horizon, stride, delay,
                                             min_offset):
    dataset = _make_dt_dataset(window, horizon, periods=1500, stride=stride,
                               delay=delay)
    splitter = AtTimeStepSplitter(first_val_ts=(2020, 1, 1),
                                  last_val_ts=(2020, 3, 1),
                                  first_test_ts=(2020, 7, 1),
                                  min_offset=min_offset)
    # 'window' refuses configs whose window gap cannot cover the horizon.
    if min_offset == 'window' and \
            dataset.samples_offset * stride < horizon:
        with pytest.raises(AssertionError):
            splitter.split(dataset)
        return
    idxs = splitter.split(dataset)
    if not all(len(idxs[k]) for k in ('train', 'val', 'test')):
        pytest.skip('degenerate split for this config')
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    # the auto train range ends exactly ``offset`` positions before validation
    expected = (int(np.ceil(dataset.sample_span / stride))
                if min_offset == 'sample' else dataset.samples_offset)
    assert int(va.min()) - int(tr.max()) == expected
    if min_offset == 'sample':
        # no time step shared in any role
        _assert_pairwise_disjoint(dataset, idxs)
    else:
        # windows may touch, but no evaluation target appears in training
        train_fp = _footprint(dataset, tr)
        assert _targets(dataset, va).isdisjoint(train_fp)
        assert _targets(dataset, te).isdisjoint(train_fp | _footprint(dataset,
                                                                      va))


# -- offset='window' --------------------------------------------------------

def test_window_offset_rejects_leaky_config():
    # horizon (24) > window (6): the window gap cannot cover the horizon, so it
    # must refuse rather than silently leak (mirrors TemporalSplitter).
    dataset = _make_dt_dataset(window=6, horizon=24, periods=400)
    splitter = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                                  last_val_ts=(2019, 10, 1),
                                  first_test_ts=(2019, 11, 1),
                                  min_offset='window')
    with pytest.raises(AssertionError):
        splitter.split(dataset)


def test_window_offset_explicit_overlap_raises():
    dataset = _make_dt_dataset(window=12, horizon=12, periods=400)
    splitter = AtTimeStepSplitter(last_train_ts=(2019, 9, 1),
                                  first_val_ts=(2019, 9, 1),
                                  last_val_ts=(2019, 10, 1),
                                  first_test_ts=(2019, 11, 1),
                                  min_offset='window')
    with pytest.raises(ValueError, match="'train' and 'val'"):
        splitter.split(dataset)


def test_unknown_offset_raises():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    splitter = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                                  first_test_ts=(2019, 11, 1),
                                  min_offset='nope')
    with pytest.raises(ValueError, match='Unknown offset'):
        splitter.split(dataset)


# -- explicit training range ------------------------------------------------

def test_explicit_non_overlapping_train_range():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter(first_train_ts=(2019, 1, 1),
                              last_train_ts=(2019, 8, 1),
                              first_val_ts=(2019, 9, 1),
                              last_val_ts=(2019, 10, 1),
                              first_test_ts=(2019, 11, 1)).split(dataset)
    assert all(len(idxs[k]) for k in ('train', 'val', 'test'))
    _assert_pairwise_disjoint(dataset, idxs)


def test_explicit_overlapping_train_range_raises():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    # last_train_ts lands on the validation start -> footprints overlap
    splitter = AtTimeStepSplitter(last_train_ts=(2019, 9, 1),
                                  first_val_ts=(2019, 9, 1),
                                  last_val_ts=(2019, 10, 1),
                                  first_test_ts=(2019, 11, 1))
    with pytest.raises(ValueError, match="'train' and 'val'"):
        splitter.split(dataset)


def test_val_test_overlap_raises():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    # validation runs right up to the test start -> footprints overlap
    splitter = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                                  last_val_ts=(2019, 11, 1),
                                  first_test_ts=(2019, 11, 1))
    with pytest.raises(ValueError, match="'val' and 'test'"):
        splitter.split(dataset)


def test_first_train_ts_only_trims_start():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    full = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                              last_val_ts=(2019, 10, 1),
                              first_test_ts=(2019, 11, 1)).split(dataset)
    trimmed = AtTimeStepSplitter(first_train_ts=(2019, 3, 1),
                                 first_val_ts=(2019, 9, 1),
                                 last_val_ts=(2019, 10, 1),
                                 first_test_ts=(2019, 11, 1)).split(dataset)
    # a later training start yields fewer training samples, same end
    assert len(trimmed['train']) < len(full['train'])
    assert int(trimmed['train'].min()) > int(full['train'].min())
    assert int(trimmed['train'].max()) == int(full['train'].max())
    _assert_pairwise_disjoint(dataset, trimmed)


def test_last_train_ts_only_sets_end():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter(last_train_ts=(2019, 6, 1),
                              first_val_ts=(2019, 9, 1),
                              last_val_ts=(2019, 10, 1),
                              first_test_ts=(2019, 11, 1)).split(dataset)
    assert int(idxs['train'].min()) == 0
    _assert_pairwise_disjoint(dataset, idxs)


# -- partial / missing holdout sets -----------------------------------------

def test_only_test_specified():
    # validation omitted -> empty val, train fills up to the test split
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter(first_test_ts=(2019, 11, 1)).split(dataset)
    assert len(idxs['val']) == 0
    assert len(idxs['train']) and len(idxs['test'])
    f_tr, f_te = _footprint(dataset, idxs['train']), _footprint(dataset,
                                                               idxs['test'])
    assert f_tr.isdisjoint(f_te)


def test_only_val_specified():
    # test omitted -> empty test, train fills up to the validation split
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                              last_val_ts=(2019, 10, 1)).split(dataset)
    assert len(idxs['test']) == 0
    assert len(idxs['train']) and len(idxs['val'])
    f_tr, f_va = _footprint(dataset, idxs['train']), _footprint(dataset,
                                                               idxs['val'])
    assert f_tr.isdisjoint(f_va)


def test_no_holdout_gives_train_whole_series():
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter().split(dataset)
    assert len(idxs['train']) == dataset.n_samples
    assert len(idxs['val']) == 0 and len(idxs['test']) == 0


def test_open_ended_val_inferred_from_test():
    # last_val_ts omitted: instead of running to the series end (and overlapping
    # test), validation is inferred to end ``offset`` before the test split.
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                              first_test_ts=(2019, 11, 1)).split(dataset)
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    assert all(len(s) for s in (tr, va, te))
    # val now reaches up to the inferred boundary, separated from test by offset
    offset = int(np.ceil(dataset.sample_span / dataset.stride))
    assert int(te.min()) - int(va.max()) == offset
    _assert_pairwise_disjoint(dataset, idxs)


def test_open_ended_val_extends_to_series_end_without_test():
    # with no test split to bound it, an open-ended validation runs to the end
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter(first_val_ts=(2019, 9, 1)).split(dataset)
    assert int(idxs['val'].max()) == dataset.n_samples - 1
    assert len(idxs['test']) == 0
    _assert_pairwise_disjoint(dataset, idxs)


def test_open_started_test_inferred_from_val():
    # first_test_ts omitted: the test start is inferred ``offset`` after the
    # validation end rather than running back to the series start.
    dataset = _make_dt_dataset(window=4, horizon=4, periods=400)
    idxs = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                              last_val_ts=(2019, 10, 1),
                              last_test_ts=(2019, 12, 1)).split(dataset)
    tr, va, te = idxs['train'], idxs['val'], idxs['test']
    assert all(len(s) for s in (tr, va, te))
    offset = int(np.ceil(dataset.sample_span / dataset.stride))
    assert int(te.min()) - int(va.max()) == offset
    _assert_pairwise_disjoint(dataset, idxs)


# -- input validation -------------------------------------------------------

def test_non_datetime_index_raises():
    dataset = SpatioTemporalDataset(target=np.arange(400, dtype='float32'),
                                    window=4, horizon=4)
    splitter = AtTimeStepSplitter(first_val_ts=(2019, 9, 1),
                                  first_test_ts=(2019, 11, 1))
    with pytest.raises(ValueError, match='DatetimeIndex'):
        splitter.split(dataset)
