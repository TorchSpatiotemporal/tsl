"""``tsl.datasets.prototypes.DatetimeDataset`` -- the datetime-indexed dataset:
chronological sorting, frequency inference / resampling, and the
``TemporalFeaturesMixin`` calendar encodings.
"""
import numpy as np
import pandas as pd
import pytest

from tsl.datasets.prototypes import DatetimeDataset

from .helpers import datetime_index, grid_dataframe, make_datetime


# -- sorting & frequency ----------------------------------------------------


def test_sort_orders_index_chronologically():
    idx = datetime_index(6, freq='1H')[::-1]  # reversed
    df = grid_dataframe(6, 2, 1, index=idx)
    ds = DatetimeDataset(target=df)  # sort_index defaults to True
    assert ds.index.is_monotonic_increasing
    # the value that was at the earliest timestamp is now first
    earliest = df.loc[idx.min()].values
    np.testing.assert_array_equal(ds.numpy()[0].ravel(), earliest)


def test_no_sort_keeps_order():
    idx = datetime_index(6, freq='1H')[::-1]
    df = grid_dataframe(6, 2, 1, index=idx)
    ds = DatetimeDataset(target=df, sort_index=False)
    assert not ds.index.is_monotonic_increasing


def test_default_similarity_score_allowed():
    # the documented default similarity_score=None must be accepted even though
    # DatetimeDataset declares similarity_options={'correntropy'}
    df = grid_dataframe(6, 2, 1, index=datetime_index(6, freq='1H'))
    ds = DatetimeDataset(target=df)
    assert ds.similarity_score is None
    # ...but actually computing a similarity without a method still errors
    with pytest.raises(ValueError):
        ds.get_similarity()


def test_freq_inferred_from_index():
    ds = make_datetime(24, freq='1H')
    assert ds.freq is not None
    assert ds.freq == pd.tseries.frequencies.to_offset('1H')


def test_forced_freq_resamples_at_init():
    # hourly values 0..3 on a single node/channel, summed into 2-hour bins
    df = grid_dataframe(4, 1, 1, index=datetime_index(4, freq='1H'))
    ds = DatetimeDataset(target=df, freq="2H")
    assert ds.length == 2
    np.testing.assert_allclose(ds.numpy().ravel(), [0 + 1, 2 + 3])


def test_resample_returns_copy():
    df = grid_dataframe(4, 1, 1, index=datetime_index(4, freq='1H'))
    ds = DatetimeDataset(target=df)
    res = ds.resample(freq='2H', aggr='sum')
    assert res.length == 2
    assert ds.length == 4  # original untouched


def test_resample_mask_tolerance():
    df = grid_dataframe(4, 1, 1, index=datetime_index(4, freq='1H'))
    ds = DatetimeDataset(target=df)
    mask = np.ones((4, 1, 1), dtype=bool)
    mask[0, 0, 0] = False  # first bin has one missing of two -> mean 0.5
    ds.set_mask(mask)
    res = ds.resample(freq='2H', aggr='sum', mask_tolerance=0.)
    # tolerance 0 -> a bin with any missing step is invalid
    assert not bool(res.mask[0, 0, 0])
    assert bool(res.mask[1, 0, 0])


# -- temporal feature encodings ---------------------------------------------


def test_datetime_encoded_shape_and_unit_circle():
    ds = make_datetime(48, freq='1H')
    enc = ds.datetime_encoded(['hour', 'day'])
    assert list(enc.columns) == ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    # sin^2 + cos^2 == 1 for each unit
    np.testing.assert_allclose(enc['hour_sin']**2 + enc['hour_cos']**2,
                               np.ones(len(enc)), atol=1e-5)


def test_datetime_encoded_matches_formula():
    from tsl.datasets.prototypes.casting import time_unit_to_nanoseconds
    ds = make_datetime(24, freq='1H')
    enc = ds.datetime_encoded('hour')
    nano = ds.index.tz_localize(None).astype('datetime64[ns]').view(np.int64)
    expected = np.sin(nano * (2 * np.pi / time_unit_to_nanoseconds('hour')))
    np.testing.assert_allclose(enc['hour_sin'].values, expected, atol=1e-5)


def test_datetime_onehot():
    ds = make_datetime(48, freq='1H')
    oh = ds.datetime_onehot('hour')
    assert oh.shape == (48, 24)
    # exactly one hot per row
    np.testing.assert_array_equal(oh.values.sum(1), np.ones(48))


def test_datetime_idx():
    ds = make_datetime(48, freq='1H')
    idx = ds.datetime_idx('hour')
    assert list(idx.columns) == ['hour']
    np.testing.assert_array_equal(idx['hour'].values, ds.index.hour.values)


def test_temporal_features_require_datetime_index():
    # a DatetimeDataset built on a non-datetime index must reject the encodings
    df = grid_dataframe(6, 2, 1, index=pd.RangeIndex(6))
    ds = DatetimeDataset(target=df)
    with pytest.raises(NotImplementedError):
        ds.datetime_encoded('hour')


def test_holidays_onehot():
    pytest.importorskip('holidays')  # optional dependency
    ds = make_datetime(48, freq='1H')
    out = ds.holidays_onehot('IT')
    assert list(out.columns) == ['holiday']
    assert len(out) == len(ds.index)


# -- similarity contract ----------------------------------------------------


def test_similarity_options_is_correntropy():
    assert DatetimeDataset.similarity_options == {'correntropy'}


def test_base_correntropy_not_implemented():
    # the prototype declares the option but leaves the computation to subclasses
    ds = make_datetime(24, freq='1H', similarity_score='correntropy')
    with pytest.raises(NotImplementedError):
        ds.get_similarity('correntropy')


def test_invalid_similarity_method_raises():
    ds = make_datetime(24, freq='1H', similarity_score='correntropy')
    with pytest.raises(ValueError):
        ds.get_similarity('not_a_method')
