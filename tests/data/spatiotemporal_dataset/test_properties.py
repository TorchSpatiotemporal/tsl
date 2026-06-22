""" Init & properties (sources, precision, shape, name, index)."""
import numpy as np
import pandas as pd
import pytest
import torch

from tsl.data import SpatioTemporalDataset


@pytest.mark.parametrize('precision,expected', [
    (16, torch.float16), (32, torch.float32), (64, torch.float64),
    ('half', torch.float16), ('full', torch.float32), ('double', torch.float64),
])
def test_precision(precision, expected):
    ds = SpatioTemporalDataset(target=np.ones((10, 2, 1), dtype='float64'),
                               precision=precision, window=2, horizon=1)
    assert ds.target.dtype == expected


@pytest.mark.parametrize('target,expected_shape', [
    (np.arange(10).astype('float32'), (10, 1, 1)),               # 1-D -> t 1 1
    (np.arange(30).reshape(10, 3).astype('float32'), (10, 3, 1)),  # 2-D -> t n 1
    (np.arange(60).reshape(10, 3, 2).astype('float32'), (10, 3, 2)),  # t n f
    (torch.arange(30).reshape(10, 3, 1).float(), (10, 3, 1)),    # torch tensor
])
def test_sources_and_shape_promotion(target, expected_shape):
    ds = SpatioTemporalDataset(target=target, window=2, horizon=1)
    assert ds.shape == expected_shape
    assert ds.target.dtype == torch.float32
    assert (ds.n_steps, ds.n_nodes, ds.n_channels) == expected_shape


def test_dataframe_source_defaults_index():
    idx = pd.date_range('2020-01-01', periods=10, freq='h')
    df = pd.DataFrame(np.arange(30).reshape(10, 3).astype('float32'), index=idx)
    ds = SpatioTemporalDataset(target=df, window=2, horizon=1)
    assert ds.shape == (10, 3, 1)
    # the frame's DatetimeIndex becomes the dataset index
    assert isinstance(ds.index, pd.DatetimeIndex)
    assert ds.index.equals(idx)


def test_name_default_and_override():
    target = np.ones((5, 2, 1), dtype='float32')
    assert SpatioTemporalDataset(target=target, window=1,
                                 horizon=1).name == 'SpatioTemporalDataset'
    assert SpatioTemporalDataset(target=target, window=1, horizon=1,
                                 name='foo').name == 'foo'
