"""Trend."""
import numpy as np
import torch

from tsl.data import SpatioTemporalDataset

from .helpers import _grid_target, fitted_standard_scaler


def test_set_trend_injects_into_scaler_bias_and_restores():
    target = _grid_target(60, 2, 1)
    sc = fitted_standard_scaler(target)
    orig_bias = sc.bias.clone()
    trend = np.full((60, 2, 1), 5.0, dtype='float32')

    ds = SpatioTemporalDataset(target=target, window=4, horizon=2,
                               scalers={'target': sc})
    ds.set_trend(trend)
    assert ds.trend is not None
    # the trend is combined with the fitted bias, not substituted for it
    assert torch.allclose(ds.scalers['target'].bias,
                          orig_bias + torch.as_tensor(trend))

    ds.set_trend(None)
    assert ds.trend is None
    assert torch.allclose(ds.scalers['target'].bias, orig_bias)


def test_trend_bias_independent_of_call_order():
    # set_trend-then-add_scaler must match add_scaler-then-set_trend
    target = _grid_target(60, 2, 1)
    orig_bias = fitted_standard_scaler(target).bias.clone()
    trend = np.full((60, 2, 1), 5.0, dtype='float32')

    a = SpatioTemporalDataset(target=target, window=4, horizon=2,
                              scalers={'target': fitted_standard_scaler(target)})
    a.set_trend(trend)

    b = SpatioTemporalDataset(target=target, window=4, horizon=2)
    b.set_trend(trend)
    b.add_scaler('target', fitted_standard_scaler(target))

    expected = orig_bias + torch.as_tensor(trend)
    assert torch.allclose(a.scalers['target'].bias, expected)
    assert torch.allclose(b.scalers['target'].bias, expected)


def test_trend_broadcasts_over_nodes():
    target = _grid_target(60, 3, 1)
    ds = SpatioTemporalDataset(target=target, window=4, horizon=2)
    ds.set_trend(np.ones((60, 1, 1), dtype='float32'))  # broadcastable on nodes
    assert tuple(ds.trend.shape) == (60, 1, 1)
