from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn

from tsl.engines import Predictor
from tsl.nn.models import BaseModel, TimesFMModel


class DummyTimesFMForecaster:
    """Small deterministic stand-in for a TimesFM 3 forecaster."""

    def __init__(self):
        self.model = nn.Linear(1, 1)
        self.config = SimpleNamespace(quantiles=[0.1, 0.5, 0.9])
        self.calls = []

    def predict_batch(
        self,
        contexts,
        horizon,
        past_only_covariates=None,
        return_quantiles=False,
        **kwargs,
    ):
        self.calls.append(
            {
                'contexts': contexts,
                'past_only_covariates': past_only_covariates,
                'return_quantiles': return_quantiles,
                **kwargs,
            }
        )
        levels = np.asarray(self.config.quantiles, dtype=np.float32)
        for context in contexts:
            variables = context.shape[0]
            point = np.tile(np.arange(horizon, dtype=np.float32), (variables, 1))
            quantiles = point[..., None] + levels if return_quantiles else None
            yield SimpleNamespace(forecast=point, quantiles=quantiles)


@pytest.fixture
def timesfm_forecaster():
    forecasters = []

    def load_forecaster(model_id, **kwargs):
        forecaster = DummyTimesFMForecaster()
        forecasters.append(forecaster)
        return forecaster

    from_pretrained = Mock(side_effect=load_forecaster)
    from_pretrained.forecasters = forecasters
    timesfm = SimpleNamespace(
        TimesFM3Forecaster=SimpleNamespace(from_pretrained=from_pretrained)
    )
    with patch(
        'tsl.nn.models.foundation.timesfm_models.require_optional_dependency',
        return_value=timesfm,
    ):
        yield from_pretrained


@pytest.mark.timesfm
def test_timesfm_model_is_base_model(timesfm_forecaster):
    model = TimesFMModel(horizon=3)

    assert isinstance(model, BaseModel)
    timesfm_forecaster.assert_called_once_with('google/timesfm-3.0-pytorch')


@pytest.mark.timesfm
def test_timesfm_model_predictor_shape(timesfm_forecaster):
    predictor = Predictor(
        model_class=TimesFMModel,
        model_kwargs={'horizon': 3},
    )
    x = torch.randn(2, 8, 4, 2)

    output = predictor(x=x)

    assert output.shape == (2, 3, 4, 2)


@pytest.mark.timesfm
def test_timesfm_model_accepts_temporal_input(timesfm_forecaster):
    model = TimesFMModel(horizon=3)

    output = model(torch.randn(2, 8, 4))

    assert output.shape == (2, 3, 4)


@pytest.mark.timesfm
@pytest.mark.parametrize(
    ('nodes_as_covariates', 'num_items', 'num_variates'),
    [(False, 2 * 4, 3), (True, 2, 4 * 3)],
)
def test_timesfm_cross_learning_never_joins_batch_samples(
    timesfm_forecaster, nodes_as_covariates, num_items, num_variates
):
    model = TimesFMModel(horizon=3, nodes_as_covariates=nodes_as_covariates)
    x = torch.arange(2 * 8 * 4 * 3).reshape(2, 8, 4, 3).float()

    output = model(x)

    call = timesfm_forecaster.forecasters[-1].calls[-1]
    assert len(call['contexts']) == num_items
    assert all(context.shape == (num_variates, 8) for context in call['contexts'])
    if nodes_as_covariates:
        np.testing.assert_allclose(
            call['contexts'][0], x[0].permute(1, 2, 0).reshape(12, 8)
        )
        np.testing.assert_allclose(
            call['contexts'][1], x[1].permute(1, 2, 0).reshape(12, 8)
        )
    assert output.shape == (2, 3, 4, 3)


@pytest.mark.timesfm
@pytest.mark.parametrize('node_exogenous', [False, True])
@pytest.mark.parametrize('nodes_as_covariates', [False, True])
def test_timesfm_exogenous_features_follow_layout(
    timesfm_forecaster, node_exogenous, nodes_as_covariates
):
    model = TimesFMModel(horizon=3, nodes_as_covariates=nodes_as_covariates)
    x = torch.randn(2, 8, 4, 2)
    u = torch.randn(2, 8, 4, 3) if node_exogenous else torch.randn(2, 8, 3)

    output = model(x, u)

    call = timesfm_forecaster.forecasters[-1].calls[-1]
    expected_items = 2 if nodes_as_covariates else 2 * 4
    expected_covariates = 4 * 3 if nodes_as_covariates else 3
    assert len(call['contexts']) == expected_items
    assert len(call['past_only_covariates']) == expected_items
    assert all(
        covariates.shape == (expected_covariates, 8)
        for covariates in call['past_only_covariates']
    )
    assert output.shape == (2, 3, 4, 2)


@pytest.mark.timesfm
def test_timesfm_returns_quantiles_and_allows_override(timesfm_forecaster):
    model = TimesFMModel(
        horizon=2,
        quantile=0.9,
        return_quantiles=True,
        quantile_levels=[0.9, 0.1],
    )
    x = torch.randn(2, 8, 3)

    prediction, quantiles = model(x)

    assert prediction.shape == (2, 2, 3)
    assert quantiles.shape == (2, 2, 3, 2)
    torch.testing.assert_close(prediction, quantiles[..., 0])
    assert isinstance(model(x, return_quantiles=False), torch.Tensor)


@pytest.mark.timesfm
def test_timesfm_rejects_unsupported_quantile(timesfm_forecaster):
    with pytest.raises(ValueError, match='unavailable levels'):
        TimesFMModel(horizon=2, quantile_levels=[0.25])
