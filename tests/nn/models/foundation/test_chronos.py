from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from tsl.engines import Predictor
from tsl.nn.models import BaseModel, ChronosModel


class DummyChronosPipeline:
    """Small deterministic stand-in for a Chronos pipeline."""

    def __init__(self):
        self.inner_model = nn.Linear(1, 1)
        self.last_inputs = None
        self.last_kwargs = None
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.last_inputs = inputs
        self.last_kwargs = kwargs
        self.calls.append((inputs, kwargs))
        n_series = inputs.shape[0]
        point = torch.arange(prediction_length).repeat(n_series, 1).float()
        levels = torch.tensor(quantile_levels).reshape(1, 1, -1)
        quantiles = point.unsqueeze(-1) + levels
        return quantiles, point


class Chronos2Pipeline(DummyChronosPipeline):
    """Stand-in preserving Chronos-2's per-task list output."""

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.last_inputs = inputs
        self.last_kwargs = kwargs
        self.calls.append((inputs, kwargs))
        point = []
        quantiles = []
        levels = torch.tensor(quantile_levels).reshape(1, 1, -1)
        for sample in inputs:
            n_variates = sample.shape[0]
            sample_point = torch.arange(prediction_length).repeat(n_variates, 1).float()
            point.append(sample_point)
            quantiles.append(sample_point.unsqueeze(-1) + levels)
        return quantiles, point


@pytest.fixture
def chronos_pipeline():
    pipelines = []

    def load_pipeline(model_id, **kwargs):
        pipeline = (
            Chronos2Pipeline()
            if model_id == 'amazon/chronos-2'
            else DummyChronosPipeline()
        )
        pipelines.append(pipeline)
        return pipeline

    from_pretrained = Mock(side_effect=load_pipeline)
    from_pretrained.pipelines = pipelines
    chronos = SimpleNamespace(
        BaseChronosPipeline=SimpleNamespace(from_pretrained=from_pretrained)
    )
    with patch(
        'tsl.nn.models.foundation.chronos_models.require_optional_dependency',
        return_value=chronos,
    ):
        yield from_pretrained


@pytest.mark.chronos
def test_chronos_model_is_base_model(chronos_pipeline):
    model = ChronosModel(horizon=3, model_id='amazon/chronos-bolt-tiny')

    assert isinstance(model, BaseModel)
    chronos_pipeline.assert_called_once_with('amazon/chronos-bolt-tiny')


@pytest.mark.chronos
def test_chronos_model_predictor_shape(chronos_pipeline):
    predictor = Predictor(
        model_class=ChronosModel,
        model_kwargs={'horizon': 3},
    )
    x = torch.randn(2, 8, 4, 2)

    output = predictor(x=x)

    assert output.shape == (2, 3, 4, 2)
    expected = torch.arange(3).repeat(2, 1).float()
    torch.testing.assert_close(output[:, :, 0, 0], expected)


@pytest.mark.chronos
def test_chronos_model_accepts_temporal_input(chronos_pipeline):
    model = ChronosModel(horizon=3)
    x = torch.randn(2, 8, 4)

    output = model(x)

    assert output.shape == (2, 3, 4)


@pytest.mark.chronos
@pytest.mark.parametrize('node_exogenous', [False, True])
def test_chronos_model_concatenates_exogenous_features(
    chronos_pipeline, node_exogenous
):
    model = ChronosModel(horizon=3)
    x = torch.randn(2, 8, 4, 2)
    u = torch.randn(2, 8, 4, 3) if node_exogenous else torch.randn(2, 8, 3)

    output = model(x, u)

    pipeline = chronos_pipeline.pipelines[-1]
    assert pipeline.last_inputs.shape == (2 * 4 * (2 + 3), 8)
    assert output.shape == (2, 3, 4, 2)


@pytest.mark.chronos
@pytest.mark.parametrize('nodes_as_covariates', [False, True])
def test_chronos2_keeps_batch_samples_independent(
    chronos_pipeline, nodes_as_covariates
):
    model = ChronosModel(
        horizon=3,
        model_id='amazon/chronos-2',
        nodes_as_covariates=nodes_as_covariates,
    )
    x = torch.arange(2 * 8 * 4 * 3).reshape(2, 8, 4, 3).float()

    output = model(x)

    pipeline = chronos_pipeline.pipelines[-1]
    assert len(pipeline.calls) == 2
    for batch_index, (inputs, kwargs) in enumerate(pipeline.calls):
        assert len(inputs) == 4
        assert all(node.shape == (3, 8) for node in inputs)
        torch.testing.assert_close(inputs[0], x[batch_index, :, 0].T)
        assert kwargs['cross_learning'] is nodes_as_covariates
    assert output.shape == (2, 3, 4, 3)


@pytest.mark.chronos
def test_chronos2_exogenous_features_follow_layout(chronos_pipeline):
    model = ChronosModel(horizon=3, model_id='amazon/chronos-2')
    x = torch.randn(2, 8, 4, 2)
    u = torch.randn(2, 8, 3)

    output = model(x, u)

    pipeline = chronos_pipeline.pipelines[-1]
    assert len(pipeline.calls) == 2
    assert all(len(inputs) == 4 for inputs, _ in pipeline.calls)
    assert all(
        node.shape == (2 + 3, 8) for inputs, _ in pipeline.calls for node in inputs
    )
    assert output.shape == (2, 3, 4, 2)


@pytest.mark.chronos
def test_chronos_model_quantile(chronos_pipeline):
    model = ChronosModel(horizon=2, quantile=0.95)
    x = torch.randn(1, 8, 2, 1)

    output = model(x)

    expected = torch.tensor([0.95, 1.95]).reshape(1, 2, 1, 1).repeat(1, 1, 2, 1)
    torch.testing.assert_close(output, expected)


@pytest.mark.chronos
def test_chronos_model_returns_quantiles_and_allows_override(chronos_pipeline):
    model = ChronosModel(
        horizon=2,
        return_quantiles=True,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    x = torch.randn(2, 8, 3)

    prediction, quantiles = model(x)

    assert prediction.shape == (2, 2, 3)
    assert quantiles.shape == (2, 2, 3, 3)
    assert isinstance(model(x, return_quantiles=False), torch.Tensor)


@pytest.mark.chronos
def test_chronos_model_forward_enables_quantiles(chronos_pipeline):
    model = ChronosModel(horizon=2)
    x = torch.randn(1, 8, 2, 1)

    prediction, quantiles = model(x, return_quantiles=True)

    assert prediction.shape == (1, 2, 2, 1)
    assert quantiles.shape == (1, 2, 2, 1, 9)


@pytest.mark.chronos
def test_chronos2_returns_quantiles_in_requested_order(chronos_pipeline):
    model = ChronosModel(
        horizon=2,
        model_id='amazon/chronos-2',
        quantile=0.5,
        return_quantiles=True,
        quantile_levels=[0.9, 0.1],
    )
    x = torch.randn(2, 8, 3, 1)

    prediction, quantiles = model(x)

    assert prediction.shape == (2, 2, 3, 1)
    assert quantiles.shape == (2, 2, 3, 1, 2)
    torch.testing.assert_close(
        quantiles[..., 0] - quantiles[..., 1], torch.full_like(prediction, 0.8)
    )
