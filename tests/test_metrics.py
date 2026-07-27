from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tsl.metrics.numpy.functional as npf
import tsl.metrics.torch.functional as trf
from tsl.data import StaticBatch
from tsl.engines.predictor import Predictor
from tsl.metrics.torch import MaskedPinballLoss
from tsl.metrics.torch.metrics import (
    MaskedMAE,
    MaskedMAPE,
    MaskedMRE,
    MaskedMSE,
    MaskedSMAPE,
)
from tsl.nn.models.temporal.tcn_model import TCNModel
from tsl.ops.framearray import framearray_to_numpy
from tsl.utils.casting import torch_to_numpy

DELTA = 1e-6


@pytest.fixture(scope='module')
def metric_context():
    torch.manual_seed(0)
    x = 1.0 + torch.arange(2 * 8 * 2 * 2,
                           dtype=torch.float32).reshape(2, 8, 2, 2)
    y = 1.0 + torch.arange(2 * 8 * 2 * 4,
                           dtype=torch.float32).reshape(2, 8, 2, 4)
    mask = torch.tensor([0., 1.], dtype=torch.float32).repeat(2, 8, 2, 2)
    mask = mask.reshape(2, 8, 2, 4)

    predictor = Predictor(
        model_class=TCNModel,
        model_kwargs={'input_size': 2, 'output_size': 4, 'horizon': 8},
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        loss_fn=MaskedMAE(),
        scale_target=False,
        metrics=dict(
            mae=MaskedMAE(),
            mape=MaskedMAPE(),
            smape=MaskedSMAPE(),
            mre=MaskedMRE(),
            mse=MaskedMSE(),
            pinball=MaskedPinballLoss(q=0.75),
        ),
    )

    batch = StaticBatch(input={'x': x}, target={'y': y}, mask=mask)
    y_hat = predictor.predict_batch(batch, preprocess=False,
                                    postprocess=True).detach()
    y, mask = batch.y, batch.get('mask')

    metrics_res = predictor.test_metrics(y_hat, y)
    predictor.test_metrics.reset()
    masked_metrics_res = predictor.test_metrics(y_hat, y, mask)
    predictor.test_metrics.reset()

    return SimpleNamespace(y_hat=y_hat,
                           y=y,
                           mask=mask,
                           metrics_res=metrics_res,
                           masked_metrics_res=masked_metrics_res)


def _numpy_context(metric_context):
    return (torch_to_numpy(metric_context.y_hat),
            torch_to_numpy(metric_context.y),
            torch_to_numpy(metric_context.mask))


def _tensor_context(metric_context):
    return (metric_context.y_hat.clone(), metric_context.y.clone(),
            metric_context.mask.clone())


def test_mae_metric(metric_context):
    y_hat_, y_, _ = _numpy_context(metric_context)
    res = npf.mae(y_hat_, y_)
    assert np.isclose(metric_context.metrics_res['test_mae'], res, atol=DELTA)


def test_mae_masked_metric(metric_context):
    y_hat_, y_, mask_ = _numpy_context(metric_context)
    res = npf.mae(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(metric_context.masked_metrics_res['test_mae'],
                      res,
                      atol=DELTA)


def test_mse_metric(metric_context):
    y_hat_, y_, _ = _numpy_context(metric_context)
    res = npf.mse(y_hat_, y_)
    assert np.isclose(metric_context.metrics_res['test_mse'], res, atol=DELTA)


def test_mse_masked_metric(metric_context):
    y_hat_, y_, mask_ = _numpy_context(metric_context)
    res = npf.mse(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(metric_context.masked_metrics_res['test_mse'],
                      res,
                      atol=DELTA)


def test_mape_metric(metric_context):
    y_hat_, y_, _ = _numpy_context(metric_context)
    res = npf.mape(y_hat_, y_)
    assert np.isclose(metric_context.metrics_res['test_mape'], res, atol=DELTA)


def test_smape_metric(metric_context):
    y_hat_, y_, _ = _numpy_context(metric_context)
    res = npf.smape(y_hat_, y_)
    assert np.isclose(metric_context.metrics_res['test_smape'], res, atol=DELTA)


def test_mape_masked_metric(metric_context):
    y_hat_, y_, mask_ = _numpy_context(metric_context)
    res = npf.mape(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(metric_context.masked_metrics_res['test_mape'],
                      res,
                      atol=DELTA)


def test_smape_masked_metric(metric_context):
    y_hat_, y_, mask_ = _numpy_context(metric_context)
    res = npf.smape(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(metric_context.masked_metrics_res['test_smape'],
                      res,
                      atol=DELTA)


def test_mre_metric(metric_context):
    y_hat_, y_, _ = _numpy_context(metric_context)
    res = npf.mre(y_hat_, y_)
    assert np.isclose(metric_context.metrics_res['test_mre'], res, atol=DELTA)


def test_mre_masked_metric(metric_context):
    y_hat_, y_, mask_ = _numpy_context(metric_context)
    res = npf.mre(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(metric_context.masked_metrics_res['test_mre'],
                      res,
                      atol=DELTA)


def test_mae_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.mae(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mae(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mae_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.mae(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.mae(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_mse_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.mse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mse(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mse_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.mse(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.mse(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_mape_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.mape(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mape(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mape_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.mape(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.mape(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_smape_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.smape(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.smape(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_smape_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.smape(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.smape(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_mre_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.mre(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mre(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mre_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.mre(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.mre(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_rmse_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.rmse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.rmse(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_rmse_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.rmse(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.rmse(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_nrmse_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.nrmse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.nrmse(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_nrmse_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.nrmse(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.nrmse(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_nrmse2_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.nrmse_2(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.nrmse_2(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_nrmse2_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.nrmse_2(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.nrmse_2(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_r2_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.r2(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.r2(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_r2_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.r2(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.r2(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_nmae_functional(metric_context):
    y_hat_, y_, _ = _tensor_context(metric_context)
    res_np = npf.nmae(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.nmae(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_nmae_masked_functional(metric_context):
    y_hat_, y_, mask_ = _tensor_context(metric_context)
    res_np = npf.nmae(
        framearray_to_numpy(y_hat_), framearray_to_numpy(y_), framearray_to_numpy(mask_)
    )
    res_torch = trf.nmae(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)
