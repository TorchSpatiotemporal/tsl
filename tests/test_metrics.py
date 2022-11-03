import pytest

import torch
import numpy as np

from tsl.data import Batch
from tsl.inference.predictor import Predictor
from tsl.metrics.torch import MaskedPinballLoss
from tsl.nn.models.temporal.tcn_model import TCNModel
from tsl.metrics.torch.metrics import MaskedMAE, MaskedMSE, MaskedMAPE, MaskedMRE
from tsl.nn.utils import casting

from tsl.metrics.torch.metrics import MaskedMAE, MaskedMSE, MaskedMAPE, MaskedMRE
from tsl.metrics.numpy.functional import mae, mse, mape, mre

metrics_res = dict(mae=MaskedMAE(),
                   mse=MaskedMSE(),
                   mape=MaskedMAPE(),
                   mre=MaskedMRE(),
                   pinball=MaskedPinballLoss(q=0.75))

DELTA = 1e-6
x = 1. + torch.rand((2, 8, 2, 2), dtype=torch.float32)
y = 1. + torch.rand((2, 8, 2, 4), dtype=torch.float32)
mask = torch.bernoulli(0.5*torch.ones((2, 8, 2, 4), dtype=torch.float32))

predictor = Predictor(model_class=TCNModel,
                      model_kwargs={'input_size': 2, 'output_size': 4, 'horizon': 8},
                      optim_class=torch.optim.Adam,
                      optim_kwargs={'lr': 0.001},
                      loss_fn=MaskedMAE(compute_on_step=True),
                      scale_target=False,
                      metrics=metrics_res)

batch = Batch(input={'x': x}, target={'y': y}, mask=mask)
y_hat = predictor.predict_batch(batch, preprocess=False, postprocess=True)
y, mask = batch.y, batch.get('mask')
y_hat = y_hat.detach()

predictor.test_metrics.update(y_hat, y)
metrics_res = predictor.test_metrics.compute()
predictor.test_metrics.reset()
predictor.test_metrics.update(y_hat, y, mask)
masked_metrics_res = predictor.test_metrics.compute()
predictor.test_metrics.reset()

# @pytest.fixture(scope='module', autouse=False)
# def predictor_masked_metrics():
#     predictor = Predictor(model_class=TCNModel,
#                           model_kwargs={'input_size': 2, 'output_size': 4, 'horizon': 8},
#                           optim_class=torch.optim.Adam,
#                           optim_kwargs={'lr': 0.001},
#                           loss_fn=MaskedMAE(compute_on_step=True),
#                           scale_target=False,
#                           metrics=metrics_res)
#
#     batch = Batch(input={'x': x}, target={'y': y}, mask=mask)
#     out, y_hat = predictor.compute_metrics(batch)
#     out, y_hat = casting.numpy(out), casting.numpy(y_hat)
#     return out, y_hat


def test_mae():
    y_hat_, y_ = casting.numpy(y_hat), casting.numpy(y)
    res = mae(y_hat_, y_)
    assert(np.abs(metrics_res['test_mae'] - res) < DELTA)


def test_mae_masked():
    y_hat_, y_, mask_ = casting.numpy(y_hat), casting.numpy(y), casting.numpy(mask)
    res = mae(y_hat_, y_, mask_.astype(np.bool))
    assert(np.abs(masked_metrics_res['test_mae'] - res) < DELTA)


def test_mse():
    y_hat_, y_ = casting.numpy(y_hat), casting.numpy(y)
    res = mse(y_hat_, y_)
    assert(np.abs(metrics_res['test_mse'] - res) < DELTA)


def test_mse_masked():
    y_hat_, y_, mask_ = casting.numpy(y_hat), casting.numpy(y), casting.numpy(mask)
    res = mse(y_hat_, y_, mask_.astype(np.bool))
    assert(np.abs(masked_metrics_res['test_mse'] - res) < DELTA)


def test_mape():
    y_hat_, y_ = casting.numpy(y_hat), casting.numpy(y)
    res = mape(y_hat_, y_)
    assert(np.abs(metrics_res['test_mape'] - res) < DELTA)


def test_mape_masked():
    y_hat_, y_, mask_ = casting.numpy(y_hat), casting.numpy(y), casting.numpy(mask)
    res = mape(y_hat_, y_, mask_.astype(np.bool))
    assert(np.abs(masked_metrics_res['test_mape'] - res) < DELTA)


def test_mre():
    y_hat_, y_ = casting.numpy(y_hat), casting.numpy(y)
    res = mre(y_hat_, y_)
    assert(np.abs(metrics_res['test_mre'] - res) < DELTA)


def test_mre_masked():
    y_hat_, y_, mask_ = casting.numpy(y_hat), casting.numpy(y), casting.numpy(mask)
    res = mre(y_hat_, y_, mask_.astype(np.bool))
    assert(np.abs(masked_metrics_res['test_mre'] - res) < DELTA)


# if __name__ == '__main__':
#     out_masked = predictor_masked_metrics()

