import numpy as np
import torch

import tsl.metrics.numpy.functional as npf
import tsl.metrics.torch.functional as trf
from tsl.data import StaticBatch
from tsl.engines.predictor import Predictor
from tsl.metrics.torch import MaskedPinballLoss
from tsl.metrics.torch.metrics import (MaskedMAE, MaskedMAPE, MaskedMRE,
                                       MaskedMSE)
from tsl.nn.models.temporal.tcn_model import TCNModel
from tsl.ops.framearray import framearray_to_numpy
from tsl.utils.casting import torch_to_numpy

metrics_res = dict(mae=MaskedMAE(),
                   mse=MaskedMSE(),
                   mape=MaskedMAPE(),
                   mre=MaskedMRE(),
                   pinball=MaskedPinballLoss(q=0.75))

DELTA = 1e-6
x = 1. + torch.rand((2, 8, 2, 2), dtype=torch.float32)
y = 1. + torch.rand((2, 8, 2, 4), dtype=torch.float32)
mask = torch.bernoulli(0.5 * torch.ones((2, 8, 2, 4), dtype=torch.float32))

predictor = Predictor(model_class=TCNModel,
                      model_kwargs={
                          'input_size': 2,
                          'output_size': 4,
                          'horizon': 8
                      },
                      optim_class=torch.optim.Adam,
                      optim_kwargs={'lr': 0.001},
                      loss_fn=MaskedMAE(compute_on_step=True),
                      scale_target=False,
                      metrics=metrics_res)

batch = StaticBatch(input={'x': x}, target={'y': y}, mask=mask)
y_hat = predictor.predict_batch(batch, preprocess=False, postprocess=True)
y, mask = batch.y, batch.get('mask')
y_hat = y_hat.detach()

metrics_res = predictor.test_metrics(y_hat, y)
predictor.test_metrics.reset()
masked_metrics_res = predictor.test_metrics(y_hat, y, mask)
predictor.test_metrics.reset()


def test_mae_metric():
    y_hat_, y_ = torch_to_numpy(y_hat), torch_to_numpy(y)
    res = npf.mae(y_hat_, y_)
    assert np.isclose(metrics_res['test_mae'], res, atol=DELTA)


def test_mae_masked_metric():
    y_hat_, y_, mask_ = torch_to_numpy(y_hat), torch_to_numpy(
        y), torch_to_numpy(mask)
    res = npf.mae(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(masked_metrics_res['test_mae'], res, atol=DELTA)


def test_mse_metric():
    y_hat_, y_ = torch_to_numpy(y_hat), torch_to_numpy(y)
    res = npf.mse(y_hat_, y_)
    assert np.isclose(metrics_res['test_mse'], res, atol=DELTA)


def test_mse_masked_metric():
    y_hat_, y_, mask_ = torch_to_numpy(y_hat), torch_to_numpy(
        y), torch_to_numpy(mask)
    res = npf.mse(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(masked_metrics_res['test_mse'], res, atol=DELTA)


def test_mape_metric():
    y_hat_, y_ = torch_to_numpy(y_hat), torch_to_numpy(y)
    res = npf.mape(y_hat_, y_)
    assert np.isclose(metrics_res['test_mape'], res, atol=DELTA)


def test_mape_masked_metric():
    y_hat_, y_, mask_ = torch_to_numpy(y_hat), torch_to_numpy(
        y), torch_to_numpy(mask)
    res = npf.mape(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(masked_metrics_res['test_mape'], res, atol=DELTA)


def test_mre_metric():
    y_hat_, y_ = torch_to_numpy(y_hat), torch_to_numpy(y)
    res = npf.mre(y_hat_, y_)
    assert np.isclose(metrics_res['test_mre'], res, atol=DELTA)


def test_mre_masked_metric():
    y_hat_, y_, mask_ = torch_to_numpy(y_hat), torch_to_numpy(
        y), torch_to_numpy(mask)
    res = npf.mre(y_hat_, y_, mask_.astype(bool))
    assert np.isclose(masked_metrics_res['test_mre'], res, atol=DELTA)


def test_mae_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.mae(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mae(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mae_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.mae(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                     framearray_to_numpy(mask_))
    res_torch = trf.mae(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_mse_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.mse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mse(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mse_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.mse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                     framearray_to_numpy(mask_))
    res_torch = trf.mse(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_mape_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.mape(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mape(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mape_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.mape(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                      framearray_to_numpy(mask_))
    res_torch = trf.mape(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_mre_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.mre(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.mre(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_mre_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.mre(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                     framearray_to_numpy(mask_))
    res_torch = trf.mre(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_rmse_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.rmse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.rmse(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_rmse_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.rmse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                      framearray_to_numpy(mask_))
    res_torch = trf.rmse(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_nrmse_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.nrmse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.nrmse(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_nrmse_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.nrmse(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                       framearray_to_numpy(mask_))
    res_torch = trf.nrmse(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_nrmse2_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.nrmse_2(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.nrmse_2(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_nrmse2_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.nrmse_2(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                         framearray_to_numpy(mask_))
    res_torch = trf.nrmse_2(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_r2_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.r2(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.r2(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_r2_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.r2(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                    framearray_to_numpy(mask_))
    res_torch = trf.r2(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)


def test_nmae_functional():
    y_hat_, y_ = y_hat.clone(), y.clone()
    res_np = npf.nmae(framearray_to_numpy(y_hat_), framearray_to_numpy(y_))
    res_torch = trf.nmae(y_hat_, y_)
    assert np.isclose(res_np, res_torch)


def test_nmae_masked_functional():
    y_hat_, y_, mask_ = y_hat.clone(), y.clone(), mask.clone()
    res_np = npf.nmae(framearray_to_numpy(y_hat_), framearray_to_numpy(y_),
                      framearray_to_numpy(mask_))
    res_torch = trf.nmae(y_hat_, y_, mask_)
    assert np.isclose(res_np, res_torch)
