import os
import shutil

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.engines import Imputer
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn.models import BiRNNImputerModel, GRINModel, RNNImputerModel
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy


def get_model_class(model_str):
    if model_str == 'rnni':
        model = RNNImputerModel
    elif model_str == 'birnni':
        model = BiRNNImputerModel
    elif model_str == 'grin':
        model = GRINModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_name: str, p_fault=0., p_noise=0.):
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    if dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    if dataset_name == 'la':
        return add_missing_values(MetrLA(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(),
                                  p_fault=p_fault,
                                  p_noise=p_noise,
                                  min_seq=12,
                                  max_seq=12 * 4,
                                  seed=56789)
    raise ValueError(f"Dataset {dataset_name} not available in this setting.")


def init_experiment():
    from datetime import datetime
    exp_id = 'imputation_' + datetime.now().strftime("%Y%m%d%H%M%S")
    # create temporary logging directory
    base_dir = os.path.dirname(__file__)
    log_dir = os.path.join(base_dir, exp_id)
    os.makedirs(log_dir, exist_ok=True)
    # load cfg with hydra
    path_to_yamls = os.path.join('.', 'config')
    with initialize(config_path=path_to_yamls,
                    job_name='test_example_imputation',
                    version_base=None):
        cfg = compose(config_name='test_imputation', overrides=[])
    return cfg, log_dir


@pytest.mark.slow
@pytest.mark.integration
def test_example_imputation():
    cfg, log_dir = init_experiment()
    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset.name,
                          p_fault=cfg.get('p_fault'),
                          p_noise=cfg.get('p_noise'))

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity)

    # instantiate dataset
    torch_dataset = ImputationDataset(target=dataset.dataframe(),
                                      mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      transform=MaskInput(),
                                      connectivity=adj,
                                      window=cfg.window,
                                      stride=cfg.stride)

    scalers = {'target': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size)
    dm.setup(stage='fit')

    ########################################
    # imputer                              #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    loss_fn = torch_metrics.MaskedMAE(compute_on_step=True)

    log_metrics = {
        'mae': torch_metrics.MaskedMAE(),
        'mape': torch_metrics.MaskedMAPE()
    }

    # setup imputer
    imputer = Imputer(model_class=model_cls,
                      model_kwargs=model_kwargs,
                      optim_class=getattr(torch.optim, cfg.optimizer.name),
                      optim_kwargs=dict(cfg.optimizer.hparams),
                      loss_fn=loss_fn,
                      metrics=log_metrics,
                      whiten_prob=cfg.whiten_prob,
                      prediction_loss_weight=cfg.prediction_loss_weight,
                      impute_only_missing=cfg.impute_only_missing,
                      warm_up_steps=cfg.warm_up_steps)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=cfg.patience,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        default_root_dir=log_dir,
        logger=None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        gradient_clip_val=cfg.grad_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
        limit_train_batches=1,
        log_every_n_steps=1,
    )

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    res_test = trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    res_functional = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
                          test_mre=numpy_metrics.mre(y_hat, y_true, mask),
                          test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    res_val = trainer.validate(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = imputer.collate_prediction_outputs(output)
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    res_functional.update(
        dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
             val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
             val_mape=numpy_metrics.mape(y_hat, y_true, mask)))

    # clean out directory
    shutil.rmtree(log_dir)

    assert np.isclose(res_test[0]['test_mae'], res_functional['test_mae'])
    assert np.isclose(res_test[0]['test_mape'], res_functional['test_mape'])
    assert np.isclose(res_val[0]['val_mae'], res_functional['val_mae'])
    assert np.isclose(res_val[0]['val_mape'], res_functional['val_mape'])
