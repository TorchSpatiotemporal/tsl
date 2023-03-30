import os

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tsl import __path__ as tsl_path
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.engines import Imputer
from tsl.metrics import numpy as numpy_metrics
from tsl.metrics import torch as torch_metrics
from tsl.nn.models import BiRNNImputerModel, GRINModel, RNNImputerModel
from tsl.ops.imputation import add_missing_values
from tsl.transforms import MaskInput
from tsl.utils import remove_files
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


# force test folder to be cwd
test_path = os.path.abspath(os.path.join(tsl_path[0], '..', 'tests'))
os.chdir(test_path)

# load cfg with hydra
path_to_yamls = os.path.join('..', 'examples', 'imputation', 'config')
with initialize(config_path=path_to_yamls,
                job_name='test_example_imputation',
                version_base=None):
    cfg = compose(config_name='test', overrides=[])


@pytest.mark.slow
@pytest.mark.integration
def test_example_imputation():
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
        batch_size=cfg.batch_size,
        workers=cfg.workers)
    dm.setup(stage='fit')

    if cfg.get('in_sample', False):
        dm.trainset = list(range(len(torch_dataset)))

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

    if cfg.lr_scheduler is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup imputer
    imputer = Imputer(model_class=model_cls,
                      model_kwargs=model_kwargs,
                      optim_class=getattr(torch.optim, cfg.optimizer.name),
                      optim_kwargs=dict(cfg.optimizer.hparams),
                      loss_fn=loss_fn,
                      metrics=log_metrics,
                      scheduler_class=scheduler_class,
                      scheduler_kwargs=scheduler_kwargs,
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
        dirpath=os.getcwd(),
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      default_root_dir=os.getcwd(),
                      logger=None,
                      gpus=1 if torch.cuda.is_available() else None,
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(imputer, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)

    imputer.freeze()
    res_test = trainer.test(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader())
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    res_functional = dict(test_mae=numpy_metrics.mae(y_hat, y_true, mask),
                          test_mre=numpy_metrics.mre(y_hat, y_true, mask),
                          test_mape=numpy_metrics.mape(y_hat, y_true, mask))

    res_val = trainer.validate(imputer, datamodule=dm)

    output = trainer.predict(imputer, dataloaders=dm.val_dataloader())
    output = torch_to_numpy(output)
    y_hat, y_true, mask = (output['y_hat'], output['y'],
                           output.get('eval_mask', None))
    res_functional.update(
        dict(val_mae=numpy_metrics.mae(y_hat, y_true, mask),
             val_rmse=numpy_metrics.rmse(y_hat, y_true, mask),
             val_mape=numpy_metrics.mape(y_hat, y_true, mask)))

    # clean out directory
    remove_files(os.getcwd(), extension='.ckpt')

    assert np.isclose(res_test[0]['test_mae'], res_functional['test_mae'])
    assert np.isclose(res_test[0]['test_mape'], res_functional['test_mape'])
    assert np.isclose(res_val[0]['val_mae'], res_functional['val_mae'])
    assert np.isclose(res_val[0]['val_mape'], res_functional['val_mape'])
