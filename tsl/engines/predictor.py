import inspect
from typing import Type, Mapping, Callable, Optional

import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection, Metric

from tsl import logger
from tsl.data import Data
from tsl.metrics.torch import MaskedMetric
from tsl.nn.models import BaseModel
from tsl.utils import foo_signature


class Predictor(pl.LightningModule):
    """:class:`~pytorch_lightning.core.LightningModule` to implement predictors.

    Input data should follow the format [batch, steps, nodes, features].

    Args:
        model (torch.nn.Module, optional): Model implementing the predictor.
            Ignored if argument `model_class` is not :obj:`None`. This argument
            should mainly be used for inference.
            (default: :obj:`None`)
        model_class (type, optional): Class of :obj:`~torch.nn.Module`
            implementing the predictor. If not :obj:`None`, argument ``model``
            will be ignored.
            (default: :obj:`None`)
        model_kwargs (mapping, optional): Dictionary of arguments to be
            forwarded to :obj:`model_class` at instantiation.
            (default: :obj:`None`)
        optim_class (type, optional): Class of :obj:`~torch.optim.Optimizer`
            implementing the optimizer to be used for training the model.
            (default: :obj:`None`)
        optim_kwargs (mapping, optional): Dictionary of arguments to be
            forwarded to :obj:`optim_class` at instantiation.
            (default: :obj:`None`)
        loss_fn (callable, optional): Loss function to be used for training the
            model.
            (default: :obj:`None`)
        scale_target (bool): Whether to scale target before evaluating the loss.
            The metrics instead will always be evaluated in the original range.
            (default: :obj:`False`)
        metrics (mapping, optional): Set of metrics to be logged during
            train, val and test steps. The metric's name will be automatically
            prefixed with the loop in which the metric is computed (e.g., metric
            :obj:`mae` will be logged as :obj:`train_mae` when evaluated during
            training).
            (default: :obj:`None`)
        scheduler_class (type, optional): Class of
            :obj:`~torch.optim.lr_scheduler._LRScheduler` implementing the
            learning rate scheduler to be used during training.
            (default: :obj:`None`)
        scheduler_kwargs (mapping, optional): Dictionary of arguments to be
            forwarded to :obj:`scheduler_class` at instantiation.
            (default: :obj:`None`)
    """

    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super(Predictor, self).__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'model'],
                                  logger=False)
        self.model_cls = model_class
        self.model_kwargs = model_kwargs or dict()
        self._model_fwd_signature = None  # automatic set on model assignment

        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()

        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn, on_step=True)
        else:
            self.loss_fn = None

        self.scale_target = scale_target

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)

        if self.model_cls is not None:
            # instantiate model
            self.model = self.model_cls(**self.model_kwargs)
        else:
            self.model = model

    def __setattr__(self, key, value):
        super(Predictor, self).__setattr__(key, value)
        if key == 'model' and value is not None:
            self._model_fwd_signature = foo_signature(self.model.forward)
            self._check_kwargs = True

    def reset_model(self):
        """"""
        if self.model_cls is not None:
            self.model = self.model_cls(**self.model_kwargs)
        else:
            self.model = None

    def load_model(self, filename: str):
        """Load model's weights from checkpoint at :attr:`filename`.

        Differently from
        :meth:`~pytorch_lightning.core.LightningModule.load_from_checkpoint`,
        this method allows to load the state_dict also for models instantiated
        outside the predictor, without checking that hyperparameters of the
        checkpoint's model are the same of the predictor's model.
        """
        storage = torch.load(filename, lambda storage, loc: storage)
        # if predictor.model has been instantiated inside predictor
        if self.model_cls is not None:
            model_cls = storage['hyper_parameters']['model_class']
            model_kwargs = storage['hyper_parameters']['model_kwargs']
            # check model class and hyperparameters are the same
            assert model_cls == self.model_cls
            if model_kwargs is not None:
                for k, v in model_kwargs.items():
                    assert v == self.model_kwargs[k]
        else:
            logger.warning("Predictor with already instantiated model is loading "
                           f"a state_dict from {filename}. Cannot check if model "
                           "hyperparameters are the same.")
        self.load_state_dict(storage['state_dict'])

    @property
    def is_tsl_model(self):
        """"""
        return self.model is not None and isinstance(self.model, BaseModel)

    @property
    def trainable_parameters(self) -> int:
        """"""
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def filter_forward_kwargs(self) -> bool:
        """"""
        return self._model_fwd_signature is not None and \
               not self._model_fwd_signature['has_kwargs']

    def _filter_forward_kwargs(self, kwargs: dict) -> dict:
        """"""
        if self._check_kwargs:
            model_args = self._model_fwd_signature['signature']
            filtered = set(kwargs).difference(model_args)
            forwarded = set(kwargs).intersection(model_args)
            msg = f"Only args {list(forwarded)} are forwarded to the model " \
                  f"({self.model.__class__.__name__}). "
            if len(filtered):
                msg = f"Arguments {list(filtered)} are filtered out. " + msg
            logger.warn(msg)
            self._check_kwargs = False
        return {k: v for k, v in kwargs.items() if k in
                self._model_fwd_signature['signature']}

    def forward(self, *args, **kwargs):
        """"""
        if self.filter_forward_kwargs:
            kwargs = self._filter_forward_kwargs(kwargs)
        return self.model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """"""
        predict_fn = self.model.predict if self.is_tsl_model else self.model
        if self.filter_forward_kwargs:
            kwargs = self._filter_forward_kwargs(kwargs)
        return predict_fn(*args, **kwargs)

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = dict()
            return MaskedMetric(metric, compute_on_step=on_step,
                                metric_fn_kwargs=metric_kwargs)
        metric = metric.clone()
        metric.reset()
        return metric

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix='train_')
        self.val_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix='val_')
        self.test_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix='test_')

    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(metrics, on_step=False, on_epoch=True,
                      logger=True, prog_bar=True, **kwargs)

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(name + '_loss', loss.detach(), on_step=False, on_epoch=True,
                 logger=True, prog_bar=False, **kwargs)

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.

        :param batch: the batch
        :return: batch_data, batch_preprocessing
        """
        inputs, targets = batch.input, batch.target
        mask = batch.get('mask')
        transform = batch.get('transform')
        return inputs, targets, mask, transform

    def predict_batch(self, batch: Data,
                      preprocess: bool = False, postprocess: bool = True,
                      return_target: bool = False,
                      **forward_kwargs):
        """This method takes as input a :class:`~tsl.data.Data` object and
        outputs the predictions.

        Note that this method works seamlessly for all :class:`~tsl.data.Data`
        subclasses like :class:`~tsl.data.StaticBatch` and
        :class:`~tsl.data.DisjointBatch`.

        Args:
            batch (Data): The batch to be forwarded to the model.
            preprocess (bool, optional): If :obj:`True`, then preprocess tensors
                in :attr:`batch.input` using transformation modules in
                :attr:`batch.transform`. Note that inputs are preprocessed
                before creating the batch by default.
                (default: :obj:`False`)
            postprocess (bool, optional): If :obj:`True`, then postprocess the
                model output using transformation modules for
                :attr:`batch.target` in :attr:`batch.transform`.
                (default: :obj:`True`)
            return_target (bool, optional): If :obj:`True`, then returns also
                the prediction target :attr:`batch.target` and the prediction
                mask :attr:`batch.mask`, besides the model output. In this case,
                the order of the arguments in the return is
                :attr:`batch.target`, :obj:`y_hat`, :attr:`batch.mask`.
                (default: :obj:`False`)
            **forward_kwargs: additional keyword arguments passed to the forward
                method.
        """
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        y_hat = self.forward(**inputs, **forward_kwargs)
        # Rescale outputs
        if postprocess:
            trans = transform.get('y')
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = targets.get('y')
            return y, y_hat, mask
        return y_hat

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """"""
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    def on_predict_epoch_end(self, results):
        """"""
        # iterate over results of each dataloader
        processed_results = []
        for res in results:
            processed_res = dict()
            # iterate over outputs for each batch
            for b_res in res:
                for k, v in b_res.items():
                    try:
                        processed_res[k].append(v)
                    except KeyError:
                        processed_res[k] = [v]
            processed_results.append(processed_res)
        results[:] = processed_results
        # concatenate results
        for res in results:
            for k, v in res.items():
                res[k] = torch.cat(v, 0)

    def training_step(self, batch, batch_idx):
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions and compute loss
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                        postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """"""
        y = y_loss = batch.y
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                        postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        val_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        """"""
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        y, mask = batch.y, batch.get('mask')
        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    def compute_metrics(self, batch, preprocess=False, postprocess=True):
        """"""
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess, postprocess)
        y, mask = batch.y, batch.get('mask')
        self.test_metrics.update(y_hat.detach(), y, mask)
        metrics_dict = self.test_metrics.compute()
        self.test_metrics.reset()
        return metrics_dict, y_hat

    def configure_optimizers(self):
        """"""
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg['optimizer'] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg['lr_scheduler'] = scheduler
            if metric is not None:
                cfg['monitor'] = metric
        return cfg
