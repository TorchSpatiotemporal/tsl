import inspect
from typing import Type, Mapping, Callable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import move_data_to_device
from torchmetrics import MetricCollection, Metric

from tsl.nn.metrics.metric_base import MaskedMetric


class Predictor(pl.LightningModule):
    """:class:`~pytorch_lightning.core.LightningModule` to implement predictors.

    Input data should follow the format [batch, steps, nodes, features].

    Args:
        model_class (type): Class of :obj:`~torch.nn.Module` implementing the
            predictor.
        model_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`model_class` at instantiation.
        optim_class (type): Class of :obj:`~torch.optim.Optimizer` implementing
            the optimizer to be used for training the model.
        optim_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`optim_class` at instantiation.
        loss_fn (callable): Loss function to be used for training the model.
        scale_target (bool): Whether to scale target before evaluating the loss.
            The metrics instead will always be evaluated in the original range.
            (default: :obj:`False`)
        metrics (mapping, optional): Set of metrics to be logged during
            train, val and test steps. The metric's name will be automatically
            prefixed with the loop in which the metric is computed (e.g., metric
            :obj:`mae` will be logged as :obj:`train_mae` when evaluated during
            training).
            (default: :obj:`None`)
        scheduler_class (type): Class of :obj:`~torch.optim.lr_scheduler._LRScheduler`
            implementing the learning rate scheduler to be used during training.
            (default: :obj:`None`)
        scheduler_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`scheduler_class` at instantiation.
            (default: :obj:`None`)
    """
    def __init__(self,
                 model_class: Type,
                 model_kwargs: Mapping,
                 optim_class: Type,
                 optim_kwargs: Mapping,
                 loss_fn: Callable,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super(Predictor, self).__init__()
        self.save_hyperparameters(logger=False)
        self.model_cls = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        if scheduler_kwargs is None:
            self.scheduler_kwargs = dict()
        else:
            self.scheduler_kwargs = scheduler_kwargs

        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn, on_step=True)
        else:
            self.loss_fn = None

        self.scale_target = scale_target

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)
        # instantiate model
        self.model = self.model_cls(**self.model_kwargs)

    def reset_model(self):
        self.model = self.model_cls(**self.model_kwargs)

    def load_model(self, filename: str):
        model = torch.load(filename, lambda storage, loc: storage)
        model_cls, model_kwargs = model['hyper_parameters']['model_class'], \
                                  model['hyper_parameters']['model_kwargs']
        assert model_cls == self.model_cls
        for k, v in model_kwargs.items():
            assert v == self.model_kwargs[k]
        self.load_state_dict(model['state_dict'])

    @property
    def trainable_parameters(self):
        return sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = dict()
            return MaskedMetric(metric, compute_on_step=on_step,
                                metric_kwargs=metric_kwargs)
        metric = metric.clone()
        metric.reset()
        return metric

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            {f'train_{k}': self._check_metric(m, on_step=True) for k, m in
             metrics.items()})
        self.val_metrics = MetricCollection(
            {f'val_{k}': self._check_metric(m) for k, m in metrics.items()})
        self.test_metrics = MetricCollection(
            {f'test_{k}': self._check_metric(m) for k, m in metrics.items()})

    def log_metrics(self, metrics, **kwargs):
        self.log_dict(metrics, on_step=False, on_epoch=True,
                      logger=True, prog_bar=True, **kwargs)

    def log_loss(self, name, loss, **kwargs):
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

    def predict_batch(self, batch, preprocess=False, postprocess=True,
                      return_target=False):
        """
        This method takes as an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Prediction should have a shape [batch, nodes, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...},
                                                              preprocessing:
                                                                {'bias': ..., 'scale': ..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that inputs are by default preprocessed before creating the batch)
        :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the trasformed signal)
        :param return_target: whether to return the prediction target y_true and the prediction mask
        :return: (y_true), y_hat, (mask)
        """
        x, y, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in x:
                    x[key] = trans.transform(x[key])
        y_hat = self.forward(**x)
        # Rescale outputs
        if postprocess:
            trans = transform.get('y')
            y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = y.get('y')
            return y, y_hat, mask
        return y_hat

    def predict_loader(self, loader, preprocess=False, postprocess=True,
                       return_mask=True):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_true, y_hat
        """
        targets, preds, masks = [], [], []
        for batch in loader:
            batch = move_data_to_device(batch, self.device)
            y, y_hat, mask = self.predict_batch(batch,
                                                preprocess=preprocess,
                                                postprocess=postprocess,
                                                return_target=True)
            targets.append(y)
            preds.append(y_hat)
            masks.append(mask)

        y = torch.cat(targets, 0)
        y_hat = torch.cat(preds, 0)
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        return y, y_hat

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    def on_predict_epoch_end(self, results):
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

        y = y_loss = batch.y
        mask = batch.mask

        # Compute predictions and compute loss
        y_hat = y_hat_loss = self.predict_batch(batch, preprocess=False,
                                                postprocess=not self.scale_target)
        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.train_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):

        y = y_loss = batch.y
        mask = batch.mask

        # Compute predictions
        y_hat = y_hat_loss = self.predict_batch(batch, preprocess=False,
                                                postprocess=not self.scale_target)
        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        val_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.val_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):

        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        y, mask = batch.y, batch.mask
        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    def configure_optimizers(self):
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
