from typing import Callable, List, Mapping, Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch_geometric.data.storage import recursive_apply
from torchmetrics import Metric

from .predictor import Predictor


class Imputer(Predictor):
    r""":class:`~pytorch_lightning.core.LightningModule` to implement imputers.

    An imputer is an engines designed to fill out missing values in
    spatiotemporal data.

    Args:
        model (torch.nn.Module, optional): Model implementing the imputer.
            Ignored if argument `model_class` is not null. This argument should
            mainly be used for inference.
            (default: :obj:`None`)
        model_class (type, optional): Class of :obj:`~torch.nn.Module`
            implementing the imputer. If not `None`, argument `model` will be
            ignored.
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
        whiten_prob (float or list): Randomly mask out a valid datapoint during
            a training step with probability :obj:`whiten_prob`. If a list is
            passed, :obj:`whiten_prob` is sampled from the list for each batch.
            (default: :obj:`0.05`)
        prediction_loss_weight (float): The weight to assign to predictions
            (if any) in the loss. The loss is computed as

            .. math::

                L = \ell(\bar{y}, y, m) + \lambda \sum_i \ell(\hat{y}_i, y, m)

            where :math:`\ell(\bar{y}, y, m)` is the imputation loss,
            :math:`\ell(\bar{y}_i, y, m)` is the forecasting error of prediction
            :math:`\bar{y}_i`, and :math:`\lambda` is
            :obj:`prediction_loss_weight`.
            (default: :obj:`1.0`)
        impute_only_missing (bool): Whether to impute only missing values in
            inference or the whole sequence.
            (default: :obj:`True`)
        warm_up_steps (int, tuple): Number of steps to be considered as warm up
            stage at the beginning of the sequence. If a tuple is provided, the
            padding is applied both at the beginning and the end of the
            sequence.
            (default: :obj:`0`)
        metrics (mapping, optional): Set of metrics to be logged during
            train, val and test steps. The metric's name will be automatically
            prefixed with the loop in which the metric is computed (e.g., metric
            :obj:`mae` will be logged as :obj:`train_mae` when evaluated during
            training).
            (default: :obj:`None`)
        scheduler_class (type): Class of
            :obj:`~torch.optim.lr_scheduler._LRScheduler` implementing the
            learning rate scheduler to be used during training.
            (default: :obj:`None`)
        scheduler_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`scheduler_class` at instantiation.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        loss_fn: Optional[Callable] = None,
        scale_target: bool = False,
        metrics: Optional[Mapping[str, Metric]] = None,
        *,
        whiten_prob: Optional[Union[float, List[float]]] = 0.05,
        prediction_loss_weight: float = 1.0,
        impute_only_missing: bool = True,
        warm_up_steps: Union[int, Tuple[int, int]] = 0,
        model_class: Optional[Type] = None,
        model_kwargs: Optional[Mapping] = None,
        optim_class: Optional[Type] = None,
        optim_kwargs: Optional[Mapping] = None,
        scheduler_class: Optional = None,
        scheduler_kwargs: Optional[Mapping] = None,
    ):
        super(Imputer, self).__init__(model=model,
                                      model_class=model_class,
                                      model_kwargs=model_kwargs,
                                      optim_class=optim_class,
                                      optim_kwargs=optim_kwargs,
                                      loss_fn=loss_fn,
                                      scale_target=scale_target,
                                      metrics=metrics,
                                      scheduler_class=scheduler_class,
                                      scheduler_kwargs=scheduler_kwargs)

        if isinstance(whiten_prob, (list, tuple)):
            self.whiten_prob = torch.tensor(whiten_prob)
        else:
            self.whiten_prob = whiten_prob

        self.prediction_loss_weight = prediction_loss_weight
        self.impute_only_missing = impute_only_missing

        if isinstance(warm_up_steps, int):
            self.warm_up_steps = (warm_up_steps, 0)
        elif isinstance(warm_up_steps, (list, tuple)):
            self.warm_up_steps = tuple(warm_up_steps)
        if len(self.warm_up_steps) != 2:
            raise ValueError(
                "'warm_up_steps' must be an int of time steps to "
                "be cut at the beginning of the sequence or a "
                "pair of int if the sequence must be trimmed in a "
                "bidirectional way.")

    def trim_warm_up(self, *args):
        """Trim all tensors in :obj:`args` removing a number of first and last
        steps equals to :obj:`(self.warm_up_steps[0], self.warm_up_steps[1])`,
        respectively."""
        left, right = self.warm_up_steps
        # assume time in second dimension (after batch dim)
        trim = lambda s: s[:, left:s.size(1) - right]  # noqa
        args = recursive_apply(args, trim)
        if len(args) == 1:
            return args[0]
        return args

    # Imputation data hooks ###################################################

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        r"""For every training batch, randomly mask out value with probability
        :obj:`p = self.whiten_prob`. Then, whiten missing values in
        :obj:`batch.input.x`."""
        super(Imputer, self).on_train_batch_start(batch, batch_idx)
        if self.whiten_prob is not None:
            # randomly mask out value with probability p = whiten_prob
            mask = batch.mask
            batch.original_mask = mask
            p = self.whiten_prob
            if isinstance(p, Tensor) and p.ndim > 0:
                # broadcast p to mask size
                p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
                # sample p for each batch
                p = p[torch.randint(len(p), p_size)].to(device=mask.device)
            # set each non-zero element of mask to 0 with probability p
            whiten_mask = torch.rand(mask.size(), device=mask.device) > p
            batch.mask = mask & whiten_mask
            # whiten missing values
            if 'x' in batch.input:
                batch.input.x = batch.input.x * batch.mask

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Make predictions
        y_hat = self.predict(**batch.input)
        # Rescale outputs
        trans = batch.transform.get('y')
        if trans is not None:
            y_hat = trans.inverse_transform(y_hat)
        # fill missing values in target data
        if self.impute_only_missing:
            y_hat = torch.where(batch.mask.bool(), batch.y, y_hat)
        # return dict
        output = dict(**batch.target,
                      y_hat=y_hat,
                      mask=batch.mask,
                      eval_mask=batch.eval_mask)
        return output

    def shared_step(self, batch, mask):
        y = y_loss = batch.y
        y_hat = y_hat_loss = self.predict_batch(
            batch, preprocess=False, postprocess=not self.scale_target)

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        if isinstance(y_hat_loss, (list, tuple)):
            imputation, predictions = y_hat_loss
            y_hat = y_hat[0]
        else:
            imputation, predictions = y_hat_loss, []

        loss = self.loss_fn(imputation, y_loss, mask)
        for pred in predictions:
            pred_loss = self.loss_fn(pred, y_loss, mask)
            loss += self.prediction_loss_weight * pred_loss

        return y_hat.detach(), y, loss

    def training_step(self, batch, batch_idx):

        y_hat, y, loss = self.shared_step(batch, batch.original_mask)

        # Logging
        self.train_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):

        y_hat, y, val_loss = self.shared_step(batch, batch.mask)

        # Logging
        self.val_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Compute outputs and rescale
        y_hat = self.predict_step(batch, batch_idx)['y_hat']

        # reconstruction loss
        test_loss = self.loss_fn(y_hat, batch.y, batch.mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), batch.y, batch.eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss
