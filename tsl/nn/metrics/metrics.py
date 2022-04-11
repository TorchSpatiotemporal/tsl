import tsl
from .metric_base import MaskedMetric
from torch.nn import functional as F
import torch

from torchmetrics.utilities.checks import _check_same_shape


def mape(y_hat, y):
    return torch.abs((y_hat - y) / y)


class MaskedMAE(MaskedMetric):
    """
        Mean Absolute Error Metric.

        Args:
            mask_nans (bool, optional): Whether to automatically mask nan values.
            mask_inf (bool, optional): Whether to automatically mask infinite values.
            compute_on_step (bool, optional): Whether to compute the metric right-away or if accumulate the results.
                             This should be `True` when using the metric to compute a loss function, `False` if the metric
                             is used for logging the aggregate error across different minibatches.
            at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMAE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)


class MaskedMAPE(MaskedMetric):
    """
        Mean Absolute Percentage Error Metric.

        Args:
            mask_nans (bool, optional): Whether to automatically mask nan values.
            compute_on_step (bool, optional): Whether to compute the metric right-away or if accumulate the results.
                             This should be `True` when using the metric to compute a loss function, `False` if the metric
                             is used for logging the aggregate error across different minibatches.
            at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    def __init__(self,
                 mask_nans=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):

        super(MaskedMAPE, self).__init__(metric_fn=mape,
                                         mask_nans=mask_nans,
                                         mask_inf=True,
                                         compute_on_step=compute_on_step,
                                         dist_sync_on_step=dist_sync_on_step,
                                         process_group=process_group,
                                         dist_sync_fn=dist_sync_fn,
                                         at=at)


class MaskedMSE(MaskedMetric):
    """
        Mean Squared Error Metric.

        Args:
            mask_nans (bool, optional): Whether to automatically mask nan values.
            mask_inf (bool, optional): Whether to automatically mask infinite values.
            compute_on_step (bool, optional): Whether to compute the metric right-away or if accumulate the results.
                             This should be `True` when using the metric to compute a loss function, `False` if the metric
                             is used for logging the aggregate error across different minibatches.
            at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMSE, self).__init__(metric_fn=F.mse_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)


class MaskedMRE(MaskedMetric):
    """
        Mean Relative Error Metric.

        Args:
            mask_nans (bool, optional): Whether to automatically mask nan values.
            mask_inf (bool, optional): Whether to automatically mask infinite values.
            compute_on_step (bool, optional): Whether to compute the metric right-away or if accumulate the results.
                             This should be `True` when using the metric to compute a loss function, `False` if the metric
                             is used for logging the aggregate error across different minibatches.
            at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedMRE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        compute_on_step=compute_on_step,
                                        dist_sync_on_step=dist_sync_on_step,
                                        process_group=process_group,
                                        dist_sync_fn=dist_sync_fn,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)
        self.add_state('tot', dist_reduce_fx='sum', default=torch.tensor(0., dtype=torch.float))

    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.zeros_like(val))
        y_masked = torch.where(mask, y, torch.zeros_like(y))
        return val.sum(), mask.sum(), y_masked.sum()

    def _compute_std(self, y_hat, y):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel(), y.sum()

    def compute(self):
        if self.tot > tsl.epsilon:
            return self.value / self.tot
        return self.value

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val, numel, tot = self._compute_masked(y_hat, y, mask)
        else:
            val, numel, tot = self._compute_std(y_hat, y)
        self.value += val
        self.numel += numel
        self.tot += tot


