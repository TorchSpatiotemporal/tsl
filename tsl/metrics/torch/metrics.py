from typing import Any, Optional

import torch
from torch.nn import functional as F
from torchmetrics.utilities.checks import _check_same_shape

import tsl

from .functional import mape
from .metric_base import MaskedMetric


class MaskedMAE(MaskedMetric):
    """Mean Absolute Error Metric.

    Args:
        mask_nans (bool): Whether to automatically mask nan values.
            (default: :obj:`False`)
        mask_inf (bool): Whether to automatically mask infinite
            values.
            (default: :obj:`False`)
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
            (default: :obj:`None`)
        dim (int): The index of the dimension that represents time in a batch.
            Relevant only when also 'at' is defined.
            Default assumes [b t n f] format.
            (default: :obj:`1`)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 mask_nans: bool = False,
                 mask_inf: bool = False,
                 at: Optional[int] = None,
                 dim: int = 1,
                 **kwargs: Any):
        super(MaskedMAE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        dim=dim,
                                        **kwargs)


class MaskedMAPE(MaskedMetric):
    """Mean Absolute Percentage Error Metric.

    Args:
        mask_nans (bool): Whether to automatically mask nan values.
            (default: :obj:`False`)
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
            (default: :obj:`None`)
        dim (int): The index of the dimension that represents time in a batch.
            Relevant only when also 'at' is defined.
            Default assumes [b t n f] format.
            (default: :obj:`1`)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 mask_nans: bool = False,
                 at: Optional[int] = None,
                 dim: int = 1,
                 **kwargs: Any):
        super(MaskedMAPE,
              self).__init__(metric_fn=mape,
                             mask_nans=mask_nans,
                             mask_inf=True,
                             metric_fn_kwargs={'reduction': 'none'},
                             at=at,
                             dim=dim,
                             **kwargs)


class MaskedMSE(MaskedMetric):
    """Mean Squared Error Metric.

    Args:
        mask_nans (bool): Whether to automatically mask nan values.
            (default: :obj:`False`)
        mask_inf (bool): Whether to automatically mask infinite
            values.
            (default: :obj:`False`)
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
            (default: :obj:`None`)
        dim (int): The index of the dimension that represents time in a batch.
            Relevant only when also 'at' is defined.
            Default assumes [b t n f] format.
            (default: :obj:`1`)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 mask_nans: bool = False,
                 mask_inf: bool = False,
                 at: Optional[int] = None,
                 dim: int = 1,
                 **kwargs: Any):
        super(MaskedMSE, self).__init__(metric_fn=F.mse_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        dim=dim,
                                        **kwargs)


class MaskedMRE(MaskedMetric):
    """Mean Relative Error Metric.

    Args:
        mask_nans (bool): Whether to automatically mask nan values.
            (default: :obj:`False`)
        mask_inf (bool): Whether to automatically mask infinite
            values.
            (default: :obj:`False`)
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
            (default: :obj:`None`)
        dim (int): The index of the dimension that represents time in a batch.
            Relevant only when also 'at' is defined.
            Default assumes [b t n f] format.
            (default: :obj:`1`)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 mask_nans: bool = False,
                 mask_inf: bool = False,
                 at: Optional[int] = None,
                 dim: int = 1,
                 **kwargs: Any):
        super(MaskedMRE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_fn_kwargs={'reduction': 'none'},
                                        at=at,
                                        dim=dim,
                                        **kwargs)
        self.add_state('tot',
                       dist_reduce_fx='sum',
                       default=torch.tensor(0., dtype=torch.float))

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
        if self.at is not None:
            y_hat = y_hat.select(self.dim, self.at)
            y = y.select(self.dim, self.at)
            if mask is not None:
                mask = mask.select(self.dim, self.at)
        if self.is_masked(mask):
            val, numel, tot = self._compute_masked(y_hat, y, mask)
        else:
            val, numel, tot = self._compute_std(y_hat, y)
        self.value += val
        self.numel += numel
        self.tot += tot
