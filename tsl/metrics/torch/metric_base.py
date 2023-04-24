import inspect
from copy import deepcopy
from functools import partial
from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def convert_to_masked_metric(metric_fn, **kwargs):
    """
    Simple utility function to transform a callable into a `MaskedMetric`.

    Args:
        metric_fn: Callable to be wrapped.
        **kwargs: Keyword arguments that will be passed to the callable.

    Returns:

    """
    if not isinstance(metric_fn, MaskedMetric):
        if 'reduction' in inspect.getfullargspec(metric_fn).args:
            metric_kwargs = {'reduction': 'none'}
        else:
            metric_kwargs = dict()
        return MaskedMetric(metric_fn,
                            metric_fn_kwargs=metric_kwargs,
                            **kwargs)
    assert not len(kwargs)
    return deepcopy(metric_fn)


class MaskedMetric(Metric):
    r"""Base class to implement the metrics used in `tsl`.

    In particular a `MaskedMetric` accounts for missing values in the input
    sequences by accepting a boolean mask as additional input.

    Args:
        metric_fn: Base function to compute the metric point-wise.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
        t_dim (int): The index of the dimension that represents time in a batch.
            Default assumes [b t n f] format, hence is 1.
    """

    is_differentiable: bool = None
    higher_is_better: bool = None
    full_state_update: bool = None

    def __init__(self,
                 metric_fn,
                 mask_nans=False,
                 mask_inf=False,
                 metric_fn_kwargs=None,
                 at=None,
                 full_state_update: bool = None,
                 t_dim: int = 1,
                 **kwargs: Any):
        # set 'full_state_update' before Metric instantiation
        if full_state_update is not None:
            self.__dict__['full_state_update'] = full_state_update
        super(MaskedMetric, self).__init__(**kwargs)

        if metric_fn_kwargs is None:
            metric_fn_kwargs = dict()

        self.metric_fn = partial(metric_fn, **metric_fn_kwargs)

        self.mask_nans = mask_nans
        self.mask_inf = mask_inf
        if at is None:
            self.at = slice(None)
        else:
            self.at = slice(at, at + 1)
        self.t_dim = t_dim
        self.add_state('value',
                       dist_reduce_fx='sum',
                       default=torch.tensor(0., dtype=torch.float))
        self.add_state('numel',
                       dist_reduce_fx='sum',
                       default=torch.tensor(0., dtype=torch.float))

    def _check_mask(self, mask, val):
        if mask is None:
            mask = torch.ones_like(val, dtype=torch.bool)
        else:
            mask = mask.bool()
            _check_same_shape(mask, val)
        if self.mask_nans:
            mask = mask & ~torch.isnan(val)
        if self.mask_inf:
            mask = mask & ~torch.isinf(val)
        return mask

    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.zeros_like(val))
        return val.sum(), mask.sum()

    def _compute_std(self, y_hat, y):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel()

    def is_masked(self, mask):
        return self.mask_inf or self.mask_nans or (mask is not None)

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat.select(self.t_dim, self.at)
        y = y.select(self.t_dim, self.at)
        if mask is not None:
            mask = mask.select(self.t_dim, self.at)
        if self.is_masked(mask):
            val, numel = self._compute_masked(y_hat, y, mask)
        else:
            val, numel = self._compute_std(y_hat, y)
        self.value += val
        self.numel += numel

    def compute(self):
        if self.numel > 0:
            return self.value / self.numel
        return self.value
