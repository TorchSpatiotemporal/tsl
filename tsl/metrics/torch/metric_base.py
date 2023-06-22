import inspect
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape

from tsl.typing import Slicer
from tsl.utils.python_utils import parse_slicing_string


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
    Multiple metric functions can be specified,
    in which case they will be averaged.
    Weights can be assigned to perform a
    weighted average of the different metrics.

    Args:
        metric_fn (Sequence[callable], callable):
            Base function to compute the metric
            point-wise, multiple functions can be passed as a sequence.
        mask_nans (bool): Whether to automatically mask nan values.
            (default: :obj:`False`)
        mask_inf (bool): Whether to automatically mask infinite
            values.
            (default: :obj:`False`)
        metric_fn_kwargs (Sequence[dict], dict, optional):
            Keyword arguments needed by :obj:`metric_fn`.
            Use a sequence of keyword arguments if different :obj:`metric_fn`
            require different arguments.
            (default: :obj:`None`)
        metric_fn_kwargs (Sequence[float], float, optional):
            Weight assigned to each :obj:`metric_fn`.
            Use a sequence if different :obj:`metric_fn`
            require different weights.
            (default: :obj:`None`)
        at (str, Sequence[Tuple[Slicer, ...] | str], tuple[Slicer, ...],
            Slicer, optional):
            Numpy style slicing to define specific parts
            of the output to compute the metrics on.
            Either one for all metric or a sequence for each metric.
            Slicing can either be a proper slicing tuple
            or a string representation containing just
            the part you would put inside square brackets
            to index an array/tensor.
            (default: :obj:`None`)
        full_state_update (bool, optional): Set this to overwrite the
            :obj:`full_state_update` value of the
            :obj:`torchmetrics.Metric` base class.
            (default: :obj:`None`)
    """

    is_differentiable: bool = None
    higher_is_better: bool = None
    full_state_update: bool = None

    def __init__(
        self,
        metric_fn: Union[Sequence[Callable], Callable],
        metric_fn_kwargs: Optional[Union[Sequence[Dict[str, Any]],
                                         Dict[str, Any]]] = None,
        mask_nans: bool = False,
        mask_inf: bool = False,
        at: Union[str, Sequence[Union[Tuple[Slicer, ...], str]],
                  tuple[Slicer, ...], Slicer] = ...,
        weights: Optional[Sequence[float]] = None,
        full_state_update: Optional[bool] = None,
        **kwargs: Any,
    ):
        super().__init__(
            metric_fn=None,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            metric_fn_kwargs=None,
            at=None,
            full_state_update=full_state_update,
            **kwargs,
        )
        assert (
            len({
                len(e)
                for e in (metric_fn, metric_fn_kwargs, at, weights)
                if isinstance(e, Sequence)
            }) == 1
        ), "All sequences used as masked metric arguments " \
           "must have the same length."
        if metric_fn_kwargs is None:
            metric_fn_kwargs = {}
        if isinstance(metric_fn, Sequence) and isinstance(
                metric_fn_kwargs, Sequence):
            self.metric_fn = tuple(
                partial(fn, **fn_kwargs)
                for fn, fn_kwargs in zip(metric_fn, metric_fn_kwargs))
        elif isinstance(metric_fn, Sequence):
            self.metric_fn = tuple(
                partial(fn, **metric_fn_kwargs) for fn in metric_fn)
        else:
            self.metric_fn = (partial(metric_fn, **metric_fn_kwargs), )
        if isinstance(at, str) or not isinstance(at, Sequence):
            at = (at, )
        at = list(
            parse_slicing_string(e) if isinstance(e, str) else e for e in at)
        self.at = at * len(self.metric_fn) if len(at) == 1 else at
        if weights is None:
            self.weights = (1.0, ) * len(self.metric_fn)
        else:
            self.weights = weights
        self.mask_nans = mask_nans
        self.mask_inf = mask_inf
        self.add_state("value",
                       dist_reduce_fx="sum",
                       default=torch.tensor(0.0, dtype=torch.float))
        self.add_state("numel",
                       dist_reduce_fx="sum",
                       default=torch.tensor(0.0, dtype=torch.float))

    def _check_mask(self, mask, val, at=...):
        if mask is None:
            mask = torch.ones_like(val, dtype=torch.bool)
        else:
            mask = mask[at].bool()
            _check_same_shape(mask, val)
        if self.mask_nans:
            mask = mask & ~torch.isnan(val)
        if self.mask_inf:
            mask = mask & ~torch.isinf(val)
        return mask

    def is_masked(self, mask):
        return self.mask_inf or self.mask_nans or (mask is not None)

    def update(self, y_hat, y, mask=None):
        _check_same_shape(y_hat, y)
        for i in range(len(self.metric_fn)):
            val = self.metric_fn[i](y_hat[self.at[i]], y[self.at[i]])
            if self.is_masked(mask):
                mask = self._check_mask(mask, val, self.at[i])
                val[~mask] = 0
                numel = mask.sum()
            else:
                numel = val.numel()
            self.value += val.sum() * self.weights[i]
            self.numel += numel

    def compute(self):
        if self.numel > 0:
            return self.value / self.numel
        return self.value
