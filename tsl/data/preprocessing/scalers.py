from copy import deepcopy
from typing import Tuple, List, Union, Optional

import numpy as np
import torch
from scipy import stats
from torch import Tensor
from torch.nn import Module
from torch_geometric.data.storage import recursive_apply

import tsl
from tsl.ops.pattern import check_pattern, take, outer_pattern
from tsl.typing import TensArray

__all__ = [
    'Scaler',
    'StandardScaler',
    'MinMaxScaler',
    'RobustScaler',
    'ScalerModule'
]


def zeros_to_one_(scale):
    """Set to 1 scales of near constant features, detected by identifying
    scales close to machine precision, in place.
    Adapted from :class:`sklearn.preprocessing._data._handle_zeros_in_scale`
    """
    if np.isscalar(scale):
        return 1.0 if np.isclose(scale, 0.) else scale
    eps = 10 * np.finfo(scale.dtype).eps
    zeros = np.isclose(scale, 0., atol=eps, rtol=eps)
    scale[zeros] = 1.0
    return scale


def fit_wrapper(fit_function):
    def fit(obj: "Scaler", x, *args, **kwargs) -> "Scaler":
        x_type = type(x)
        x = np.asarray(x)
        fit_function(obj, x, *args, **kwargs)
        if x_type is Tensor:
            obj.torch()
        return obj

    return fit


class Scaler:
    r"""Base class for linear :class:`~tsl.data.SpatioTemporalDataset` scalers.

    A :class:`~tsl.data.preprocessing.Scaler` is the base class for
    linear scaler objects. A linear scaler apply a linear transformation to the
    input using parameters `bias` :math:`\mu` and `scale` :math:`\sigma`:

    .. math::
      f(x) = (x - \mu) / \sigma.

    Args:
        bias (float): the offset of the linear transformation.
            (default: 0.)
        scale (float): the scale of the linear transformation.
            (default: 1.)
    """

    def __init__(self, bias=0., scale=1.):
        self.bias = bias
        self.scale = scale
        super(Scaler, self).__init__()

    def __repr__(self) -> str:
        sizes = []
        for k, v in self.params().items():
            param = f"{k}={tuple(v.shape) if hasattr(v, 'shape') else v}"
            sizes.append(param)
        return "{}({})".format(self.__class__.__name__, ', '.join(sizes))

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def params(self) -> dict:
        """Dictionary of the scaler parameters `bias` and `scale`.

        Returns:
            dict: Scaler's parameters `bias` and `scale.`
        """
        return dict(bias=self.bias, scale=self.scale)

    def torch(self, inplace=True):
        scaler = self
        if not inplace:
            scaler = deepcopy(self)
        for name, param in scaler.params().items():
            param = torch.atleast_1d(torch.as_tensor(param))
            setattr(scaler, name, param)
        return scaler

    def numpy(self, inplace=True):
        r"""Transform all tensors to numpy arrays."""
        scaler = self
        if not inplace:
            scaler = deepcopy(self)
        for name, param in scaler.params().items():
            if isinstance(param, Tensor):
                param = param.detach().cpu().numpy()
            setattr(scaler, name, param)
        return scaler

    @fit_wrapper
    def fit(self, x: TensArray, *args, **kwargs):
        """Fit scaler's parameters using input :obj:`x`."""
        raise NotImplementedError()

    def transform(self, x: TensArray):
        """Apply transformation :math:`f(x) = (x - \mu) / \sigma`."""
        return (x - self.bias) / self.scale + tsl.epsilon

    def inverse_transform(self, x: TensArray):
        """Apply inverse transformation
        :math:`f(x) = (x \cdot \sigma) + \mu`."""
        return x * (self.scale + tsl.epsilon) + self.bias

    def fit_transform(self, x: TensArray, *args, **kwargs):
        """Fit scaler's parameters using input :obj:`x` and then transform
        :obj:`x`."""
        self.fit(x, *args, **kwargs)
        return self.transform(x)


class StandardScaler(Scaler):
    """Apply standardization to data by removing mean and scaling to unit
    variance.

    Args:
        axis (int): dimensions of input to fit parameters on.
            (default: 0)
    """

    def __init__(self, axis: Union[int, Tuple] = 0):
        super(StandardScaler, self).__init__()
        self.axis = axis

    @fit_wrapper
    def fit(self, x: TensArray, mask=None, keepdims=True):
        """Fit scaler's parameters `bias` :math:`\mu` and `scale`
        :math:`\sigma` as the mean and the standard deviation of :obj:`x`,
        respectively.

        Args:
            x: array-like input
            mask (optional): boolean mask to denote elements of :obj:`x` on
                which to fit the parameters.
                (default: :obj:`None`)
            keepdims (bool): whether to keep the same dimensions as :obj:`x` in
                the parameters.
                (default: :obj:`True`)
        """
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmean(x.astype(np.float32), axis=self.axis,
                                   keepdims=keepdims).astype(x.dtype)
            self.scale = np.nanstd(x.astype(np.float32), axis=self.axis,
                                   keepdims=keepdims).astype(x.dtype)
        else:
            self.bias = x.mean(axis=self.axis, keepdims=keepdims)
            self.scale = x.std(axis=self.axis, keepdims=keepdims)
        self.scale = zeros_to_one_(self.scale)
        return self


class MinMaxScaler(Scaler):
    """Rescale data such that all lay in the specified range (default is
    :math:`[0,1]`).

    Args:
        axis (int): dimensions of input to fit parameters on.
            (default: 0)
        out_range (tuple): output range of transformed data.
            (default: :obj:`(0, 1)`)
    """

    def __init__(self, axis: Union[int, Tuple] = 0,
                 out_range: Tuple[float, float] = (0., 1.)):
        super(MinMaxScaler, self).__init__()
        self.axis = axis
        self.out_range = out_range

    @fit_wrapper
    def fit(self, x: TensArray, mask=None, keepdims=True):
        """Fit scaler's parameters `bias` :math:`\mu` and `scale`
        :math:`\sigma` as the mean and the standard deviation of :obj:`x`.

        Args:
            x: array-like input
            mask (optional): boolean mask to denote elements of :obj:`x` on
                which to fit the parameters.
                (default: :obj:`None`)
            keepdims (bool): whether to keep the same dimensions as :obj:`x` in
                the parameters.
                (default: :obj:`True`)
        """
        out_min, out_max = self.out_range
        if out_min >= out_max:
            raise ValueError(
                "Output range minimum must be smaller than maximum. Got {}."
                .format(self.out_range))

        if mask is not None:
            x = np.where(mask, x, np.nan)
            x_min = np.nanmin(x.astype(np.float32), axis=self.axis,
                              keepdims=keepdims).astype(x.dtype)
            x_max = np.nanmax(x.astype(np.float32), axis=self.axis,
                              keepdims=keepdims).astype(x.dtype)
        else:
            x_min = x.min(axis=self.axis, keepdims=keepdims)
            x_max = x.max(axis=self.axis, keepdims=keepdims)
        scale = (x_max - x_min) / (out_max - out_min)
        scale = zeros_to_one_(scale)
        bias = x_min - out_min * scale
        self.bias, self.scale = bias, scale
        return self


class RobustScaler(Scaler):
    r"""Removes the median and scales the data according to the quantile range.

    Default range is the Interquartile Range (IQR), i.e., the range between the
    1st quartile (25th quantile) and the 3rd quartile (75th quantile).

    Args:
        axis (int): dimensions of input to fit parameters on.
            (default: 0)
        quantile_range (tuple): quantile range :math:`(q_{\min}, q_{\max})`,
            with :math:`0.0 < q_{\min} < q_{\max} < 100.0`, used to calculate
            :obj:`scale`.
            (default: :obj:`(25.0, 75.0)`)
    """

    def __init__(self, axis: Union[int, Tuple] = 0,
                 quantile_range: Tuple[float, float] = (25.0, 75.0),
                 unit_variance: bool = False):
        super(RobustScaler, self).__init__()
        self.axis = axis
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance

    @fit_wrapper
    def fit(self, x: TensArray, mask=None, keepdims=True):
        r"""Fit scaler's parameters `bias` :math:`\mu` and `scale`
        :math:`\sigma` as the median and difference between quantiles of
        :obj:`x`, respectively.

        Args:
            x: array-like input
            mask (optional): boolean mask to denote elements of :obj:`x` on
                which to fit the parameters.
                (default: :obj:`None`)
            keepdims (bool): whether to keep the same dimensions as :obj:`x` in
                the parameters.
                (default: :obj:`True`)
        """
        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: {}"
                             .format(self.quantile_range))

        dtype = x.dtype
        if mask is not None:
            x = np.where(mask, x, np.nan).astype(np.float32)
            self.bias = np.nanmedian(x, axis=self.axis,
                                     keepdims=keepdims).astype(dtype)
            min_q, max_q = np.nanpercentile(x, self.quantile_range,
                                            axis=self.axis, keepdims=keepdims)
        else:
            self.bias = np.median(x, axis=self.axis, keepdims=keepdims)
            min_q, max_q = np.percentile(x, self.quantile_range,
                                         axis=self.axis, keepdims=keepdims)
        self.scale = (max_q - min_q).astype(dtype)
        self.scale = zeros_to_one_(self.scale)
        if self.unit_variance:
            adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(
                q_min / 100.0)
            self.scale = self.scale / adjust
        return self


class ScalerModule(Module):
    r"""Converts a :class:`Scaler` to a :class:`torch.nn.Module`, to insert
    transformation parameters and functions into the minibatch."""

    def __init__(self, scaler: Optional[Union[Scaler, "ScalerModule"]] = None,
                 *, bias: Union[Tensor, float] = 0.,
                 scale: Union[Tensor, float] = 1.,
                 pattern: Optional[str] = None):
        super(ScalerModule, self).__init__()
        self.training = False
        self.inherited_from = None
        self.pattern = check_pattern(pattern) if pattern is not None else None
        # initialize from scaler (if any)
        if isinstance(scaler, Scaler):
            scaler = scaler.torch()
            self.inherited_from = scaler.__class__
        elif isinstance(scaler, ScalerModule):
            self.inherited_from = scaler.inherited_from
            self.pattern = scaler.pattern
        if scaler is not None:
            bias = scaler.bias.clone().detach()
            scale = scaler.scale.clone().detach()
        # initialize from params
        else:
            bias = torch.atleast_1d(torch.as_tensor(bias))
            scale = torch.atleast_1d(torch.as_tensor(scale))
        # register scaling params as non-trainable parameters
        self.register_buffer('bias', bias)
        self.register_buffer('scale', scale)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def _get_name(self):
        if self.inherited_from is not None:
            return self.inherited_from.__name__ + 'Module'
        return self.__class__.__name__

    def extra_repr(self) -> str:
        s = ["bias={}".format(tuple(self.bias.shape)),
             "scale={}".format(tuple(self.scale.shape))]
        if self.pattern is not None:
            s.append("pattern='{}'".format(self.pattern))
        return ', '.join(s)

    def params(self) -> dict:
        """Dictionary of the scaler parameters `bias` and `scale`.

        Returns:
            dict: Scaler's parameters `bias` and `scale.`
        """
        return dict(bias=self.bias, scale=self.scale)

    @property
    def t(self) -> int or None:
        if self.pattern is not None and 't' in self.pattern:
            return max(self.scale.size(0), self.bias.size(0))

    def transform_tensor(self, x: Tensor) -> Tensor:
        r"""Apply transformation :math:`f(x) = (x - \mu) / \sigma`."""
        return (x - self.bias) / self.scale + tsl.epsilon

    def inverse_transform_tensor(self, x: Tensor) -> Tensor:
        r"""Apply inverse transformation
        :math:`f(x) = (x \cdot \sigma) + \mu`."""
        return x * (self.scale + tsl.epsilon) + self.bias

    def transform(self, x):
        return recursive_apply(x, self.transform_tensor)

    def inverse_transform(self, x):
        return recursive_apply(x, self.inverse_transform_tensor)

    def numpy(self):
        r"""Transform all tensors to numpy arrays, either for all attributes or
        only the ones given in :obj:`*args`."""
        b = self.bias.detach().cpu().numpy()
        s = self.scale.detach().cpu().numpy()
        return Scaler(bias=b, scale=s)

    def slice(self, time_index: Union[List, Tensor] = None,
              node_index: Union[List, Tensor] = None):
        if self.pattern is None:
            raise RuntimeError("You are trying to slice a scaler with no "
                               "pattern.")
        # move to new object
        scaler = ScalerModule(self)
        # if time-unvarying scaler, just apply unsqueezing indexing
        new_axes = None
        if self.t == 1 and time_index is not None:
            if time_index.ndim > 1:
                new_axes = torch.zeros([1] * time_index.ndim,
                                         dtype=torch.long)
                scaler.pattern = 'b ' * (time_index.ndim - 1) + scaler.pattern
        # shortcut for when scaler is time-unvarying and node_index is None
        if time_index is None and node_index is None:
            return scaler
        # slice params
        scaler.bias = take(scaler.bias, self.pattern,
                           time_index if self.bias.size(0) > 1 else new_axes,
                           node_index)
        scaler.scale = take(scaler.scale, self.pattern,
                            time_index if self.scale.size(0) > 1 else new_axes,
                            node_index)
        return scaler

    @staticmethod
    def cat_tensors(scalers, sizes, key, dim, fill_value):
        # arrange tensors in numbered dictionary where if tensors[i] exists then
        # the i-th scaler is not None and has a tensor at {scaler}.{key}
        tensors = {i: getattr(s, key) for i, s in enumerate(scalers)
                   if s is not None and getattr(s, key) is not None}
        # if no valid tensor return
        if len(tensors) == 0:
            return None
        # get dtype and device of first tensor and assume equal for all
        elem = next(iter(tensors.values()))
        dtype, device = elem.dtype, elem.device
        # for each scaler (also the ones with no tensor to be concatenated)
        # retrieve the tensor (or create one if not present) and the broadcast
        # shape
        out, shapes = [], []
        for i, scaler in enumerate(scalers):
            # retrieve tensor
            tensor = tensors.get(i)
            if tensor is None:  # i.e., if scaler is None or has key=None
                shape = [1] * len(sizes[i])
                tensor = torch.full(shape, fill_value,
                                    dtype=dtype, device=device)
            out.append(tensor)
            # compute broadcast shape
            shape = list(tensor.size())
            shape[dim] = sizes[i][dim]
            shapes.append(shape)
        # compute out shape as maximum shape in all dims but concat dim
        expand_dims = list(np.max(shapes, 0))
        # expand each tensor for output shape
        for i, shape in enumerate(shapes):
            expand_dims[dim] = shape[dim]
            out[i] = out[i].expand(*expand_dims)
        return torch.cat(out, dim=dim)

    @classmethod
    def cat(cls, scalers: Union[List, Tuple], dim: int = -1,
            sizes: Union[List, Tuple] = None):
        # if all scalers are None, return None
        if all([scaler is None for scaler in scalers]):
            return None
        # if there are at least one scaler and one 'None', sizes must be a list
        # containing the shape of the corresponding tensors
        if None in scalers:
            assert sizes is not None
        # scale
        scale = cls.cat_tensors(scalers, sizes, 'scale', dim, 1)
        # bias
        bias = cls.cat_tensors(scalers, sizes, 'bias', dim, 0)
        # pattern
        pattern = outer_pattern([scaler.pattern for scaler in scalers
                                 if scaler is not None and
                                 scaler.pattern is not None])
        return cls(bias=bias, scale=scale, pattern=pattern)
