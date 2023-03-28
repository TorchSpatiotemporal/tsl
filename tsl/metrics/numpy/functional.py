from typing import Literal, Optional, Tuple, Union

import numpy as np

import tsl
from tsl.ops.framearray import framearray_to_numpy
from tsl.typing import FrameArray

ReductionType = Literal['mean', 'sum', 'none']
MetricOutputType = Union[float, np.ndarray]

__all__ = [
    'mae', 'nmae', 'mape', 'mse', 'rmse', 'nrmse', 'nrmse_2', 'r2', 'mre'
]


def _masked_reduce(x: FrameArray,
                   reduction: ReductionType,
                   mask: Optional[FrameArray] = None,
                   nan_to_zero: bool = False) -> MetricOutputType:
    x = framearray_to_numpy(x)  # covert x to ndarray if not already (no copy)
    # 'none': return x with x[i] = 0/nan where mask[i] == False
    if reduction == 'none':
        if mask is not None:
            masked_idxs = np.logical_not(framearray_to_numpy(mask))
            x[masked_idxs] = 0 if nan_to_zero else np.nan
        return x
    # 'mean'/'sum': return mean/sum of x[mask == True]
    if mask is not None:
        mask = framearray_to_numpy(mask).astype(bool)
        x = x[mask]
    if reduction == 'mean':
        return np.mean(x)
    elif reduction == 'sum':
        return np.sum(x)
    else:
        raise ValueError(f"reduction {reduction} not allowed, must be one of "
                         "['mean', 'sum', 'none'].")


def mae(y_hat: FrameArray,
        y: FrameArray,
        mask: Optional[FrameArray] = None,
        reduction: ReductionType = 'mean',
        nan_to_zero: bool = False) -> MetricOutputType:
    r"""Compute the `Mean Absolute Error (MAE)
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ between the estimate
    :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MAE} = \frac{\sum_{i=1}^n |\hat{y}_i - y_i|}{n}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
            If :attr:`mask` is not :obj:`None` and :attr:`reduction` is
            :obj:`'none'`, masked indices are set to :obj:`nan` (see
            :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | np.ndarray: The Mean Absolute Error.
    """
    err = np.abs(y_hat - y)
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def nmae(y_hat: FrameArray,
         y: FrameArray,
         mask: Optional[FrameArray] = None,
         reduction: ReductionType = 'mean',
         nan_to_zero: bool = False) -> MetricOutputType:
    r"""Compute the *Normalized Mean Absolute Error* (NMAE) between the estimate
    :math:`\hat{y}` and the true value :math:`y`. The NMAE is the `Mean Absolute
    Error (MAE) <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ scaled by
    the max-min range of the target data, i.e.

    .. math::

        \text{NMAE} = \frac{\frac{1}{N} \sum_{i=1}^n |\hat{y}_i - y_i|}
        {\max(y) - \min(y)}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
            If :attr:`mask` is not :obj:`None` and :attr:`reduction` is
            :obj:`'none'`, masked indices are set to :obj:`nan` (see
            :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | np.ndarray: The Normalized Mean Absolute Error
    """
    delta = np.max(y) - np.min(y) + tsl.epsilon
    err = np.abs(y_hat - y) / delta
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def mape(y_hat: FrameArray,
         y: FrameArray,
         mask: Optional[FrameArray] = None,
         reduction: ReductionType = 'mean',
         nan_to_zero: bool = False) -> MetricOutputType:
    r"""Compute the `Mean Absolute Percentage Error (MAPE).
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MAPE} = \frac{1}{n} \sum_{i=1}^n \frac{|\hat{y}_i - y_i|}
        {y_i}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
            If :attr:`mask` is not :obj:`None` and :attr:`reduction` is
            :obj:`'none'`, masked indices are set to :obj:`nan` (see
            :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | np.ndarray: The Mean Absolute Percentage Error.
    """
    err = np.abs((y_hat - y) / (y + tsl.epsilon))
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def mse(y_hat: FrameArray,
        y: FrameArray,
        mask: Optional[FrameArray] = None,
        reduction: ReductionType = 'mean',
        nan_to_zero: bool = False) -> MetricOutputType:
    r"""Compute the `Mean Squared Error (MSE)
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MSE} = \frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{n}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
            If :attr:`mask` is not :obj:`None` and :attr:`reduction` is
            :obj:`'none'`, masked indices are set to :obj:`nan` (see
            :attr:`nan_to_zero`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | np.ndarray: The Mean Squared Error.
    """
    err = np.square(y_hat - y)
    return _masked_reduce(err, reduction, mask, nan_to_zero)


def rmse(y_hat: FrameArray,
         y: FrameArray,
         mask: Optional[FrameArray] = None,
         reduction: ReductionType = 'mean') -> MetricOutputType:
    r"""Compute the `Root Mean Squared Error (RMSE)
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{RMSE} = \sqrt{\frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{n}}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. ``'mean'``: the sum of the output will be
            divided by the number of elements in the output, ``'sum'``: the
            output will be summed.
            (default: ``'mean'``)

    Returns:
        float: The Root Mean Squared Error.
    """
    err = np.square(y_hat - y)
    return np.sqrt(_masked_reduce(err, reduction, mask))


def nrmse(y_hat: FrameArray,
          y: FrameArray,
          mask: Optional[FrameArray] = None,
          reduction: ReductionType = 'mean') -> MetricOutputType:
    r"""Compute the `Normalized Root Mean Squared Error (NRMSE)
        <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ between the
        estimate :math:`\hat{y}` and the true value :math:`y`, i.e.
        Normalization is by the max-min range of the data

        .. math::

            \text{NRMSE} = \frac{\sqrt{\frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}
            {n}} }{\max y - \min y}

        Args:
            y_hat (FrameArray): The estimated variable.
            y (FrameArray): The ground-truth variable.
            mask (FrameArray, optional): If provided, compute the metric using
                only the values at valid indices (with :attr:`mask` set to
                :obj:`True`).
            reduction (str): Specifies the reduction to apply to the output:
                ``'mean'`` | ``'sum'``. ``'mean'``: the sum of the output will
                be divided by the number of elements in the output, ``'sum'``:
                the output will be summed.
                (default: ``'mean'``)

        Returns:
            float: The range-normalzized NRMSE
        """
    delta = np.max(y) - np.min(y) + tsl.epsilon
    return rmse(y_hat, y, mask, reduction) / delta


def nrmse_2(y_hat: FrameArray,
            y: FrameArray,
            mask: Optional[FrameArray] = None,
            reduction: ReductionType = 'mean') -> MetricOutputType:
    r"""Compute the `Normalized Root Mean Squared Error (NRMSE)
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.
    Normalization is by the power of the true signal :math:`y`

    .. math::

        \text{NRMSE}_2 = \frac{\sqrt{\frac{\sum_{i=1}^n (\hat{y}_i -
        y_i)^2}{n}} }{\sum_{i=1}^n y_i^2}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric
            using only the values at valid indices (with :attr:`mask`
            set to :obj:`True`).
        reduction (str): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. ``'mean'``: the sum of the output
            will be divided by the number of elements in the output,
            ``'sum'``: the output will be summed.
            (default: ``'mean'``)

    Returns:
        float: The power-normalzized NRMSE.
    """
    if mask is None:
        power_y = np.square(y).sum()
    else:
        power_y = np.square(y[np.asarray(mask, dtype=bool)]).sum()
    return rmse(y_hat, y, mask, reduction) / power_y


def r2(y_hat: FrameArray,
       y: FrameArray,
       mask: Optional[FrameArray] = None,
       reduction: ReductionType = 'mean',
       nan_to_zero: bool = False,
       mean_axis: Union[int, Tuple] = None) -> float:
    r"""Compute the `coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_ :math:`R^2`
    between the estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        R^{2} = 1 - \frac{\sum_{i} (\hat{y}_i - y_i)^2}
        {\sum_{i} (\bar{y} - y_i)^2}

    where :math:`\bar{y}=\frac{1}{n}\sum_{i=1}^n y_i` is the mean of :math:`y`.

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric using only
            the values at valid indices (with :attr:`mask` set to :obj:`True`).
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_zero (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)
        mean_axis (int, Tuple, optional): the axis along which the mean of y is
            computed, to compute the variance of y needed in the denominator of
            the R2 formula.
    Returns:
         float | np.ndarray: The :math:`R^2`.
    """
    mse_ = mse(y_hat, y, mask, reduction, nan_to_zero)
    mean_val = np.mean(y, axis=mean_axis, keepdims=True)
    variance = mse(mean_val, y, mask, reduction, nan_to_zero)
    return 1. - (mse_ / variance)


def mre(y_hat: FrameArray,
        y: FrameArray,
        mask: Optional[FrameArray] = None) -> float:
    r"""Compute the MAE normalized by the L1-norm of the true signal :math:`y`,
    i.e.

    .. math::

        \text{MRE} = \frac{\sum_{i=1}^n |\hat{y}_i - y_i|}{\sum_{i=1}^n |y_i|}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (tFrameArray): The ground-truth variable.
        mask (FrameArray, optional): If provided, compute the metric using
            only the values at valid indices (with :attr:`mask` set to
            :obj:`True`).
            (default: :obj:`None`)

    Returns:
        float: The computed MRE value.
    """
    if mask is None:
        den = np.sum(np.abs(y)) + tsl.epsilon
    else:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        den = np.sum(np.abs(y[mask])) + tsl.epsilon
    err = mae(y_hat, y, mask, reduction='sum')
    return err / den
