from typing import Literal, Union, Optional

import numpy as np

import tsl
from tsl.ops.framearray import framearray_to_numpy
from tsl.typing import FrameArray

ReductionType = Literal['mean', 'sum', 'none']
MetricOutputType = Union[float, np.ndarray]

__all__ = [
    'mae', 'nmae', 'mape', 'mse', 'rmse', 'nrmse', 'nrmse_2', 'r2'
]


def _masked_reduce(x: FrameArray, reduction: ReductionType,
                   mask: Optional[FrameArray] = None,
                   nan_to_num: bool = False) -> MetricOutputType:
    x = framearray_to_numpy(x)  # covert x to ndarray if not already (no copy)
    # 'none': return x with x[i] = 0/nan where mask[i] == False
    if reduction == 'none':
        if mask is not None:
            masked_idxs = np.logical_not(framearray_to_numpy(mask))
            x[masked_idxs] = 0 if nan_to_num else np.nan
        return x
    # 'mean'/'sum': return mean/sum of x[mask == True]
    if mask is not None:
        x = x[framearray_to_numpy(mask)]
    if reduction == 'mean':
        return np.mean(x)
    elif reduction == 'sum':
        return np.sum(x)
    else:
        raise ValueError(f"reduction {reduction} not allowed, must be one of "
                         "['mean', 'sum', 'none'].")


def mae(y_hat: FrameArray, y: FrameArray, mask: Optional[FrameArray] = None,
        reduction: ReductionType = 'mean',
        nan_to_num: bool = False) -> MetricOutputType:
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
            :attr:`nan_to_num`).
            (default: :obj:`None`)
        reduction (str): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the sum of the output will be divided by the
            number of elements in the output, ``'sum'``: the output will be
            summed. (default: ``'mean'``)
        nan_to_num (bool): If :obj:`True`, then masked values in output are
            converted to :obj:`0`. This has an effect only when :attr:`mask` is
            not :obj:`None` and :attr:`reduction` is :obj:`'none'`.
            (default: :obj:`False`)

    Returns:
        float | np.ndarray: The Mean Absolute Error.
    """
    err = np.abs(y_hat - y)
    return _masked_reduce(err, reduction, mask, nan_to_num)


def nmae(y_hat: FrameArray, y: FrameArray) -> float:
    r"""Compute the *Normalized Mean Absolute Error* (NMAE) between the estimate
    :math:`\hat{y}` and the true value :math:`y`. The NMAE is the `Mean Absolute
    Error (MAE) <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ scaled by
    the maximum difference in the target data, i.e.

    .. math::

        \text{NMAE}_{\%} = \frac{\sum_{i=1}^n |\hat{y}_i - y_i|}{n} \cdot
        \frac{100\%}{\max(y) - \min(y)}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.

    Returns:
        float: The Normalized Mean Absolute Error in percentage.
    """
    delta = np.max(y) - np.min(y) + tsl.epsilon
    return mae(y_hat, y) * 100 / delta


def mape(y_hat: FrameArray, y: FrameArray) -> float:
    r"""Compute the `Mean Absolute Percentage Error (MAPE)
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MAPE}_{\%} = \frac{100\%}{n} \sum_{i=1}^n \frac{|\hat{y}_i - y_i|}
        {y_i}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.

    Returns:
        float: The Mean Absolute Percentage Error in percentage.
    """
    return 100 * np.abs((y_hat - y) / (y + tsl.epsilon)).mean()


def mse(y_hat: FrameArray, y: FrameArray) -> float:
    r"""Compute the `Mean Squared Error (MSE)
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{MSE} = \frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{n}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.

    Returns:
        float: The Mean Squared Error.
    """
    return np.square(y_hat - y).mean()


def rmse(y_hat: FrameArray, y: FrameArray) -> float:
    r"""Compute the `Root Mean Squared Error (MSE)
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_ between the
    estimate :math:`\hat{y}` and the true value :math:`y`, i.e.

    .. math::

        \text{RMSE} = \sqrt{\frac{\sum_{i=1}^n (\hat{y}_i - y_i)^2}{n}}

    Args:
        y_hat (FrameArray): The estimated variable.
        y (FrameArray): The ground-truth variable.

    Returns:
        float: The Root Mean Squared Error.
    """
    return np.sqrt(mse(y_hat, y))


def nrmse(y_hat, y):
    delta = np.max(y) - np.min(y) + tsl.epsilon
    return rmse(y_hat, y) * 100 / delta


def nrmse_2(y_hat, y):
    nrmse_ = np.sqrt(np.square(y_hat - y).sum() / np.square(y).sum())
    return nrmse_ * 100


def r2(y_hat: FrameArray, y: FrameArray) -> float:
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

    Returns:
        float: The :math:`R^2`.
    """
    return 1. - np.square(y_hat - y).sum() / (np.square(y.mean(0) - y).sum())


def masked_mae(y_hat, y, mask=None):
    if mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = y_hat[mask] - y[mask]
    return np.abs(err).mean()


def masked_mape(y_hat, y, mask=None):
    if mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = (y_hat[mask] - y[mask]) / (y[mask] + tsl.epsilon)
    return np.abs(err).mean()


def masked_mse(y_hat, y, mask=None):
    if mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = y_hat[mask] - y[mask]
    return np.square(err).mean()


def masked_rmse(y_hat, y, mask=None):
    if mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = np.square(y_hat[mask] - y[mask])
    return np.sqrt(err.mean())


def masked_mre(y_hat, y, mask=None):
    if mask is None:
        mask = slice(None)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = np.abs(y_hat[mask] - y[mask])
    return err.sum() / (y[mask].sum() + tsl.epsilon)
