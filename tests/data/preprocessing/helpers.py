"""Shared fixtures and independent numpy references for the preprocessing
(scaler) unit tests.
"""
import numpy as np
from scipy import stats

# -- machine-precision rule (mirrors scalers.zeros_to_one_) ------------------


def ref_zeros_to_one(scale):
    """Independent reimplementation of ``scalers.zeros_to_one_`` for arrays."""
    scale = np.array(scale, copy=True)
    eps = 10 * np.finfo(scale.dtype).eps
    scale[np.isclose(scale, 0., atol=eps, rtol=eps)] = 1.0
    return scale


# -- data builders -----------------------------------------------------------


def random_target(shape=(50, 3, 2), seed=0):
    """A reproducible ``float32`` array of shape ``(T, N, F)``."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def constant_feature_target(shape=(50, 3, 2), seed=0):
    """Like :func:`random_target` but with the last feature held constant, to
    exercise the near-zero-scale (``zeros_to_one_``) path."""
    x = random_target(shape, seed)
    x[..., -1] = 7.0
    return x


def masked_target(shape=(50, 3, 2), seed=0, frac=0.3):
    """A target together with a boolean mask selecting ``~(1 - frac)`` entries.

    The underlying data is finite everywhere; the masked-out entries are the
    ones the scaler must ignore (the source replaces them with NaN internally).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    mask = rng.random(shape) > frac
    return x, mask


# -- independent parameter references ---------------------------------------


def _masked_nan(x, mask):
    return np.where(mask, x, np.nan).astype(np.float32) if mask is not None \
        else x


def ref_standard(x, axis=0, mask=None, keepdims=True):
    x = np.asarray(x)
    if mask is None:
        bias = x.mean(axis=axis, keepdims=keepdims)
        scale = x.std(axis=axis, keepdims=keepdims)
    else:
        xm = _masked_nan(x, mask)
        bias = np.nanmean(xm, axis=axis, keepdims=keepdims).astype(x.dtype)
        scale = np.nanstd(xm, axis=axis, keepdims=keepdims).astype(x.dtype)
    return bias, ref_zeros_to_one(scale)


def ref_minmax(x, axis=0, out_range=(0., 1.), mask=None, keepdims=True):
    x = np.asarray(x)
    out_min, out_max = out_range
    if mask is None:
        x_min = x.min(axis=axis, keepdims=keepdims)
        x_max = x.max(axis=axis, keepdims=keepdims)
    else:
        xm = _masked_nan(x, mask)
        x_min = np.nanmin(xm, axis=axis, keepdims=keepdims).astype(x.dtype)
        x_max = np.nanmax(xm, axis=axis, keepdims=keepdims).astype(x.dtype)
    scale = (x_max - x_min) / (out_max - out_min)
    scale = ref_zeros_to_one(scale)
    bias = x_min - out_min * scale
    return bias, scale


def ref_robust(x, axis=0, quantile_range=(25., 75.), mask=None,
               unit_variance=False, keepdims=True):
    x = np.asarray(x)
    q_min, q_max = quantile_range
    if mask is None:
        bias = np.median(x, axis=axis, keepdims=keepdims)
        min_q, max_q = np.percentile(x, quantile_range, axis=axis,
                                     keepdims=keepdims)
    else:
        xm = _masked_nan(x, mask)
        bias = np.nanmedian(xm, axis=axis, keepdims=keepdims).astype(x.dtype)
        min_q, max_q = np.nanpercentile(xm, quantile_range, axis=axis,
                                        keepdims=keepdims)
    scale = (max_q - min_q).astype(x.dtype)
    scale = ref_zeros_to_one(scale)
    if unit_variance:
        adjust = stats.norm.ppf(q_max / 100.) - stats.norm.ppf(q_min / 100.)
        scale = scale / adjust
    return bias, scale
