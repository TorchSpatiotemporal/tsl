"""Unit tests for :class:`~tsl.data.preprocessing.scalers.RobustScaler`."""
import numpy as np
import pytest
from scipy import stats

from tsl.data.preprocessing.scalers import RobustScaler

from .helpers import (constant_feature_target, masked_target, random_target,
                      ref_robust)


def test_default_iqr_median_and_range():
    x = random_target((60, 3, 2))
    sc = RobustScaler().fit(x)
    bias, scale = ref_robust(x, axis=0)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-6)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)


def test_custom_quantile_range():
    x = random_target((60, 3, 2))
    sc = RobustScaler(quantile_range=(10., 90.)).fit(x)
    bias, scale = ref_robust(x, axis=0, quantile_range=(10., 90.))
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-6)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)


@pytest.mark.parametrize('axis', [0, 1, (0, 1)])
def test_various_axes(axis):
    x = random_target((40, 4, 2))
    sc = RobustScaler(axis=axis).fit(x)
    bias, scale = ref_robust(x, axis=axis)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-6)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)


def test_invalid_quantile_range_raises():
    x = random_target((20, 2, 1))
    with pytest.raises(ValueError):
        RobustScaler(quantile_range=(75., 25.)).fit(x)
    with pytest.raises(ValueError):
        RobustScaler(quantile_range=(-1., 75.)).fit(x)
    with pytest.raises(ValueError):
        RobustScaler(quantile_range=(25., 110.)).fit(x)


def test_unit_variance_adjusts_scale():
    x = random_target((60, 3, 2))
    q = (25., 75.)
    sc = RobustScaler(quantile_range=q, unit_variance=True).fit(x)
    bias, scale = ref_robust(x, axis=0, quantile_range=q, unit_variance=True)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)
    # sanity: equals the non-adjusted scale divided by the ppf span
    plain = RobustScaler(quantile_range=q).fit(x)
    adjust = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    np.testing.assert_allclose(sc.scale, plain.scale / adjust, rtol=1e-6)


def test_masked_fit_uses_nan_statistics():
    x, mask = masked_target((60, 3, 2))
    sc = RobustScaler().fit(x, mask=mask)
    bias, scale = ref_robust(x, axis=0, mask=mask)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-5)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-5)


def test_constant_feature_scale_is_one():
    x = constant_feature_target((60, 3, 2))
    sc = RobustScaler().fit(x)
    np.testing.assert_allclose(sc.scale[..., -1], 1.0)
    assert np.isfinite(sc.transform(x)).all()
