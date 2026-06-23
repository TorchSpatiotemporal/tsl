"""Unit tests for :class:`~tsl.data.preprocessing.scalers.MinMaxScaler`."""
import numpy as np
import pytest

from tsl.data.preprocessing.scalers import MinMaxScaler

from .helpers import (constant_feature_target, masked_target, random_target,
                      ref_minmax)


def test_default_range_maps_to_unit_interval():
    x = random_target((50, 3, 2))
    sc = MinMaxScaler().fit(x)
    out = sc.transform(x)
    np.testing.assert_allclose(out.min(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.max(axis=0), 1.0, atol=1e-5)


def test_custom_range_params_match_reference():
    x = random_target((50, 3, 2))
    sc = MinMaxScaler(out_range=(-1., 1.)).fit(x)
    bias, scale = ref_minmax(x, axis=0, out_range=(-1., 1.))
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-6)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)
    out = sc.transform(x)
    np.testing.assert_allclose(out.min(axis=0), -1.0, atol=1e-5)
    np.testing.assert_allclose(out.max(axis=0), 1.0, atol=1e-5)


def test_invalid_range_raises():
    x = random_target((20, 2, 1))
    with pytest.raises(ValueError):
        MinMaxScaler(out_range=(1., 0.)).fit(x)
    with pytest.raises(ValueError):
        MinMaxScaler(out_range=(1., 1.)).fit(x)


def test_masked_fit_uses_nan_min_max():
    x, mask = masked_target((50, 3, 2))
    sc = MinMaxScaler().fit(x, mask=mask)
    bias, scale = ref_minmax(x, axis=0, mask=mask)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-5)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-5)


def test_constant_feature_scale_is_one():
    x = constant_feature_target((50, 3, 2))
    sc = MinMaxScaler().fit(x)
    np.testing.assert_allclose(sc.scale[..., -1], 1.0)
    assert np.isfinite(sc.transform(x)).all()


@pytest.mark.parametrize('axis', [0, 1])
def test_various_axes(axis):
    x = random_target((40, 4, 2))
    sc = MinMaxScaler(axis=axis).fit(x)
    bias, scale = ref_minmax(x, axis=axis)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-6)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)


def test_keepdims_false_drops_reduced_dim():
    x = random_target((50, 3, 2))
    sc = MinMaxScaler().fit(x, keepdims=False)
    assert sc.bias.shape == (3, 2)
