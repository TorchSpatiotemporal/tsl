"""Unit tests for :class:`~tsl.data.preprocessing.scalers.StandardScaler`."""
import numpy as np
import pytest
import torch

from tsl.data.preprocessing.scalers import StandardScaler

from .helpers import (constant_feature_target, masked_target, random_target,
                      ref_standard)


def test_fit_default_axis_mean_std():
    x = random_target((50, 3, 2))
    sc = StandardScaler().fit(x)
    bias, scale = ref_standard(x, axis=0)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-6)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)


def test_fit_returns_self():
    sc = StandardScaler()
    assert sc.fit(random_target((10, 2, 1))) is sc


@pytest.mark.parametrize('axis', [0, 1, (0, 1)])
def test_fit_various_axes(axis):
    x = random_target((40, 4, 3))
    sc = StandardScaler(axis=axis).fit(x)
    bias, scale = ref_standard(x, axis=axis)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-6)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-6)


def test_keepdims_true_preserves_ndim():
    x = random_target((50, 3, 2))
    sc = StandardScaler().fit(x, keepdims=True)
    assert sc.bias.shape == (1, 3, 2)


def test_keepdims_false_drops_reduced_dim():
    x = random_target((50, 3, 2))
    sc = StandardScaler().fit(x, keepdims=False)
    assert sc.bias.shape == (3, 2)


def test_masked_fit_uses_nan_statistics():
    x, mask = masked_target((50, 3, 2))
    sc = StandardScaler().fit(x, mask=mask)
    bias, scale = ref_standard(x, axis=0, mask=mask)
    np.testing.assert_allclose(sc.bias, bias, rtol=1e-5)
    np.testing.assert_allclose(sc.scale, scale, rtol=1e-5)


def test_constant_feature_scale_is_one():
    x = constant_feature_target((50, 3, 2))
    sc = StandardScaler().fit(x)
    # the constant last feature would have std ~0 -> forced to 1
    np.testing.assert_allclose(sc.scale[..., -1], 1.0)
    # transform of the constant feature does not blow up
    out = sc.transform(x)
    assert np.isfinite(out).all()


def test_fit_transform_matches_fit_then_transform():
    x = random_target((30, 2, 2))
    sc1 = StandardScaler()
    out = sc1.fit_transform(x)
    sc2 = StandardScaler().fit(x)
    np.testing.assert_allclose(out, sc2.transform(x), rtol=1e-6)


def test_dtype_preserved():
    x = random_target((20, 2, 1)).astype(np.float32)
    sc = StandardScaler().fit(x)
    assert sc.bias.dtype == np.float32
    assert sc.scale.dtype == np.float32


def test_tensor_input_produces_tensor_params():
    x = torch.as_tensor(random_target((20, 2, 1)))
    sc = StandardScaler().fit(x)
    assert isinstance(sc.bias, torch.Tensor)
    assert isinstance(sc.scale, torch.Tensor)
