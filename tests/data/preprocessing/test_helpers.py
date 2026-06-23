"""Unit tests for the module-level helpers ``zeros_to_one_`` and
``fit_wrapper`` in :mod:`tsl.data.preprocessing.scalers`."""
import numpy as np
import torch

from tsl.data.preprocessing.scalers import (Scaler, StandardScaler,
                                            zeros_to_one_)


def test_zeros_to_one_scalar_near_zero():
    assert zeros_to_one_(0.0) == 1.0
    assert zeros_to_one_(1e-20) == 1.0


def test_zeros_to_one_scalar_nonzero_unchanged():
    assert zeros_to_one_(2.5) == 2.5


def test_zeros_to_one_array_inplace():
    scale = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    out = zeros_to_one_(scale)
    # operates in place and returns the same array
    assert out is scale
    np.testing.assert_array_equal(scale, [1.0, 1.0, 2.0])


def test_zeros_to_one_array_only_near_zero_changed():
    scale = np.array([1e-20, 3.0, 1e-18], dtype=np.float64)
    zeros_to_one_(scale)
    np.testing.assert_array_equal(scale, [1.0, 3.0, 1.0])


def test_fit_wrapper_numpy_input_keeps_numpy_params():
    x = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)
    sc = StandardScaler().fit(x)
    assert isinstance(sc.bias, np.ndarray)
    assert isinstance(sc.scale, np.ndarray)


def test_fit_wrapper_tensor_input_triggers_torch():
    x = torch.as_tensor(
        np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32))
    sc = StandardScaler().fit(x)
    assert isinstance(sc.bias, torch.Tensor)
    assert isinstance(sc.scale, torch.Tensor)


def test_fit_wrapper_returns_scaler_object():
    sc = StandardScaler()
    ret = sc.fit(np.ones((5, 2), dtype=np.float32))
    assert isinstance(ret, Scaler)
    assert ret is sc
