"""Unit tests for the :class:`~tsl.data.preprocessing.scalers.Scaler` base
class: linear transform, numpy/torch conversions, repr and save/load."""
import os

import numpy as np
import pytest
import torch

import tsl
from tsl.data.preprocessing.scalers import Scaler

from .helpers import random_target


def test_default_params():
    sc = Scaler()
    assert sc.bias == 0.
    assert sc.scale == 1.
    assert set(sc.params()) == {'bias', 'scale'}


def test_transform_uses_epsilon():
    # transform divides by (scale + tsl.epsilon), not by scale alone
    x = np.array([2.0], dtype=np.float32)
    sc = Scaler(bias=0., scale=2.)
    expected = (x - 0.) / (2. + tsl.epsilon)
    np.testing.assert_allclose(sc.transform(x), expected, rtol=1e-6)


def test_transform_inverse_roundtrip():
    x = random_target((20, 2, 1))
    sc = Scaler(bias=3.0, scale=2.5)
    np.testing.assert_allclose(sc.inverse_transform(sc.transform(x)), x,
                               rtol=1e-5)


def test_call_is_transform():
    x = random_target((10, 2, 1))
    sc = Scaler(bias=1.0, scale=4.0)
    np.testing.assert_array_equal(sc(x), sc.transform(x))


def test_torch_inplace_converts_params():
    sc = Scaler(bias=np.float32(2.0), scale=np.float32(3.0))
    ret = sc.torch()
    assert ret is sc  # inplace returns self
    assert isinstance(sc.bias, torch.Tensor)
    assert isinstance(sc.scale, torch.Tensor)
    # torch() applies atleast_1d
    assert sc.bias.ndim >= 1 and sc.scale.ndim >= 1


def test_torch_not_inplace_leaves_original():
    sc = Scaler(bias=2.0, scale=3.0)
    out = sc.torch(inplace=False)
    assert out is not sc
    assert isinstance(out.bias, torch.Tensor)
    # original untouched
    assert not isinstance(sc.bias, torch.Tensor)


def test_numpy_inplace_converts_params():
    sc = Scaler(bias=torch.tensor([2.0]), scale=torch.tensor([3.0]))
    ret = sc.numpy()
    assert ret is sc
    assert isinstance(sc.bias, np.ndarray)
    assert isinstance(sc.scale, np.ndarray)


def test_numpy_not_inplace_leaves_original():
    sc = Scaler(bias=torch.tensor([2.0]), scale=torch.tensor([3.0]))
    out = sc.numpy(inplace=False)
    assert out is not sc
    assert isinstance(out.bias, np.ndarray)
    assert isinstance(sc.bias, torch.Tensor)


def test_base_fit_raises_not_implemented():
    sc = Scaler()
    with pytest.raises(NotImplementedError):
        sc.fit(random_target((10, 2, 1)))


def test_save_load_numpy_npz(tmp_path):
    sc = Scaler(bias=np.array([1.5], dtype=np.float32),
                scale=np.array([2.5], dtype=np.float32))
    path = sc.save(str(tmp_path / 'scaler'))
    # numpy params are stored via np.savez_compressed; save() must return the
    # actual '.npz' path so the round-trip below works
    assert path.endswith('.npz')
    assert os.path.exists(path)
    loaded = Scaler.load(path)
    np.testing.assert_allclose(loaded.bias, sc.bias)
    np.testing.assert_allclose(loaded.scale, sc.scale)


def test_save_load_torch_pt(tmp_path):
    sc = Scaler(bias=torch.tensor([1.5]), scale=torch.tensor([2.5]))
    path = sc.save(str(tmp_path / 'scaler'))
    assert path.endswith('.pt')
    loaded = Scaler.load(path)
    assert torch.allclose(loaded.bias, sc.bias)
    assert torch.allclose(loaded.scale, sc.scale)


def test_save_make_dir_creates_nested_dir(tmp_path):
    sc = Scaler(bias=torch.tensor([0.]), scale=torch.tensor([1.]))
    target = tmp_path / 'a' / 'b' / 'scaler'
    path = sc.save(str(target), make_dir=True)
    assert (tmp_path / 'a' / 'b').is_dir()
    assert Scaler.load(path) is not None


def test_load_invalid_extension_raises(tmp_path):
    bad = tmp_path / 'scaler.txt'
    bad.write_text('not a scaler')
    with pytest.raises(RuntimeError):
        Scaler.load(str(bad))
