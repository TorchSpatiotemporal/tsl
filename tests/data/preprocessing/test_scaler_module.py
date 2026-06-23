"""Unit tests for :class:`~tsl.data.preprocessing.scalers.ScalerModule`."""
import numpy as np
import pytest
import torch

import tsl
from tsl.data.preprocessing.scalers import (Scaler, ScalerModule,
                                            StandardScaler)

from .helpers import random_target


def _module(t=5, n=3, f=2, pattern='t n f'):
    bias = torch.arange(t * n * f, dtype=torch.float32).reshape(t, n, f)
    scale = torch.ones(t, n, f) * 2.0
    return ScalerModule(bias=bias, scale=scale, pattern=pattern)


# -- construction ------------------------------------------------------------


def test_init_from_scaler_sets_inherited_and_buffers():
    sc = StandardScaler().fit(random_target((20, 3, 2)))
    mod = ScalerModule(sc)
    assert mod.inherited_from is StandardScaler
    assert isinstance(mod.bias, torch.Tensor)
    assert isinstance(mod.scale, torch.Tensor)


def test_init_from_module_inherits_pattern_and_class():
    sc = StandardScaler().fit(random_target((20, 3, 2)))
    src = ScalerModule(sc)
    src.pattern = 't n f'
    mod = ScalerModule(src)
    assert mod.inherited_from is StandardScaler
    assert mod.pattern == 't n f'


def test_init_from_raw_params():
    mod = ScalerModule(bias=2.0, scale=3.0)
    assert mod.inherited_from is None
    assert torch.allclose(mod.bias, torch.tensor([2.0]))
    assert torch.allclose(mod.scale, torch.tensor([3.0]))


def test_params_registered_as_buffers():
    mod = _module()
    sd = mod.state_dict()
    assert 'bias' in sd and 'scale' in sd
    # bias/scale are buffers, hence not parameters
    assert len(list(mod.parameters())) == 0


def test_setattr_detaches_and_atleast_1d():
    mod = ScalerModule(bias=0., scale=1.)
    src = torch.tensor(5.0, requires_grad=True)
    mod.scale = src
    assert mod.scale.ndim >= 1
    assert not mod.scale.requires_grad


# -- transform ---------------------------------------------------------------


def test_transform_inverse_roundtrip_tensor():
    mod = _module()
    x = torch.randn(5, 3, 2)
    np.testing.assert_allclose(mod.inverse_transform(mod.transform(x)).numpy(),
                               x.numpy(), rtol=1e-4)


def test_transform_uses_epsilon():
    mod = ScalerModule(bias=0., scale=2.0)
    x = torch.tensor([4.0])
    expected = x / (2.0 + tsl.epsilon)
    np.testing.assert_allclose(mod.transform_tensor(x).numpy(),
                               expected.numpy(), rtol=1e-6)


def test_transform_recurses_over_dict_and_list():
    mod = _module()
    x = torch.randn(5, 3, 2)
    out_dict = mod.transform({'a': x})
    assert isinstance(out_dict, dict)
    np.testing.assert_allclose(out_dict['a'].numpy(),
                               mod.transform_tensor(x).numpy())
    out_list = mod.transform([x])
    assert isinstance(out_list, list)
    np.testing.assert_allclose(out_list[0].numpy(),
                               mod.transform_tensor(x).numpy())


def test_call_is_transform():
    mod = _module()
    x = torch.randn(5, 3, 2)
    np.testing.assert_array_equal(mod(x).numpy(), mod.transform(x).numpy())


# -- pattern bookkeeping -----------------------------------------------------


def test_pattern_axes_and_size_properties():
    mod = _module(t=5, n=3, f=2, pattern='t n f')
    assert mod.t_axis == 0
    assert mod.n_axis == 1
    assert mod.t == 5
    assert mod.n == 3


def test_properties_none_when_dim_absent():
    mod = ScalerModule(bias=torch.zeros(3, 2), scale=torch.ones(3, 2),
                       pattern='n f')
    assert mod.t is None  # no 't' in pattern
    assert mod.n == 3


def test_multiple_node_dims_raises():
    with pytest.raises(RuntimeError):
        ScalerModule(bias=torch.zeros(3, 3), scale=torch.ones(3, 3),
                     pattern='n n')


# -- rearrange ---------------------------------------------------------------


def test_rearrange_end_pattern_only():
    mod = _module(t=5, n=3, f=2, pattern='t n f')
    out = mod.rearrange('t f n')
    assert out.pattern == 't f n'
    assert tuple(out.bias.shape) == (5, 2, 3)


def test_rearrange_explicit_pattern():
    mod = _module(t=5, n=3, f=2, pattern='t n f')
    out = mod.rearrange('t n f -> t f n')
    assert out.pattern == 't f n'
    assert tuple(out.scale.shape) == (5, 2, 3)


def test_rearrange_mismatched_start_pattern_raises():
    mod = _module(t=5, n=3, f=2, pattern='t n f')
    with pytest.raises(RuntimeError):
        mod.rearrange('t f n -> n f t')


def test_rearrange_inplace():
    mod = _module(t=5, n=3, f=2, pattern='t n f')
    ret = mod.rearrange('t f n', inplace=True)
    assert ret is mod
    assert mod.pattern == 't f n'
    assert tuple(mod.bias.shape) == (5, 2, 3)


# -- slice -------------------------------------------------------------------


def test_slice_without_pattern_raises():
    mod = ScalerModule(bias=torch.zeros(5, 3, 2), scale=torch.ones(5, 3, 2))
    with pytest.raises(RuntimeError):
        mod.slice(node_index=torch.tensor([0, 1]))


def test_slice_by_node_index():
    mod = _module(t=5, n=3, f=2, pattern='t n f')
    out = mod.slice(node_index=torch.tensor([0, 2]))
    assert out.bias.size(1) == 2
    np.testing.assert_allclose(out.bias.numpy(),
                               mod.bias[:, [0, 2]].numpy())


def test_slice_by_time_index():
    mod = _module(t=5, n=3, f=2, pattern='t n f')
    out = mod.slice(time_index=torch.tensor([1, 3]))
    assert out.bias.size(0) == 2
    np.testing.assert_allclose(out.bias.numpy(), mod.bias[[1, 3]].numpy())


def test_slice_leaves_broadcastable_params_untouched():
    # time-invariant bias (size-1 on time axis) is not sliced along time
    bias = torch.zeros(1, 3, 2)
    scale = torch.ones(1, 3, 2)
    mod = ScalerModule(bias=bias, scale=scale, pattern='t n f')
    out = mod.slice(time_index=torch.tensor([1, 3]))
    assert out.bias.size(0) == 1


def test_slice_no_index_returns_copy():
    mod = _module()
    out = mod.slice()
    assert isinstance(out, ScalerModule)
    np.testing.assert_array_equal(out.bias.numpy(), mod.bias.numpy())


# -- cat ---------------------------------------------------------------------


def test_cat_two_modules_along_feature():
    a = _module(t=5, n=3, f=2, pattern='t n f')
    b = _module(t=5, n=3, f=1, pattern='t n f')
    # the only caller (SpatioTemporalDataset.get_tensor) always passes the
    # tensor sizes; cat_tensors relies on them for the broadcast shape
    out = ScalerModule.cat([a, b], dim=-1, sizes=[(5, 3, 2), (5, 3, 1)])
    assert out.scale.size(-1) == 3
    assert out.pattern == 't n f'


def test_cat_all_none_returns_none():
    assert ScalerModule.cat([None, None]) is None


def test_cat_with_none_uses_fill_and_sizes():
    a = _module(t=5, n=3, f=2, pattern='t n f')
    sizes = [(5, 3, 2), (5, 3, 1)]
    out = ScalerModule.cat([a, None], dim=-1, sizes=sizes)
    assert out.scale.size(-1) == 3
    # the filled part: bias filled with 0, scale filled with 1
    np.testing.assert_allclose(out.bias[..., -1].numpy(), 0.0)
    np.testing.assert_allclose(out.scale[..., -1].numpy(), 1.0)


# -- conversions & repr ------------------------------------------------------


def test_numpy_returns_scaler_with_ndarray_params():
    mod = _module()
    sc = mod.numpy()
    assert isinstance(sc, Scaler)
    assert isinstance(sc.bias, np.ndarray)
    np.testing.assert_allclose(sc.bias, mod.bias.numpy())


def test_get_name_reflects_inherited():
    sc = StandardScaler().fit(random_target((20, 3, 2)))
    mod = ScalerModule(sc)
    assert mod._get_name() == 'StandardScalerModule'


def test_extra_repr_includes_pattern():
    mod = _module(pattern='t n f')
    assert "pattern='t n f'" in mod.extra_repr()
