"""Unit tests for :class:`tsl.data.mixin.DataParsingMixin`.

The mixin is exercised in isolation through a minimal host that only exposes the
attributes the mixin reads (``precision``, ``n_steps``, ``n_nodes``, ``n_edges``,
``edge_index``).
"""
import numpy as np
import pytest
import torch

from tsl.data.mixin import DataParsingMixin


class _Host(DataParsingMixin):
    def __init__(self, n_steps=10, n_nodes=3, n_edges=4, precision=32,
                 edge_index=None):
        self.n_steps = n_steps
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.precision = precision
        self.edge_index = edge_index


def _host(**kwargs):
    return _Host(**kwargs)


# -- _parse_target ----------------------------------------------------------

def test_parse_target_promotes_to_t_n_c():
    h = _host()
    out = h._parse_target(np.arange(30).reshape(10, 3).astype('float32'))
    assert tuple(out.shape) == (10, 3, 1)


def test_parse_target_1d_promoted():
    h = _host()
    out = h._parse_target(np.arange(10).astype('float32'))
    assert tuple(out.shape) == (10, 1, 1)


@pytest.mark.parametrize('precision,dtype', [
    (16, torch.float16), (32, torch.float32), (64, torch.float64),
])
def test_parse_target_precision(precision, dtype):
    h = _host(precision=precision)
    out = h._parse_target(np.zeros((10, 3), dtype='float64'))
    assert out.dtype == dtype


# -- _parse_covariate -------------------------------------------------------

def test_parse_covariate_infers_pattern():
    h = _host()
    obj, pattern = h._parse_covariate(np.zeros((10, 3, 2), dtype='float32'))
    assert pattern == 't n f'
    assert tuple(obj.shape) == (10, 3, 2)


def test_parse_covariate_explicit_pattern_validated():
    h = _host()
    obj, pattern = h._parse_covariate(np.zeros((10, 2), dtype='float32'),
                                      pattern='t f')
    assert pattern == 't f'


def test_parse_covariate_pattern_shape_mismatch():
    h = _host()
    with pytest.raises(ValueError):
        h._parse_covariate(np.zeros((9, 3), dtype='float32'), pattern='t n')


def test_parse_covariate_allow_broadcasting():
    h = _host()
    obj, _ = h._parse_covariate(np.zeros((1, 3), dtype='float32'),
                                pattern='t n', allow_broadcasting=True)
    assert tuple(obj.shape) == (1, 3)
    # without broadcasting the size-1 axis is rejected
    with pytest.raises(ValueError):
        h._parse_covariate(np.zeros((1, 3), dtype='float32'), pattern='t n')


def test_parse_covariate_precision_gated():
    h = _host(precision=32)
    converted, _ = h._parse_covariate(np.zeros((10, 3), dtype='float64'),
                                      pattern='t n')
    assert converted.dtype == torch.float32
    kept, _ = h._parse_covariate(np.zeros((10, 3), dtype='float64'),
                                 pattern='t n', convert_precision=False)
    assert kept.dtype == torch.float64


# -- _parse_connectivity ----------------------------------------------------

def test_parse_connectivity_none():
    assert _host()._parse_connectivity(None) == (None, None)


def test_parse_connectivity_edge_index_with_weight():
    h = _host()
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
    edge_weight = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    ei, ew = h._parse_connectivity((edge_index, edge_weight),
                                   target_layout='edge_index')
    assert tuple(ei.shape) == (2, 4)
    assert ew.dtype == torch.float32  # precision converted


def test_parse_connectivity_dense_validates_nodes_no_weight():
    h = _host()
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
    edge_weight = torch.tensor([1., 2., 3., 4.])
    adj, ew = h._parse_connectivity((edge_index, edge_weight),
                                    target_layout='dense')
    assert tuple(adj.shape) == (3, 3)
    assert ew is None


# -- _check_pattern / _check_same_dim ---------------------------------------

def test_check_pattern_validates_each_token():
    h = _host(n_steps=10, n_nodes=3)
    h._check_pattern(torch.zeros(10, 3, 2), 't n f', 'x')  # ok
    with pytest.raises(ValueError):
        h._check_pattern(torch.zeros(10, 5, 2), 't n f', 'x')


def test_check_pattern_e_requires_edge_index():
    h = _host(edge_index=None)
    with pytest.raises(AssertionError):
        h._check_pattern(torch.zeros(4, 2), 'e f', 'x')


def test_check_same_dim_broadcasting():
    h = _host(n_nodes=3)
    # size 1 allowed only when broadcasting
    h._check_same_dim(1, 'n_nodes', 'x', allow_broadcasting=True)
    with pytest.raises(ValueError):
        h._check_same_dim(1, 'n_nodes', 'x', allow_broadcasting=False)


# -- _check_name ------------------------------------------------------------

def test_check_name_rejects_existing_attribute():
    h = _host()
    with pytest.raises(ValueError):
        h._check_name('n_steps')


def test_check_name_accepts_fresh_name():
    # should not raise
    _host()._check_name('a_brand_new_attribute')


# -- _value_to_kwargs -------------------------------------------------------

def test_value_to_kwargs_dataarray():
    h = _host()
    arr = np.zeros((10, 3))
    assert h._value_to_kwargs(arr) == dict(value=arr)


def test_value_to_kwargs_sequence_positional():
    h = _host()
    out = h._value_to_kwargs([np.zeros(3), 't n', True])
    assert out['pattern'] == 't n'
    assert out['add_to_input_map'] is True


def test_value_to_kwargs_mapping_unknown_key():
    h = _host()
    with pytest.raises(AssertionError):
        h._value_to_kwargs({'not_a_valid_key': 1})


def test_value_to_kwargs_invalid_type():
    h = _host()
    with pytest.raises(TypeError):
        h._value_to_kwargs(42)
