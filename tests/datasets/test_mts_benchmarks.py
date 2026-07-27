"""Multivariate time-series benchmarks (Electricity / Traffic / Solar /
Exchange), sharing ``_MTSBenchmarkDataset``.

These have no graph (``similarity_options is None``) and are not imputed: the mask
simply flags zero entries, so the leakage check is that the mask matches the raw
zero pattern and the target is left untouched.
"""
import numpy as np
import pytest

from tsl.datasets import (ElectricityBenchmark, ExchangeBenchmark,
                          SolarBenchmark, TrafficBenchmark)

from .helpers import assert_dataset_contract

pytestmark = pytest.mark.dataset_download

# (class, n_nodes, freq)
CASES = [
    (ElectricityBenchmark, 321, '1H'),
    (TrafficBenchmark, 862, '1H'),
    (SolarBenchmark, 137, '10T'),
    (ExchangeBenchmark, 8, '1D'),
]


@pytest.mark.parametrize('cls,n_nodes,freq', CASES)
def test_contract(cls, n_nodes, freq):
    ds = cls()
    assert_dataset_contract(ds, n_nodes=n_nodes, n_channels=1, has_mask=True,
                            conn_method=None)  # no connectivity for these
    assert ds.similarity_options is None
    assert ds.freq is not None


@pytest.mark.parametrize('cls,n_nodes,freq', CASES)
def test_mask_matches_raw_zeros(cls, n_nodes, freq):
    ds = cls()
    # the mask flags exactly the non-zero entries of the (un-imputed) target
    expected = ds.numpy() != 0.
    np.testing.assert_array_equal(ds.mask, expected)
