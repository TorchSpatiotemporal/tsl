"""PemsBay (325 traffic sensors, 5-min sampling).

Leakage focus: same as MetrLA -- mask from raw readings, forward-fill imputation
only touches masked-out cells.
"""
import numpy as np
import pytest

from tsl.datasets import PemsBay

from .helpers import assert_dataset_contract

pytestmark = pytest.mark.dataset_download

N_NODES = 325


def test_contract():
    ds = PemsBay()
    assert_dataset_contract(ds, n_nodes=N_NODES, n_channels=1, has_mask=True,
                            similarity_options={'distance', 'stcn'},
                            conn_method='distance')


def test_specifics():
    ds = PemsBay()
    assert ds.name == 'PemsBay'
    assert ds.dist.shape == (N_NODES, N_NODES)
    # both declared similarity methods are implemented
    for method in ('distance', 'stcn'):
        adj = ds.get_connectivity(method=method, layout='dense')
        assert adj.shape == (N_NODES, N_NODES)


def test_mask_is_from_raw_and_imputation_is_safe():
    imputed = PemsBay(mask_zeros=True)
    raw = PemsBay(mask_zeros=False)
    # raw load masks only NaNs; imputed also masks zeros -> imputed mask subset
    assert np.all(imputed.mask <= raw.mask)
    # observed (valid) cells are identical between the two loads
    valid = imputed.mask & ~np.isnan(raw.numpy())
    np.testing.assert_array_equal(imputed.numpy()[valid], raw.numpy()[valid])
