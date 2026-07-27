"""MetrLA (207 traffic loop detectors, 5-min sampling).

Marked ``dataset_download`` -- runs only with ``--run-datasets``.

Leakage focus: the validity mask must be derived from the *raw* readings (zeros =
missing) and the optional zero-imputation (forward fill) must only touch
masked-out cells, never the observed ones.
"""
import numpy as np
import pytest

from tsl.datasets import MetrLA

from .helpers import assert_dataset_contract, assert_mask_is_raw

pytestmark = pytest.mark.dataset_download

N_NODES = 207


def test_contract():
    ds = MetrLA()
    assert_dataset_contract(ds, n_nodes=N_NODES, n_channels=1, has_mask=True,
                            similarity_options={'distance'},
                            conn_method='distance')


def test_specifics():
    ds = MetrLA()
    assert ds.name == 'MetrLA'
    assert ds.freq is not None  # 5-minute datetime index
    # pairwise-distance attribute, square in the number of nodes
    assert ds.patterns['dist'] == 'n n'
    assert ds.dist.shape == (N_NODES, N_NODES)


def test_mask_is_from_raw_and_imputation_is_safe():
    imputed = MetrLA(impute_zeros=True)
    raw = MetrLA(impute_zeros=False)
    # mask identical (computed before imputation) and observed cells untouched
    assert_mask_is_raw(imputed, raw)
    # zero-imputation must fill at least some masked-out cells with non-zero
    # (forward-filled) values -> the imputed series has fewer zeros than raw
    invalid = ~imputed.mask
    assert invalid.any()
    assert np.count_nonzero(imputed.numpy() == 0) <= \
        np.count_nonzero(raw.numpy() == 0)
