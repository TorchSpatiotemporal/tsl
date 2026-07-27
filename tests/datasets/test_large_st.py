"""LargeST (large-scale California traffic). We exercise the smallest subset
(San Diego, single year 2019) to keep the download bounded.

Leakage focus: the mask comes from the raw (resampled) NaN pattern; the default
``"zero"`` imputation only fills masked-out cells. The ``"nearest"`` mode uses
``ffill().bfill()`` -- the backfill is non-causal, so we additionally confirm
those backfilled cells stay flagged invalid by the mask.
"""
import numpy as np
import pytest

from tsl.datasets import LargeST

from .helpers import assert_dataset_contract, assert_mask_is_raw

pytestmark = pytest.mark.dataset_download

N_NODES_SD = 716


def _sd(imputation_mode='zero'):
    return LargeST(subset='SD', year=2019, imputation_mode=imputation_mode)


def test_contract():
    ds = _sd()
    assert_dataset_contract(ds, n_nodes=N_NODES_SD, n_channels=1, has_mask=True,
                            similarity_options={'precomputed'},
                            conn_method='precomputed')


def test_specifics():
    ds = _sd()
    assert ds.name == 'LargeST-SD'
    assert ds.patterns['metadata'] == 'n f'
    assert ds.patterns['adj'] == 'n n'
    assert ds.adj.shape == (N_NODES_SD, N_NODES_SD)


def test_invalid_subset_raises():
    with pytest.raises(ValueError):
        LargeST(subset='NOPE', year=2019)


def test_zero_imputation_is_leak_safe():
    imputed = _sd('zero')
    raw = LargeST(subset='SD', year=2019, imputation_mode=None)
    # mask from raw NaNs; observed cells untouched; holes filled with 0
    assert_mask_is_raw(imputed, raw)
    invalid = ~imputed.mask
    if invalid.any():
        assert np.all(imputed.numpy()[invalid] == 0)


def test_nearest_imputation_backfill_stays_masked():
    nearest = _sd('nearest')
    raw = LargeST(subset='SD', year=2019, imputation_mode=None)
    # ffill+bfill removes all NaNs, but every filled cell remains invalid in the
    # mask, so a mask-respecting model never trains on the (possibly future-
    # derived) imputed values.
    assert np.all(nearest.mask == raw.mask)
    invalid = ~nearest.mask
    if invalid.any():
        assert not np.isnan(nearest.numpy()[invalid]).any()
