"""PvUS (simulated US photovoltaic power). We load the smaller ``west`` zone.

No imputation; the optional ``mask_zeros`` masks night-time (zero-production)
hours, so the leakage check is that masked cells correspond exactly to zeros.
"""
import numpy as np
import pytest

from tsl.datasets import PvUS

from .helpers import assert_dataset_contract

pytestmark = pytest.mark.dataset_download

N_NODES_WEST = 1082


def test_contract():
    ds = PvUS(zones='west')  # mask_zeros defaults to False -> no mask
    assert_dataset_contract(ds, n_nodes=N_NODES_WEST, n_channels=1,
                            has_mask=False,
                            similarity_options={'distance', 'correntropy'},
                            conn_method='distance')


def test_specifics():
    ds = PvUS(zones='west')
    assert ds.name == 'PvUS-west'
    assert ds.patterns['metadata'] == 'n f'
    assert ds.metadata.shape[0] == N_NODES_WEST


def test_invalid_zone_raises():
    with pytest.raises(ValueError):
        PvUS(zones='north')


def test_mask_zeros_flags_night_hours():
    ds = PvUS(zones='west', mask_zeros=True)
    assert ds.has_mask
    # masked-out cells are exactly the zero-production (night) entries
    np.testing.assert_array_equal(ds.mask, ds.numpy() > 0)
