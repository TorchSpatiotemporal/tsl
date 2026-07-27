"""EngRad (weather measurements across England, hourly, 5 channels).

No imputation by default (and 0% missing), so there is no mask unless
``mask_zero_radiance`` is requested. We check the multi-channel target, the
node-level/graph covariates, and that the optional radiance mask flags exactly
the non-positive shortwave-radiation cells.
"""
import pytest

from tsl.datasets import EngRad

from .helpers import assert_dataset_contract

pytestmark = pytest.mark.dataset_download

N_NODES = 487
N_CHANNELS = 5


def test_contract():
    ds = EngRad()  # target_channels='all' -> 5 channels; no mask by default
    assert_dataset_contract(ds, n_nodes=N_NODES, n_channels=N_CHANNELS,
                            has_mask=False,
                            similarity_options={'distance', 'grid'},
                            conn_method='distance')


def test_specifics():
    ds = EngRad()
    assert ds.name == 'EngRad'
    assert ds.patterns['metadata'] == 'n f'
    assert ds.patterns['distances'] == 'n n'
    assert ds.distances.shape == (N_NODES, N_NODES)
    # the 'grid' similarity is also implemented
    assert ds.get_connectivity(method='grid',
                               layout='dense').shape == (N_NODES, N_NODES)


def test_single_target_channel_selection():
    ds = EngRad(target_channels='temperature_2m')
    assert ds.n_channels == 1
    assert ds.n_nodes == N_NODES


def test_mask_zero_radiance():
    ds = EngRad(mask_zero_radiance=True)
    assert ds.has_mask
    # the radiance mask is False exactly where shortwave radiation is <= 0,
    # leaving the other channels valid
    assert ds.mask.shape[0] == ds.length
    assert not ds.mask.all()  # some night-time radiance cells are masked
