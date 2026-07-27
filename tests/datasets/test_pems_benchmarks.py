"""PeMSD benchmarks (PeMS03/04/07/08), sharing the ``_PeMS`` base.

These come pre-imputed (0% missing), so the mask is opt-in via ``mask_zeros``.
PeMS04 and PeMS08 additionally carry ``occupancy`` and ``speed`` covariates that
must stay aligned to the flow target's nodes/timesteps.
"""
import numpy as np
import pytest

from tsl.datasets import PeMS03, PeMS04, PeMS07, PeMS08

from .helpers import assert_dataset_contract

pytestmark = pytest.mark.dataset_download

# (class, n_nodes, has_occupancy_speed)
CASES = [
    (PeMS03, 358, False),
    (PeMS04, 307, True),
    (PeMS07, 883, False),
    (PeMS08, 170, True),
]
SIM_OPTS = {'distance', 'stcn', 'binary'}


@pytest.mark.parametrize('cls,n_nodes,_extra', CASES)
def test_contract(cls, n_nodes, _extra):
    ds = cls()  # mask_zeros defaults to False -> no mask
    assert_dataset_contract(ds, n_nodes=n_nodes, n_channels=1, has_mask=False,
                            similarity_options=SIM_OPTS, conn_method='distance')


@pytest.mark.parametrize('cls,n_nodes,has_extra', CASES)
def test_covariates(cls, n_nodes, has_extra):
    ds = cls()
    assert ds.dist.shape == (n_nodes, n_nodes)
    if has_extra:
        # occupancy/speed are time-varying covariates aligned to the target
        for name in ('occupancy', 'speed'):
            assert ds.patterns[name] == 't n f'
            cov = getattr(ds, name)
            assert cov.shape[0] == ds.length
            # aligned to the same nodes as the flow target
            np.testing.assert_array_equal(cov.columns.unique(0),
                                          ds.target.columns.unique(0))


@pytest.mark.parametrize('cls,n_nodes', [(PeMS08, 170)])
def test_binary_similarity_is_zero_one(cls, n_nodes):
    ds = cls()
    adj = ds.get_connectivity(method='binary', layout='dense')
    assert set(np.unique(adj)).issubset({0.0, 1.0})


def test_mask_zeros_flags_zero_flow():
    ds = PeMS08(mask_zeros=True)
    assert ds.has_mask
    # a masked-out cell must correspond to a zero in the (pre-imputed) flow
    invalid = ~ds.mask
    if invalid.any():
        assert np.all(ds.numpy()[invalid] == 0)
