"""Elergone (370 electricity load profiles, 15-min sampling).

The loader fills missing steps with zero and then derives the mask as the
non-zero entries, so the leakage check is that masked cells are exactly zeros.
The default ``correntropy`` similarity is computed over the full (140k-step)
series and is too expensive to run here, so connectivity is only checked for the
*declared-but-unimplemented* ``pearson`` option (must raise NotImplementedError).
"""
import numpy as np
import pytest

from tsl.datasets import Elergone

from .helpers import assert_dataset_contract

pytestmark = pytest.mark.dataset_download

N_NODES = 370


def test_contract():
    ds = Elergone()
    assert_dataset_contract(ds, n_nodes=N_NODES, n_channels=1, has_mask=True,
                            similarity_options={'correntropy', 'pearson'},
                            conn_method=None)  # correntropy too slow to compute


def test_specifics():
    ds = Elergone()
    assert ds.name == 'Electricity'
    assert ds.freq is not None


def test_pearson_option_not_implemented():
    # pearson is advertised in similarity_options but not implemented
    ds = Elergone()
    with pytest.raises(NotImplementedError):
        ds.get_similarity('pearson')


def test_mask_matches_zero_fill():
    ds = Elergone()
    # missing values are zero-filled and then masked out -> mask == (value != 0)
    np.testing.assert_array_equal(ds.mask, ds.numpy() != 0.)
