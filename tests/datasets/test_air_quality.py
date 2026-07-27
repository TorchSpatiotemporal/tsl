"""AirQuality (PM2.5 imputation benchmark).

This is an *imputation* dataset (``MissingValuesMixin``), so the leakage surface
is the eval/training split of the mask: cells reserved for evaluation must never
leak into the training mask, and the (global ``temporal_mean``) imputation must
only fill masked-out cells. We test on the small 36-station Beijing subset, which
ships with a curated ``eval_mask``.
"""
import numpy as np
import pytest

from tsl.datasets import AirQuality
from tsl.datasets.air_quality import AirQualitySplitter, infer_mask

from .helpers import assert_dataset_contract, assert_mask_is_raw

pytestmark = pytest.mark.dataset_download

N_NODES_SMALL = 36
N_NODES_FULL = 437


def test_contract():
    ds = AirQuality(small=True)
    assert_dataset_contract(ds, n_nodes=N_NODES_SMALL, n_channels=1,
                            has_mask=True, similarity_options={'distance'},
                            conn_method='distance')


def test_specifics():
    ds = AirQuality(small=True)
    assert ds.name == 'AQI36'
    assert ds.dist.shape == (N_NODES_SMALL, N_NODES_SMALL)
    # the air-quality split is the default and is dataset-specific
    assert isinstance(ds.get_splitter(), AirQualitySplitter)


def test_full_dataset_node_count():
    ds = AirQuality(small=False)
    assert ds.name == 'AQI'
    assert ds.n_nodes == N_NODES_FULL


# -- leakage: eval / training mask ------------------------------------------


def test_eval_mask_subset_of_mask():
    ds = AirQuality(small=True)
    assert hasattr(ds, 'eval_mask')
    # eval cells must be observed cells (set_eval_mask intersects with mask)
    assert np.all(ds.eval_mask <= ds.mask)


def test_training_and_eval_masks_disjoint():
    ds = AirQuality(small=True)
    tr = ds.training_mask
    # nothing is both trained on and evaluated on
    assert not np.any(tr & ds.eval_mask)
    np.testing.assert_array_equal(tr, (~ds.eval_mask) & ds.mask)


def test_imputation_only_fills_masked_cells():
    imputed = AirQuality(small=True, impute_nans=True)
    raw = AirQuality(small=True, impute_nans=False)
    # mask comes from the raw NaN pattern; observed cells are not altered
    assert_mask_is_raw(imputed, raw)
    # the global temporal-mean imputation removes the NaNs from the target
    assert np.isnan(raw.numpy()).any()
    assert not np.isnan(imputed.numpy()).any()


def test_infer_mask_holds_out_a_different_month():
    # the inferred eval mask marks values present in a month but absent in the
    # neighbouring (next) month -> it must be computed on the *raw* (NaN-bearing)
    # frame, before imputation, or there is nothing to hold out.
    ds = AirQuality(small=False, impute_nans=False)
    df = ds.dataframe()  # retains NaNs (no imputation)
    eval_mask = infer_mask(df, infer_from='next').values.astype(bool)
    obs = (~np.isnan(df.values))
    # an eval cell must be an observed cell
    assert np.all(eval_mask <= obs)
    # and there must be at least one held-out cell
    assert eval_mask.any()
