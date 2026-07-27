"""``tsl.datasets.prototypes.TabularDataset`` -- the in-memory storage layer:
target parsing, shape/index properties, covariates, masks, frame extraction and
the (spatial) aggregate/reduce ops.

Targets are grid-valued (unique value per cell) so any mis-indexing surfaces, and
every expectation is built independently from numpy, never from the dataset.
"""
import numpy as np
import pandas as pd
import pytest

from tsl.datasets.prototypes import TabularDataset

from .helpers import grid_array, grid_dataframe, make_tabular

N_STEPS, N_NODES, N_CHANNELS = 10, 3, 2


# -- target parsing ---------------------------------------------------------


def test_parse_1d_array_expands_to_3d():
    ds = TabularDataset(target=np.arange(N_STEPS).astype('float32'))
    assert ds.shape == (N_STEPS, 1, 1)


def test_parse_2d_array_expands_to_3d():
    ds = TabularDataset(target=grid_array(N_STEPS, N_NODES, 1)[..., 0])
    assert ds.shape == (N_STEPS, N_NODES, 1)


def test_parse_3d_array_kept():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    assert ds.shape == (N_STEPS, N_NODES, N_CHANNELS)
    np.testing.assert_array_equal(ds.numpy(), arr)


def test_parse_4d_array_raises():
    with pytest.raises(AssertionError):
        TabularDataset(target=np.zeros((4, 3, 2, 1)))


def test_parse_dataframe_single_level_columns():
    df = pd.DataFrame(np.arange(N_STEPS * N_NODES).reshape(N_STEPS, N_NODES),
                      columns=['a', 'b', 'c'], dtype='float32')
    ds = TabularDataset(target=df)
    assert ds.n_nodes == N_NODES
    assert ds.n_channels == 1
    assert list(ds.nodes) == ['a', 'b', 'c']


def test_parse_dataframe_multiindex_columns():
    df = grid_dataframe(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=df)
    assert ds.n_nodes == N_NODES
    assert ds.n_channels == N_CHANNELS


@pytest.mark.parametrize('precision,dtype', [(16, 'float16'),
                                             (32, 'float32'),
                                             (64, 'float64')])
def test_precision_conversion(precision, dtype):
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS, dtype='float64')
    ds = TabularDataset(target=arr, precision=precision)
    assert ds.numpy().dtype == np.dtype(dtype)


# -- properties -------------------------------------------------------------


def test_dimension_properties():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    assert ds.length == N_STEPS
    assert ds.n_nodes == N_NODES
    assert ds.n_channels == N_CHANNELS
    assert ds.shape == (N_STEPS, N_NODES, N_CHANNELS)


def test_index_nodes_channels_for_array_backing():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    np.testing.assert_array_equal(ds.index, np.arange(N_STEPS))
    np.testing.assert_array_equal(ds.nodes, np.arange(N_NODES))
    np.testing.assert_array_equal(ds.channels, np.arange(N_CHANNELS))


def test_patterns_includes_target_mask_and_covariates():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    assert ds.patterns == {'target': 't n f'}
    ds.set_mask(np.ones_like(ds.numpy()))
    assert ds.patterns['mask'] == 't n f'
    ds.add_covariate('u', grid_array(N_STEPS, N_NODES, 1), 't n f')
    assert ds.patterns['u'] == 't n f'


# -- covariates -------------------------------------------------------------


def test_add_covariate_infers_pattern():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    cov = grid_array(N_STEPS, N_NODES, 4)
    ds.add_covariate('u', cov)
    assert ds.patterns['u'] == 't n f'
    np.testing.assert_array_equal(ds.u, cov)


def test_add_covariate_explicit_pattern_attribute_matrix():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    adj = np.ones((N_NODES, N_NODES), dtype='float32')
    ds.add_covariate('adj', adj, 'n n')
    assert ds.patterns['adj'] == 'n n'
    assert ds.n_covariates == 1


def test_add_covariate_name_collision_raises():
    ds = make_tabular()
    with pytest.raises(ValueError):
        ds.add_covariate('target', grid_array(N_STEPS, N_NODES, 1))


def test_add_exogenous_node_level_and_global():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    ds.add_exogenous('u', grid_array(N_STEPS, N_NODES, 1))
    assert ds.patterns['u'] == 't n f'
    ds.add_exogenous('global_g', grid_array(N_STEPS, 1, 1)[:, 0])
    assert ds.patterns['g'] == 't f'


def test_exogenous_vs_attributes_partition():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    ds.add_covariate('u', grid_array(N_STEPS, N_NODES, 1), 't n f')
    ds.add_covariate('adj', np.ones((N_NODES, N_NODES)), 'n n')
    assert set(ds.exogenous) == {'u'}
    assert set(ds.attributes) == {'adj'}
    assert ds.has_covariates and ds.n_covariates == 2


def test_covariate_delattr_removes_it():
    ds = make_tabular()
    ds.add_covariate('u', grid_array(N_STEPS, N_NODES, 1), 't n f')
    assert 'u' in ds.covariates
    del ds.u
    assert 'u' not in ds.covariates


def test_getattr_missing_raises():
    with pytest.raises(AttributeError):
        _ = make_tabular().does_not_exist


# -- mask -------------------------------------------------------------------


def test_set_and_get_mask():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    m = np.zeros((N_STEPS, N_NODES, N_CHANNELS), dtype=bool)
    m[0] = True
    ds.set_mask(m)
    assert ds.has_mask
    np.testing.assert_array_equal(ds.get_mask(), m)
    assert ds.mask.dtype == bool


def test_mask_single_feature_broadcasts():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    m = np.ones((N_STEPS, N_NODES, 1), dtype=bool)
    ds.set_mask(m)
    assert ds.mask.shape[-1] == 1


def test_mask_bad_feature_count_raises():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    bad = np.ones((N_STEPS, N_NODES, N_CHANNELS + 1), dtype=bool)
    with pytest.raises(RuntimeError):
        ds.set_mask(bad)


def test_get_mask_dtype_and_dataframe():
    ds = make_tabular(N_STEPS, N_NODES, 1)
    ds.set_mask(np.ones((N_STEPS, N_NODES, 1), dtype=bool))
    assert ds.get_mask(dtype='uint8').dtype == np.uint8
    df = ds.get_mask(as_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (N_STEPS, N_NODES)


def test_get_mask_fallback_when_unset():
    arr = grid_array(N_STEPS, N_NODES, 1)
    arr[0, 0, 0] = np.nan
    ds = TabularDataset(target=arr)
    assert not ds.has_mask
    expected = ~np.isnan(arr)
    np.testing.assert_array_equal(ds.get_mask(), expected)


def test_del_mask():
    ds = make_tabular()
    ds.set_mask(np.ones_like(ds.numpy(), dtype=bool))
    del ds.mask
    assert not ds.has_mask


# -- frame extraction -------------------------------------------------------


def test_get_frame_target_only():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    frame, pattern = ds.get_frame('target')
    assert pattern == 't n f'
    np.testing.assert_array_equal(frame, arr)


def test_get_frame_time_and_node_index():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    frame, _ = ds.get_frame('target', node_index=[0, 2],
                            time_index=[1, 3, 5])
    np.testing.assert_array_equal(frame, arr[[1, 3, 5]][:, [0, 2]])


def test_get_frame_concatenates_covariate():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    cov = grid_array(N_STEPS, N_NODES, N_CHANNELS, offset=10_000)
    ds.add_covariate('u', cov, 't n f')
    frame, pattern = ds.get_frame(['target', 'u'], cat_dim=-1)
    assert pattern == 't n f'
    np.testing.assert_array_equal(frame,
                                  np.concatenate([arr, cov], axis=-1))


def test_get_frame_channel_selection_on_dataframe():
    df = grid_dataframe(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=df)
    frame, _ = ds.get_frame({'target': [1]})
    expected = ds.numpy()[..., [1]]
    np.testing.assert_array_equal(frame, expected)


def test_get_frame_unknown_channel_raises():
    df = grid_dataframe(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=df)
    with pytest.raises(KeyError):
        ds.get_frame({'target': [99]})


def test_get_frame_as_dataframe():
    df = grid_dataframe(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=df)
    out = ds.get_frame('target', return_pattern=False, as_numpy=False)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (N_STEPS, N_NODES * N_CHANNELS)


# -- aggregate / reduce -----------------------------------------------------


def test_aggregate_default_sums_all_nodes():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    agg = ds.aggregate()
    assert agg.n_nodes == 1
    np.testing.assert_allclose(agg.numpy(), arr.sum(axis=1, keepdims=True))
    # original dataset untouched (aggregate returns a copy)
    assert ds.n_nodes == N_NODES


def test_aggregate_with_mapping():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    agg = ds.aggregate({0: [0, 1], 1: [2]})
    assert agg.n_nodes == 2
    expected0 = arr[:, [0, 1]].sum(axis=1)
    np.testing.assert_allclose(agg.numpy()[:, 0], expected0)
    np.testing.assert_allclose(agg.numpy()[:, 1], arr[:, 2])


def test_aggregate_mask_with_tolerance():
    arr = grid_array(N_STEPS, N_NODES, 1)
    ds = TabularDataset(target=arr)
    mask = np.ones((N_STEPS, N_NODES, 1), dtype=bool)
    mask[0, 0, 0] = False  # one invalid in cluster -> mean 2/3 < 1
    ds.set_mask(mask)
    agg = ds.aggregate()  # default tolerance 0 -> any invalid invalidates
    assert not bool(agg.mask[0, 0, 0])
    assert bool(agg.mask[1, 0, 0])


def test_aggregate_propagates_to_node_covariate():
    arr = grid_array(N_STEPS, N_NODES, 1)
    ds = TabularDataset(target=arr)
    cov = grid_array(N_STEPS, N_NODES, 1, offset=10_000)
    ds.add_covariate('u', cov, 't n f')
    agg = ds.aggregate()  # all nodes -> 1 cluster, summed
    # the node-level covariate must be aggregated along the node axis too
    np.testing.assert_allclose(agg.u, cov.sum(axis=1, keepdims=True))


def test_reduce_propagates_to_node_covariate():
    arr = grid_array(N_STEPS, N_NODES, 1)
    ds = TabularDataset(target=arr)
    cov = grid_array(N_STEPS, N_NODES, 1, offset=10_000)
    ds.add_covariate('u', cov, 't n f')
    red = ds.reduce(time_index=np.arange(4), node_index=[0, 2])
    np.testing.assert_array_equal(red.u, cov[:4][:, [0, 2]])


def test_reduce_by_node_and_time():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    red = ds.reduce(time_index=np.arange(4), node_index=[0, 2])
    assert red.shape == (4, 2, N_CHANNELS)
    np.testing.assert_array_equal(red.numpy(), arr[:4][:, [0, 2]])


def test_reduce_inplace_variant():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    ds.reduce_(node_index=[1])
    assert ds.n_nodes == 1
    np.testing.assert_array_equal(ds.numpy(), arr[:, [1]])


# -- misc -------------------------------------------------------------------


def test_fill_nan():
    arr = grid_array(N_STEPS, N_NODES, 1)
    arr[2, 1, 0] = np.nan
    ds = TabularDataset(target=arr)
    ds.fill_nan_(value=0.)
    assert ds.numpy()[2, 1, 0] == 0.
    assert not np.isnan(ds.numpy()).any()


def test_dataframe_numpy_roundtrip():
    arr = grid_array(N_STEPS, N_NODES, N_CHANNELS)
    ds = TabularDataset(target=arr)
    df = ds.dataframe()
    np.testing.assert_array_equal(df.values.reshape(ds.shape), arr)
    out, idx = ds.numpy(return_idx=True)
    np.testing.assert_array_equal(out, arr)
    np.testing.assert_array_equal(np.asarray(idx), np.arange(N_STEPS))


def test_copy_is_independent():
    ds = make_tabular(N_STEPS, N_NODES, N_CHANNELS)
    clone = ds.copy()
    clone.target[0, 0, 0] = -999
    assert ds.numpy()[0, 0, 0] != -999


def test_synchronize_context_manager():
    ds = make_tabular()
    assert ds.force_synchronization
    with ds.synchronize(False):
        assert not ds.force_synchronization
    assert ds.force_synchronization
