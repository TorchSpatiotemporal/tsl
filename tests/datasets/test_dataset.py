"""Base :class:`tsl.datasets.prototypes.Dataset` -- the abstract interface that
every dataset inherits: directory bookkeeping, the connectivity/similarity
pipeline, splitter resolution and pickle IO.

Connectivity expectations are computed by hand from a tiny fixed similarity
matrix (never read back from ``get_connectivity``), mirroring the golden-rule
discipline of the SpatioTemporalDataset suite.
"""
import os

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

import tsl
from tsl import config
from tsl.data.datamodule.splitters import TemporalSplitter
from tsl.datasets.prototypes import Dataset
from tsl.ops.connectivity import adj_to_edge_index

from .helpers import DummyDataset

# a small symmetric similarity with zero diagonal; every off-diagonal distinct
SIM = np.array([[0., 1., 2.],
                [1., 0., 3.],
                [2., 3., 0.]], dtype='float32')


def make(**kwargs):
    return DummyDataset(similarity=SIM, **kwargs)


# -- naming / construction --------------------------------------------------


def test_name_defaults_to_class_name():
    assert make().name == 'DummyDataset'
    assert make(name='custom').name == 'custom'


def test_invalid_similarity_score_raises():
    with pytest.raises(ValueError):
        DummyDataset(similarity=SIM, similarity_score='not_a_method')


def test_len_and_repr():
    ds = make()
    assert len(ds) == ds.length == 10
    assert 'DummyDataset' in repr(ds)
    assert 'n_nodes=3' in repr(ds)


# -- directory bookkeeping --------------------------------------------------


def test_root_dir_default(monkeypatch):
    ds = make()
    assert ds.root is None
    expected = os.path.join(config.data_dir, 'DummyDataset')
    assert ds.root_dir == expected


def test_root_dir_expands_user():
    ds = make()
    ds.root = '~/some/path/..'
    assert ds.root_dir == os.path.expanduser(os.path.normpath('~/some/path/..'))


def test_root_dir_invalid_type_raises():
    ds = make()
    ds.root = 123
    with pytest.raises(ValueError):
        _ = ds.root_dir


def test_required_file_names_default_to_raw():
    class WithRaw(DummyDataset):

        @property
        def raw_file_names(self):
            return ['a.csv', 'b.csv']

    ds = WithRaw(similarity=SIM)
    assert ds.required_file_names == ['a.csv', 'b.csv']


def test_file_paths_list_form(tmp_path):
    class WithRaw(DummyDataset):

        @property
        def raw_file_names(self):
            return ['a.csv', 'b.csv']

    ds = WithRaw(similarity=SIM)
    ds.root = str(tmp_path)
    assert ds.raw_files_paths == [str(tmp_path / 'a.csv'),
                                  str(tmp_path / 'b.csv')]
    assert ds.raw_files_paths_list == ds.raw_files_paths
    assert ds.required_files_paths_list == ds.raw_files_paths


def test_file_paths_mapping_form(tmp_path):
    class WithMap(DummyDataset):

        @property
        def raw_file_names(self):
            return {'x': 'a.csv', 'y': 'b.csv'}

    ds = WithMap(similarity=SIM)
    ds.root = str(tmp_path)
    assert ds.raw_files_paths == {'x': str(tmp_path / 'a.csv'),
                                  'y': str(tmp_path / 'b.csv')}
    # the *_list flavour collapses a mapping to its values
    assert set(ds.raw_files_paths_list) == set(ds.raw_files_paths.values())


def test_file_paths_str_form(tmp_path):
    class WithStr(DummyDataset):

        @property
        def raw_file_names(self):
            return 'only.csv'

    ds = WithStr(similarity=SIM)
    ds.root = str(tmp_path)
    assert ds.raw_files_paths == [str(tmp_path / 'only.csv')]


# -- connectivity pipeline --------------------------------------------------


def test_connectivity_full():
    adj = make().get_connectivity(method='full', layout='dense')
    np.testing.assert_array_equal(adj, np.ones((3, 3)))


def test_connectivity_identity():
    adj = make().get_connectivity(method='identity', layout='dense')
    np.testing.assert_array_equal(adj, np.eye(3))


def test_connectivity_from_similarity():
    adj = make().get_connectivity(method='fixed', layout='dense')
    np.testing.assert_array_equal(adj, SIM)


def test_connectivity_binary_weights():
    adj = make().get_connectivity(method='fixed', binary_weights=True,
                                  layout='dense')
    np.testing.assert_array_equal(adj, (SIM > 0).astype(SIM.dtype))


def test_connectivity_threshold():
    adj = make().get_connectivity(method='fixed', threshold=2., layout='dense')
    expected = SIM.copy()
    expected[expected < 2.] = 0
    np.testing.assert_array_equal(adj, expected)


def test_connectivity_include_self_false_zeros_diagonal():
    adj = make().get_connectivity(method='full', include_self=False,
                                  layout='dense')
    expected = np.ones((3, 3))
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(adj, expected)


def test_connectivity_force_symmetric():
    asym = np.array([[0., 5., 0.],
                     [0., 0., 0.],
                     [0., 0., 0.]], dtype='float32')
    adj = DummyDataset(similarity=asym).get_connectivity(method='fixed',
                                                         force_symmetric=True,
                                                         layout='dense')
    np.testing.assert_array_equal(adj, np.maximum(asym, asym.T))


def test_connectivity_normalize_axis():
    adj = make().get_connectivity(method='fixed', normalize_axis=1,
                                  layout='dense')
    expected = SIM / (SIM.sum(1, keepdims=True) + tsl.epsilon)
    np.testing.assert_allclose(adj, expected, rtol=1e-6)


def test_connectivity_knn():
    # keep only the single strongest incoming neighbour per node (+ self)
    from tsl.ops.similarities import top_k
    adj = make().get_connectivity(method='fixed', knn=1, include_self=False,
                                  layout='dense')
    expected = top_k(SIM, 1, include_self=False, keep_values=True)
    np.testing.assert_array_equal(adj, expected)


def test_connectivity_layout_edge_index():
    edge_index, edge_weight = make().get_connectivity(method='fixed',
                                                      layout='edge_index')
    exp_ei, exp_ew = adj_to_edge_index(SIM)
    np.testing.assert_array_equal(edge_index, exp_ei)
    np.testing.assert_array_equal(edge_weight, exp_ew)


@pytest.mark.parametrize('layout,cls', [('coo', coo_matrix),
                                        ('csr', csr_matrix),
                                        ('csc', csc_matrix)])
def test_connectivity_sparse_layouts(layout, cls):
    adj = make().get_connectivity(method='fixed', layout=layout)
    assert isinstance(adj, cls)
    np.testing.assert_array_equal(adj.toarray(), SIM)


def test_connectivity_invalid_layout_raises():
    with pytest.raises(ValueError):
        make().get_connectivity(method='fixed', layout='nope')


def test_connectivity_deprecated_sparse_kwarg():
    with pytest.warns(UserWarning):
        ei = make().get_connectivity(method='fixed', sparse=True)
    assert isinstance(ei, tuple) and len(ei) == 2
    with pytest.warns(UserWarning):
        dense = make().get_connectivity(method='fixed', sparse=False)
    np.testing.assert_array_equal(dense, SIM)


# -- similarity pipeline ----------------------------------------------------


def test_get_similarity_default_method():
    np.testing.assert_array_equal(make().get_similarity(), SIM)


def test_get_similarity_invalid_method_raises():
    with pytest.raises(ValueError):
        make().get_similarity('does_not_exist')


def test_get_similarity_save_and_cache(tmp_path):
    ds = make()
    ds.root = str(tmp_path)
    os.makedirs(ds.root_dir, exist_ok=True)
    sim = ds.get_similarity('fixed', save=True)
    cached = list(tmp_path.glob('sim_*.npy'))
    assert len(cached) == 1
    # second call must return the same matrix (loaded from cache)
    np.testing.assert_array_equal(ds.get_similarity('fixed', save=True), sim)


# -- splitter resolution ----------------------------------------------------


def test_get_splitter_resolves_default_method():
    ds = make()  # default_splitting_method == 'temporal'
    assert isinstance(ds.get_splitter(), TemporalSplitter)


def test_get_splitter_unknown_method_raises():
    with pytest.raises(NotImplementedError):
        make().get_splitter('not_a_splitter')


def test_get_splitter_dataset_specific_override():
    sentinel = TemporalSplitter()

    class WithSplitter(DummyDataset):

        def get_splitter(self, method=None, **kwargs):
            if method == 'custom':
                return sentinel

    ds = WithSplitter(similarity=SIM)
    assert ds.get_splitter('custom') is sentinel
    # falls back to library lookup for non-overridden methods
    assert isinstance(ds.get_splitter('temporal'), TemporalSplitter)


# -- pickle IO --------------------------------------------------------------


def test_save_load_pickle_roundtrip(tmp_path):
    ds = make(name='pkl')
    path = str(tmp_path / 'ds.pkl')
    ds.save_pickle(path)
    loaded = Dataset.load_pickle(path)
    assert isinstance(loaded, Dataset)
    assert loaded.name == 'pkl'
    np.testing.assert_array_equal(loaded.get_similarity('fixed'), SIM)


def test_getstate_excludes_wrapped_splitter():
    assert 'get_splitter' not in make().__getstate__()


def test_load_pickle_wrong_type_raises(tmp_path):
    ds = make()
    path = str(tmp_path / 'ds.pkl')
    ds.save_pickle(path)

    class Other(Dataset):
        pass

    with pytest.raises(TypeError):
        Other.load_pickle(path)
