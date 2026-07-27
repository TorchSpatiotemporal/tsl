"""Shared helpers for the :mod:`tsl.datasets` test package."""
import numpy as np
import pandas as pd

from tsl.data.datamodule.splitters import Splitter
from tsl.datasets.prototypes import Dataset, DatetimeDataset, TabularDataset

# -- grid-valued target builders --------------------------------------------


def grid_array(n_steps, n_nodes, n_channels, offset=0, dtype='float32'):
    """Array whose every ``(t, n, f)`` cell holds a unique value."""
    size = n_steps * n_nodes * n_channels
    return (offset + np.arange(size)).reshape(n_steps, n_nodes,
                                              n_channels).astype(dtype)


def grid_dataframe(n_steps, n_nodes, n_channels, offset=0, index=None):
    """DataFrame backing for a grid target, with a ``(nodes, channels)``
    MultiIndex on the columns."""
    data = grid_array(n_steps, n_nodes, n_channels, offset).reshape(
        n_steps, n_nodes * n_channels)
    columns = pd.MultiIndex.from_product(
        [np.arange(n_nodes), np.arange(n_channels)],
        names=['nodes', 'channels'])
    if index is None:
        index = np.arange(n_steps)
    return pd.DataFrame(data, index=index, columns=columns)


def datetime_index(n_steps, freq='1H', start='2020-01-01'):
    return pd.date_range(start=start, periods=n_steps, freq=freq)


def make_tabular(n_steps=10, n_nodes=3, n_channels=2, **kwargs):
    return TabularDataset(target=grid_array(n_steps, n_nodes, n_channels),
                          **kwargs)


def make_datetime(n_steps=24, n_nodes=3, n_channels=1, freq='1H', **kwargs):
    df = grid_dataframe(n_steps, n_nodes, n_channels,
                        index=datetime_index(n_steps, freq=freq))
    return DatetimeDataset(target=df, **kwargs)


# -- a minimal concrete Dataset to exercise the abstract base ----------------


class DummyDataset(Dataset):
    """Smallest possible concrete :class:`Dataset`: it only knows its shape and
    a fixed similarity matrix, which is enough to drive ``get_connectivity`` /
    ``get_similarity`` / ``get_splitter`` / pickle without touching disk."""

    similarity_options = {'fixed'}

    def __init__(self, similarity=None, n_nodes=4, similarity_score='fixed',
                 **kwargs):
        self._sim = (np.asarray(similarity, dtype='float32')
                     if similarity is not None else None)
        self._n_nodes = n_nodes
        super().__init__(similarity_score=similarity_score, **kwargs)

    @property
    def length(self):
        return 10

    @property
    def n_nodes(self):
        return self._n_nodes if self._sim is None else self._sim.shape[0]

    @property
    def n_channels(self):
        return 1

    def compute_similarity(self, method, **kwargs):
        if method == 'fixed':
            return self._sim


# -- concrete-dataset contract (used by the download tests) ------------------


def assert_dataset_contract(ds, *, n_nodes=None, n_channels=None,
                            has_mask=None, similarity_options=None,
                            conn_method=None):
    """Shared structural contract every concrete dataset must satisfy.

    Expectations are passed in (derived independently from each dataset's own
    documented facts); nothing is read back from the dataset to build them.

    ``conn_method`` selects which (implemented) similarity to exercise through
    ``get_connectivity``; pass :obj:`None` to skip the computation (e.g. when the
    only method is prohibitively expensive, like correntropy over a long series).
    """
    # shape consistency
    assert ds.length == ds.numpy().shape[0]
    assert ds.shape == (ds.length, ds.n_nodes, ds.n_channels)
    assert ds.numpy().shape == ds.shape
    if n_nodes is not None:
        assert ds.n_nodes == n_nodes
    if n_channels is not None:
        assert ds.n_channels == n_channels

    # mask
    if has_mask is not None:
        assert ds.has_mask is has_mask
    if ds.has_mask:
        assert ds.mask.shape[0] == ds.length
        assert ds.mask.shape[1] == ds.n_nodes
        assert ds.mask.dtype == bool

    # dataframe / numpy round-trip
    df = ds.dataframe()
    assert df.shape[0] == ds.length
    np.testing.assert_array_equal(ds.numpy(), df.values.reshape(ds.shape))

    # declared similarity options
    if similarity_options is not None:
        assert ds.similarity_options == similarity_options

    # connectivity for one implemented method, in several layouts
    if conn_method is not None:
        dense = ds.get_connectivity(method=conn_method, layout='dense')
        assert dense.shape == (ds.n_nodes, ds.n_nodes)
        edge_index, edge_weight = ds.get_connectivity(method=conn_method,
                                                      layout='edge_index')
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == edge_weight.shape[0]
        coo = ds.get_connectivity(method=conn_method, layout='coo')
        assert coo.shape == (ds.n_nodes, ds.n_nodes)

    # splitter
    assert isinstance(ds.get_splitter(), Splitter)


def assert_mask_is_raw(ds_imputed, ds_raw):
    """Leakage check shared by datasets that impute the target: the mask must be
    computed from the *raw* (pre-imputation) data, observed cells must be left
    untouched by imputation, and only invalid cells may differ."""
    raw = ds_raw.numpy()
    imp = ds_imputed.numpy()
    mask = ds_imputed.mask
    # mask must match the raw missing/zero pattern, regardless of imputation
    np.testing.assert_array_equal(ds_imputed.mask, ds_raw.mask)
    # observed (mask True) values are identical; imputation only fills holes
    valid = np.broadcast_to(mask, raw.shape)
    finite = valid & ~np.isnan(raw)
    np.testing.assert_array_equal(imp[finite], raw[finite])
