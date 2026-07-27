"""Unit tests for the data loaders.

:class:`StaticGraphLoader` batches items of a single dataset that share one
(static) topology, exploiting the dataset's batched indexing.
:class:`DisjointGraphLoader` merges a list of :class:`~tsl.data.Data` with
*different* topologies into one disconnected graph.
"""
import numpy as np
import pytest
import torch

from tsl.data import Data, DisjointBatch, SpatioTemporalDataset, StaticBatch
from tsl.data.loader import DisjointGraphLoader, StaticGraphLoader


# -- builders ---------------------------------------------------------------

def _dataset(n_steps=45, n_nodes=3, n_channels=1, window=4, horizon=2):
    target = np.arange(n_steps * n_nodes * n_channels).reshape(
        n_steps, n_nodes, n_channels).astype('float32')
    return SpatioTemporalDataset(target=target, window=window, horizon=horizon)


def _graph(n_nodes, t=4, c=2):
    """A single :class:`Data` with a ring topology over ``n_nodes`` nodes."""
    edge_index = torch.stack([torch.arange(n_nodes),
                              (torch.arange(n_nodes) + 1) % n_nodes])
    return Data(input={'x': torch.randn(t, n_nodes, c)},
                target={'y': torch.randn(2, n_nodes, c)},
                edge_index=edge_index,
                pattern={'x': 't n c', 'y': 't n c', 'edge_index': '2 e'})


# -- StaticGraphLoader ------------------------------------------------------

def test_static_loader_yields_static_batches_with_batch_dim():
    ds = _dataset()
    loader = StaticGraphLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, StaticBatch)
    # a leading batch dimension is added: (b, t, n, c)
    assert batch.input.x.shape[0] == 8
    assert batch.input.x.shape[1:] == (ds.window, ds.n_nodes, ds.n_channels)


@pytest.mark.parametrize('batch_size', [1, 8, 16])
def test_static_loader_batch_count_and_coverage(batch_size):
    ds = _dataset()
    loader = StaticGraphLoader(ds, batch_size=batch_size, shuffle=False)
    n = len(ds)
    sizes = [b.input.x.shape[0] for b in loader]
    assert len(loader) == -(-n // batch_size)        # ceil
    assert sum(sizes) == n                            # every sample once


def test_static_loader_drop_last():
    ds = _dataset()
    loader = StaticGraphLoader(ds, batch_size=8, drop_last=True)
    sizes = [b.input.x.shape[0] for b in loader]
    assert len(loader) == len(ds) // 8                # floor
    assert all(s == 8 for s in sizes)                 # no short batch


def test_static_loader_uses_batch_sampler():
    ds = _dataset()
    loader = StaticGraphLoader(ds, batch_size=8)
    # the loader fetches whole batches at once via the BatchSampler
    assert loader._auto_collation is False
    assert loader._index_sampler is loader.batch_sampler


def test_static_loader_shuffle_changes_order_not_content():
    ds = _dataset()
    first = lambda b: b.input.x[:, 0, 0, 0]  # distinguishing per-sample value
    ordered = torch.cat([first(b) for b in
                         StaticGraphLoader(ds, batch_size=8, shuffle=False)])
    torch.manual_seed(0)
    shuffled = torch.cat([first(b) for b in
                          StaticGraphLoader(ds, batch_size=8, shuffle=True)])
    assert set(shuffled.tolist()) == set(ordered.tolist())   # same samples
    assert shuffled.tolist() != ordered.tolist()             # different order


def test_static_loader_ignores_user_collate_fn():
    ds = _dataset()
    loader = StaticGraphLoader(ds, batch_size=8, collate_fn=lambda x: x)
    assert isinstance(next(iter(loader)), StaticBatch)


# -- DisjointGraphLoader ----------------------------------------------------

def test_disjoint_loader_merges_different_topologies():
    graphs = [_graph(3), _graph(5), _graph(4)]
    loader = DisjointGraphLoader(graphs, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert isinstance(batch, DisjointBatch)
    # nodes of all graphs are concatenated into one disconnected graph
    assert batch.input.x.shape[1] == 3 + 5 + 4
    # the batch vector groups nodes back to their source graph
    assert batch.batch.shape[0] == 3 + 5 + 4
    assert batch.batch.unique().tolist() == [0, 1, 2]


def test_disjoint_loader_force_batch_adds_dim():
    graphs = [_graph(3), _graph(5), _graph(4)]
    plain = next(iter(DisjointGraphLoader(graphs, batch_size=3)))
    forced = next(iter(DisjointGraphLoader(graphs, batch_size=3,
                                           force_batch=True)))
    assert forced.input.x.ndim == plain.input.x.ndim + 1
    assert forced.input.x.shape[0] == 1


def test_disjoint_loader_exclude_keys():
    graphs = [_graph(3), _graph(5), _graph(4)]
    batch = next(iter(DisjointGraphLoader(graphs, batch_size=3,
                                          exclude_keys=['y'])))
    assert 'y' not in batch.target.keys()


def test_disjoint_loader_stores_lightning_attrs_and_ignores_collate():
    graphs = [_graph(3), _graph(5)]
    loader = DisjointGraphLoader(graphs, batch_size=2, shuffle=True,
                                 force_batch=True, follow_batch=['x'],
                                 exclude_keys=['y'], collate_fn=lambda x: x)
    assert loader.batch_size == 2
    assert loader.shuffle is True
    assert loader.force_batch is True
    assert loader.follow_batch == ['x']
    assert loader.exclude_keys == ['y']
    # a user collate_fn is dropped in favour of DisjointBatch.from_data_list
    assert isinstance(next(iter(loader)), DisjointBatch)
