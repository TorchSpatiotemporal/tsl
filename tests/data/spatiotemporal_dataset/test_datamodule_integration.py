"""Integration tests for the dataset -> DataModule -> DataLoader pipeline.
"""
import numpy as np
import pytest
import torch

from tsl.data import (SpatioTemporalDataModule, SpatioTemporalDataset,
                      TemporalSplitter)
from tsl.data.preprocessing.scalers import StandardScaler


# -- builders / oracles -----------------------------------------------------

def _make_dataset(n_steps=300, n_nodes=3, n_channels=1, window=8, horizon=4,
                  **kwargs):
    target = np.arange(n_steps * n_nodes * n_channels).reshape(
        n_steps, n_nodes, n_channels).astype('float32')
    return SpatioTemporalDataset(target=target, window=window, horizon=horizon,
                                 **kwargs)


def _make_dm(dataset, scalers=None, splitter=None, batch_size=16, **kwargs):
    dm = SpatioTemporalDataModule(dataset=dataset, scalers=scalers,
                                  splitter=splitter, batch_size=batch_size,
                                  **kwargs)
    dm.setup()
    return dm


def _footprint(ds, indices):
    """All time steps (window + horizon) touched by the given samples."""
    expanded = ds.expand_indices(np.asarray(indices))
    steps = set(expanded['horizon'].numpy().ravel().tolist())
    if ds.window > 0:
        steps |= set(expanded['window'].numpy().ravel().tolist())
    return steps


def _targets(ds, indices):
    """Time steps used as prediction targets (horizon) by the given samples."""
    return set(ds.expand_indices(np.asarray(indices))['horizon'].numpy()
               .ravel().tolist())


# ===========================================================================
# Leakage: splits stay separated through the loaders.
# ===========================================================================

@pytest.mark.parametrize('window,horizon', [(8, 4), (12, 12), (6, 1)])
def test_window_offset_split_targets_never_leak_into_train(window, horizon):
    # default offset='window': an evaluation target must never appear anywhere
    # in the training footprint (as input or target).
    ds = _make_dataset(window=window, horizon=horizon)
    dm = _make_dm(ds, splitter=TemporalSplitter(val_len=0.1, test_len=0.2))
    tr, va, te = dm.trainset.indices, dm.valset.indices, dm.testset.indices
    assert len(tr) and len(va) and len(te)
    train_fp = _footprint(ds, tr)
    assert _targets(ds, va).isdisjoint(train_fp)
    assert _targets(ds, te).isdisjoint(train_fp | _footprint(ds, va))


def test_sample_offset_splits_fully_disjoint():
    # offset='sample': the footprints themselves are disjoint across splits.
    ds = _make_dataset(window=8, horizon=4)
    dm = _make_dm(ds, splitter=TemporalSplitter(0.1, 0.2, offset='sample'))
    f_tr = _footprint(ds, dm.trainset.indices)
    f_va = _footprint(ds, dm.valset.indices)
    f_te = _footprint(ds, dm.testset.indices)
    assert f_tr.isdisjoint(f_va)
    assert f_va.isdisjoint(f_te)
    assert f_tr.isdisjoint(f_te)


def test_split_lengths_and_slices_match_splitter():
    ds = _make_dataset(window=8, horizon=4)
    splitter = TemporalSplitter(val_len=0.1, test_len=0.2)
    dm = _make_dm(ds, splitter=splitter)
    # the datamodule's lengths reflect the splitter's index sets
    assert dm.train_len == len(dm.trainset.indices)
    assert dm.val_len == len(dm.valset.indices)
    assert dm.test_len == len(dm.testset.indices)
    # the cached *_slice is exactly the merged expanded footprint of the split
    expected = ds.expand_indices(dm.trainset.indices, merge=True)
    np.testing.assert_array_equal(dm.train_slice.numpy(), expected.numpy())


# ===========================================================================
# Leakage: scalers are fit on the train split only.
# ===========================================================================

def test_scaler_is_fit_on_train_slice_only():
    ds = _make_dataset(window=8, horizon=4)
    dm = _make_dm(ds, scalers={'target': StandardScaler(axis=0)},
                  splitter=TemporalSplitter(0.1, 0.2))
    bias = ds.scalers['target'].bias.numpy().ravel()
    target = ds.numpy()
    train_mean = target[dm.train_slice.numpy()].mean(0).ravel()
    whole_mean = target.mean(0).ravel()
    # fit on the train slice, NOT on the whole series
    np.testing.assert_allclose(bias, train_mean, atol=1e-3)
    assert not np.allclose(bias, whole_mean, atol=1e-3)


def test_val_and_test_batches_scaled_with_train_statistics():
    ds = _make_dataset(window=8, horizon=4)
    dm = _make_dm(ds, scalers={'target': StandardScaler(axis=0)},
                  splitter=TemporalSplitter(0.1, 0.2))
    scaler = ds.scalers['target']
    target = ds.numpy()
    for split in ('val', 'test'):
        loader = dm.get_dataloader(split, shuffle=False)
        batch = next(iter(loader))
        first = getattr(dm, f'{split}set').indices[0]
        w_steps = ds.get_window_indices(torch.tensor(first)).numpy()
        # the input is the raw window normalized with the train-fit scaler
        expected = scaler.transform(torch.as_tensor(target[w_steps]))
        np.testing.assert_allclose(batch.x[0].numpy(), expected.numpy(),
                                   atol=1e-4)


def test_mask_scaling_uses_only_valid_train_values():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(300, 3, 1)).astype('float32')
    mask = rng.random((300, 3, 1)) > 0.3  # valid spread over time, no dead node
    ds = SpatioTemporalDataset(target=target, mask=mask, window=8, horizon=4)
    dm = _make_dm(ds, scalers={'target': StandardScaler(axis=0)},
                  splitter=TemporalSplitter(0.1, 0.2), mask_scaling=True)
    bias = ds.scalers['target'].bias.numpy()
    ts = dm.train_slice.numpy()
    ref = np.nanmean(np.where(mask[ts], target[ts], np.nan), axis=0)
    np.testing.assert_allclose(bias.ravel(), ref.ravel(), atol=1e-3)


# ===========================================================================
# Batching / collation correctness.
# ===========================================================================

def test_batch_shapes_and_size():
    ds = _make_dataset(n_nodes=3, n_channels=2, window=8, horizon=4)
    dm = _make_dm(ds, splitter=TemporalSplitter(0.1, 0.2), batch_size=16)
    batch = next(iter(dm.val_dataloader()))
    # [batch, time, nodes, features]
    assert tuple(batch.x.shape) == (16, 8, 3, 2)
    assert tuple(batch.y.shape) == (16, 4, 3, 2)


def test_static_connectivity_shared_in_batches():
    ds = _make_dataset(n_nodes=3, window=8, horizon=4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.tensor([0.1, 0.2, 0.3])
    ds.set_connectivity((edge_index, edge_weight))
    dm = _make_dm(ds, splitter=TemporalSplitter(0.1, 0.2), batch_size=16)
    batch = next(iter(dm.train_dataloader(shuffle=False)))
    # connectivity is static: one shared [2, E] edge_index, not batched per-item
    np.testing.assert_array_equal(batch.edge_index.numpy(), edge_index.numpy())
    np.testing.assert_array_equal(batch.edge_weight.numpy(), edge_weight.numpy())


def test_train_drops_last_partial_batch_val_test_keep_all():
    ds = _make_dataset(window=8, horizon=4)
    dm = _make_dm(ds, splitter=TemporalSplitter(0.1, 0.2), batch_size=16)
    # train drops the last incomplete batch -> total seen is a multiple of bs
    train_seen = sum(b.x.shape[0] for b in dm.train_dataloader(shuffle=False))
    assert train_seen <= dm.train_len and train_seen % 16 == 0
    # val/test keep every sample
    assert sum(b.x.shape[0] for b in dm.val_dataloader()) == dm.val_len
    assert sum(b.x.shape[0] for b in dm.test_dataloader()) == dm.test_len


def test_no_splitter_only_full_loader():
    ds = _make_dataset(window=8, horizon=4)
    dm = _make_dm(ds, batch_size=16)  # no splitter
    assert dm.train_dataloader() is None
    assert dm.val_dataloader() is None
    assert dm.test_dataloader() is None
    # the unsplit loader covers every sample in the dataset
    full = dm.get_dataloader()
    assert sum(b.x.shape[0] for b in full) == ds.n_samples
