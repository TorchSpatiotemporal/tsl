"""Unit tests for :class:`tsl.data.SpatioTemporalDataModule`."""
import numpy as np
import pytest
import torch
from torch.utils.data import (RandomSampler, SequentialSampler, Subset)

from tsl.data import (SpatioTemporalDataModule, SpatioTemporalDataset,
                      TemporalSplitter)
from tsl.data.loader import StaticGraphLoader
from tsl.data.preprocessing.scalers import StandardScaler


# -- builders ---------------------------------------------------------------

def _ramp_dataset(n_steps=300, n_nodes=3, window=8, horizon=4, mask=None):
    # value = time + node, so per-node stats grow with time -> train != full
    tgt = (np.arange(n_steps)[:, None, None] * np.ones((1, n_nodes, 1))
           + np.arange(n_nodes)[None, :, None]).astype('float32')
    return SpatioTemporalDataset(target=tgt, mask=mask, window=window,
                                 horizon=horizon)


def _dm(dataset, **kwargs):
    kwargs.setdefault('splitter', TemporalSplitter(val_len=0.1, test_len=0.2))
    kwargs.setdefault('batch_size', 16)
    return SpatioTemporalDataModule(dataset=dataset, **kwargs)


# ===========================================================================
# LEAK GUARD: scalers are fit on the training slice only
# ===========================================================================

def test_scaler_fit_on_train_slice_not_full_series():
    ds = _ramp_dataset()
    dm = _dm(ds, scalers={'target': StandardScaler(axis=0)})
    dm.setup()
    scaler = ds.scalers['target']
    target = ds.numpy()
    sl = dm.train_slice.numpy()
    train_mean = target[sl].mean(0, keepdims=True)
    train_std = target[sl].std(0, keepdims=True)
    full_mean = target.mean(0, keepdims=True)
    # fit exactly on the train slice ...
    assert np.allclose(np.asarray(scaler.bias).ravel(), train_mean.ravel())
    assert np.allclose(np.asarray(scaler.scale).ravel(), train_std.ravel())
    # ... and NOT on the whole series (this is the no-leak signal)
    assert not np.allclose(np.asarray(scaler.bias).ravel(), full_mean.ravel())


def test_train_slice_is_exactly_train_footprint():
    ds = _ramp_dataset()
    dm = _dm(ds)
    dm.setup()
    expected = ds.expand_indices(np.asarray(dm.trainset.indices), merge=True)
    assert np.array_equal(np.sort(dm.train_slice.numpy()),
                          np.sort(expected.numpy()))


def test_mask_scaling_uses_only_valid_train_values():
    mask = np.ones((300, 3, 1), dtype=bool)
    mask[::5] = False  # drop some steps
    ds = _ramp_dataset(mask=mask)
    dm = _dm(ds, scalers={'target': StandardScaler(axis=0)}, mask_scaling=True)
    dm.setup()
    scaler = ds.scalers['target']
    target, sl = ds.numpy(), dm.train_slice.numpy()
    ref = np.nanmean(np.where(mask[sl], target[sl], np.nan), axis=0)
    assert np.allclose(np.asarray(scaler.bias).ravel(), ref.ravel())


@pytest.mark.parametrize('stage', ['validate', 'test', 'predict'])
def test_non_fit_stages_do_not_refit_scaler(stage):
    ds = _ramp_dataset()
    preset = StandardScaler(axis=0)
    preset.fit(torch.as_tensor(ds.numpy()), keepdims=True)  # fit on all data
    bias_before = np.asarray(preset.bias).copy()
    dm = _dm(ds, scalers={'target': preset})
    dm.setup(stage=stage)
    # validate/test/predict must keep the provided scaler untouched (no refit)
    assert np.allclose(np.asarray(ds.scalers['target'].bias), bias_before)


@pytest.mark.parametrize('stage', [None, 'fit'])
def test_fit_stages_refit_scaler_on_train_slice(stage):
    ds = _ramp_dataset()
    preset = StandardScaler(axis=0)
    preset.fit(torch.as_tensor(ds.numpy()), keepdims=True)  # fit on all data
    bias_before = np.asarray(preset.bias).copy()
    dm = _dm(ds, scalers={'target': preset})
    dm.setup(stage=stage)
    # fit (and manual None) must refit on the train slice, overwriting the
    # preset stats
    train_mean = ds.numpy()[dm.train_slice.numpy()].mean(0, keepdims=True)
    assert not np.allclose(np.asarray(ds.scalers['target'].bias).ravel(),
                           bias_before.ravel())
    assert np.allclose(np.asarray(ds.scalers['target'].bias).ravel(),
                       train_mean.ravel())


def test_missing_scaler_key_raises():
    ds = _ramp_dataset()
    dm = _dm(ds, scalers={'nonexistent': StandardScaler()})
    with pytest.raises(RuntimeError):
        dm.setup()


def test_scaler_without_splitter_raises_on_fit_stage():
    # scalers require a training slice; without a splitter there is none, so
    # setup() on a fit stage must fail loud instead of silently misbehaving.
    ds = _ramp_dataset()
    dm = SpatioTemporalDataModule(dataset=ds,
                                  scalers={'target': StandardScaler(axis=0)},
                                  batch_size=8)
    with pytest.raises(RuntimeError, match='training slice'):
        dm.setup()


def test_prefit_scaler_without_splitter_works_on_non_fit_stages():
    # external-scaler workflow: user fits a scaler themselves and hands it in;
    # setup(stage=predict/validate/test) must accept it even with no splitter.
    ds = _ramp_dataset()
    preset = StandardScaler(axis=0)
    preset.fit(torch.as_tensor(ds.numpy()), keepdims=True)
    bias_before = np.asarray(preset.bias).copy()
    dm = SpatioTemporalDataModule(dataset=ds,
                                  scalers={'target': preset},
                                  batch_size=8)
    dm.setup(stage='predict')
    assert np.allclose(np.asarray(ds.scalers['target'].bias), bias_before)


def test_covariate_scalers_slice_by_pattern():
    # temporal covariate (pattern 't n f') must be sliced on train indices;
    # static covariate (pattern 'n f') must be fit on the full tensor.
    ds = _ramp_dataset()
    n_steps, n_nodes = 300, 3
    u_dyn = (np.arange(n_steps)[:, None, None] * np.ones((1, n_nodes, 1))
             ).astype('float32')  # ramp again -> full != train
    u_stat = np.arange(n_nodes, dtype='float32').reshape(n_nodes, 1) * 10.0
    ds.add_covariate('u_dyn', u_dyn, pattern='t n f')
    ds.add_covariate('u_stat', u_stat, pattern='n f')

    dm = _dm(ds,
             scalers={'u_dyn': StandardScaler(axis=0),
                      'u_stat': StandardScaler(axis=0)})
    dm.setup()

    sl = dm.train_slice.numpy()
    # temporal covariate: fit on the train slice only
    dyn_train_mean = u_dyn[sl].mean(0, keepdims=True)
    assert np.allclose(np.asarray(ds.scalers['u_dyn'].bias).ravel(),
                       dyn_train_mean.ravel())
    assert not np.allclose(np.asarray(ds.scalers['u_dyn'].bias).ravel(),
                           u_dyn.mean(0, keepdims=True).ravel())
    # static covariate: fit on the whole tensor (no accidental t-slicing)
    stat_mean = u_stat.mean(0, keepdims=True)
    assert np.allclose(np.asarray(ds.scalers['u_stat'].bias).ravel(),
                       stat_mean.ravel())


# ===========================================================================
# LEAK GUARD: the train loader only draws training samples
# ===========================================================================

def test_train_loader_only_sees_train_samples():
    ds = _ramp_dataset()
    dm = _dm(ds)
    dm.setup()
    loader = dm.train_dataloader()
    # the loader wraps the train Subset, whose indices are exactly the
    # splitter's train indices -> val/test samples can never be drawn
    assert loader.dataset is dm.trainset
    assert list(dm.trainset.indices) == list(dm.splitter.train_idxs)
    assert set(dm.trainset.indices).isdisjoint(set(dm.valset.indices))
    assert set(dm.trainset.indices).isdisjoint(set(dm.testset.indices))


# ===========================================================================
# Module mechanics
# ===========================================================================

def test_sets_and_slices_none_before_setup():
    dm = _dm(_ramp_dataset())
    assert dm.train_len is None and dm.val_len is None and dm.test_len is None
    assert dm.train_slice is None
    assert dm.val_slice is None and dm.test_slice is None


def test_setup_populates_sets_and_lens():
    ds = _ramp_dataset()
    dm = _dm(ds)
    dm.setup()
    assert isinstance(dm.trainset, Subset)
    assert dm.train_len == len(dm.trainset)
    assert dm.train_len and dm.val_len and dm.test_len


def test_add_set_accepts_dataset_object():
    ds = _ramp_dataset()
    dm = SpatioTemporalDataModule(dataset=ds, batch_size=8)
    sub = Subset(ds, [0, 1, 2])
    dm.trainset = sub
    assert dm.trainset is sub
    assert dm.train_len == 3


def test_add_set_rejects_invalid_type():
    dm = SpatioTemporalDataModule(dataset=_ramp_dataset())
    with pytest.raises(AssertionError):
        dm.trainset = object()


def test_get_dataloader_routing():
    ds = _ramp_dataset()
    dm = _dm(ds)
    dm.setup()
    assert isinstance(dm.get_dataloader('train'), StaticGraphLoader)
    # split=None -> a loader over the whole dataset
    whole = dm.get_dataloader(None)
    assert whole.dataset is ds
    with pytest.raises(ValueError):
        dm.get_dataloader('bogus')


def test_get_dataloader_none_when_split_empty():
    dm = SpatioTemporalDataModule(dataset=_ramp_dataset(), batch_size=8)
    # no setup/splitter -> trainset is None -> no loader
    assert dm.get_dataloader('train') is None


def test_train_loader_drop_last_only_for_train():
    ds = _ramp_dataset()
    dm = _dm(ds)
    dm.setup()
    assert dm.train_dataloader().drop_last is True
    assert dm.val_dataloader().drop_last is False
    assert dm.test_dataloader().drop_last is False


# ===========================================================================
# Loader-option forwarding
# ===========================================================================

def test_workers_forwarded_to_all_loaders():
    ds = _ramp_dataset()
    dm = _dm(ds, workers=3)
    dm.setup()
    assert dm.train_dataloader().num_workers == 3
    assert dm.val_dataloader().num_workers == 3
    assert dm.test_dataloader().num_workers == 3


def test_pin_memory_only_on_train_loader():
    ds = _ramp_dataset()
    dm = _dm(ds, pin_memory=True)
    dm.setup()
    # asymmetric on purpose: train pins, val/test/full do not
    assert dm.train_dataloader().pin_memory is True
    assert not dm.val_dataloader().pin_memory
    assert not dm.test_dataloader().pin_memory
    assert not dm.get_dataloader(None).pin_memory


def test_batch_size_default_and_per_call_override():
    ds = _ramp_dataset()
    dm = _dm(ds, batch_size=16)
    dm.setup()
    # defaults to the module batch size
    assert dm.train_dataloader().batch_size == 16
    assert dm.val_dataloader().batch_size == 16
    assert dm.test_dataloader().batch_size == 16
    # per-call override wins
    assert dm.train_dataloader(batch_size=4).batch_size == 4
    assert dm.val_dataloader(batch_size=7).batch_size == 7
    assert dm.test_dataloader(batch_size=9).batch_size == 9


def test_shuffle_argument_controls_sampler():
    ds = _ramp_dataset()
    dm = _dm(ds)
    dm.setup()
    # defaults: train shuffles, val/test do not
    assert isinstance(dm.train_dataloader().sampler, RandomSampler)
    assert isinstance(dm.val_dataloader().sampler, SequentialSampler)
    assert isinstance(dm.test_dataloader().sampler, SequentialSampler)
    # per-call override propagates in both directions
    assert isinstance(dm.train_dataloader(shuffle=False).sampler,
                      SequentialSampler)
    assert isinstance(dm.val_dataloader(shuffle=True).sampler, RandomSampler)


def test_getattr_delegates_to_dataset():
    ds = _ramp_dataset(window=8)
    dm = _dm(ds)
    # attributes not on the datamodule fall through to the wrapped dataset
    assert dm.window == ds.window
    assert dm.n_nodes == ds.n_nodes


def test_repr_contains_lens_and_batch_size():
    ds = _ramp_dataset()
    dm = _dm(ds, batch_size=16)
    dm.setup()
    text = repr(dm)
    assert 'SpatioTemporalDataModule' in text
    assert 'batch_size=16' in text
    assert f'train_len={dm.train_len}' in text
