"""Group 4 -- Mask."""
import numpy as np
import pytest
import torch

from tsl.data import SpatioTemporalDataset

from .helpers import (_grid_target, _make_grid_dataset, ref_horizon_steps,
                      ref_window_steps)


def test_mask_accepts_bool_and_uint8():
    target = _grid_target(20, 3, 2)
    m = (np.arange(20 * 3 * 2).reshape(20, 3, 2) % 2)
    ds_bool = SpatioTemporalDataset(target=target, mask=m.astype(bool))
    ds_uint = SpatioTemporalDataset(target=target, mask=m.astype('uint8'))
    assert ds_bool.has_mask and ds_uint.has_mask
    assert ds_bool.mask.dtype == torch.bool
    assert ds_uint.mask.dtype == torch.uint8


def test_mask_float_raises():
    target = _grid_target(20, 3, 2)
    with pytest.raises(RuntimeError):
        SpatioTemporalDataset(target=target,
                              mask=np.ones((20, 3, 2), dtype='float32'))


def test_mask_wrong_temporal_dim_raises():
    target = _grid_target(20, 3, 2)
    with pytest.raises(ValueError):
        SpatioTemporalDataset(target=target,
                              mask=np.ones((21, 3, 2), dtype=bool))


def test_mask_wrong_node_dim_raises():
    # a non-broadcastable node dimension (5 vs 3) must be rejected
    target = _grid_target(20, 3, 2)
    with pytest.raises(ValueError):
        SpatioTemporalDataset(target=target,
                              mask=np.ones((20, 5, 2), dtype=bool))


def test_mask_broadcasts_over_channels():
    target = _grid_target(20, 3, 2)
    ds = SpatioTemporalDataset(target=target,
                               mask=np.ones((20, 3, 1), dtype=bool),
                               window=4, horizon=2)
    assert tuple(ds.mask.shape) == (20, 3, 1)
    # the broadcast (single-channel) mask is delivered in the item, horizon-
    # aligned, keeping its single channel
    item = ds[0]
    assert tuple(item.mask.shape) == (2, 3, 1)  # horizon=2 steps


def test_get_mask_from_nans():
    target = _grid_target(10, 2, 2)
    target[0, 0, 0] = np.nan
    target[3, 1, 1] = np.nan
    ds = SpatioTemporalDataset(target=target, window=2, horizon=2)
    assert not ds.has_mask
    mask = ds.get_mask()
    expected = ~np.isnan(target)
    np.testing.assert_array_equal(mask.numpy(), expected)
    assert mask.dtype == torch.bool
    # dtype aliases all resolve to the right torch dtype
    assert ds.get_mask(dtype='uint8').dtype == torch.uint8
    assert ds.get_mask(dtype='bool').dtype == torch.bool


def test_get_mask_invalid_dtype_raises():
    ds, _ = _make_grid_dataset(window=2, horizon=2)
    with pytest.raises(AssertionError):
        ds.get_mask(dtype='float32')


def test_mask_aligned_to_horizon_not_window():
    # The mask validates predictions, so in an item it must be synchronized to
    # the HORIZON steps, never the window -- a window-aligned mask would leak
    # which target steps are observed.
    target = _grid_target(40, 3, 2)
    mask = (np.arange(40 * 3 * 2).reshape(40, 3, 2) % 3 == 0)
    ds = SpatioTemporalDataset(target=target, mask=mask, window=4, horizon=3,
                               delay=1)
    item = ds[0]
    h_steps = ref_horizon_steps(0, 4, 3, 1, 1)  # [5, 6, 7]
    np.testing.assert_array_equal(item.mask.numpy(), mask[h_steps])
    assert tuple(item.mask.shape) == (len(h_steps), 3, 2)
    # and explicitly NOT aligned to the window steps
    w_steps = ref_window_steps(0, 4, 1)  # [0, 1, 2, 3]
    assert not np.array_equal(item.mask.numpy(), mask[w_steps])
