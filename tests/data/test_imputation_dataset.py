"""Unit tests for :class:`tsl.data.ImputationDataset`.

The defining feature of an imputation dataset is that ``window == horizon`` and
``delay == -window``, so a sample's input window and target horizon cover the
**same** time steps. The only thing stopping the model from reading the values
it must impute straight off the input is the input ``mask``: it must be ``False``
wherever ``eval_mask`` is ``True``. These tests center that no-leak invariant.
"""
import numpy as np
import pytest
import torch

from tsl.data import ImputationDataset
from tsl.data.synch_mode import HORIZON, WINDOW


# -- builders ---------------------------------------------------------------

def _target(t=12, n=3, f=1):
    return np.arange(t * n * f, dtype='float32').reshape(t, n, f)


def _eval_mask(t=12, n=3, f=1, cells=((5, 1, 0), (7, 2, 0), (9, 0, 0))):
    m = np.zeros((t, n, f), dtype=bool)
    for c in cells:
        m[c] = True
    return m


def _make(target=None, eval_mask=None, mask=None, window=4, **kwargs):
    target = _target() if target is None else target
    eval_mask = _eval_mask() if eval_mask is None else eval_mask
    return ImputationDataset(target=target, eval_mask=eval_mask, mask=mask,
                             window=window, **kwargs)


# -- window configuration ---------------------------------------------------

def test_imputation_window_equals_horizon():
    ds = _make(window=4)
    assert ds.window == ds.horizon == 4
    assert ds.delay == -4
    # horizon aligns with the window: same steps are input and target
    assert ds.horizon_offset == 0
    exp = ds.expand_indices(np.array([0]))
    assert exp['window'].numpy().ravel().tolist() == \
        exp['horizon'].numpy().ravel().tolist()


# -- LEAK GUARD: eval values are hidden from the input ----------------------

def test_eval_cells_never_valid_in_input_mask():
    # The core invariant: no cell flagged for evaluation is marked valid in the
    # input mask -> the model can never see a value it is asked to impute.
    ds = _make()
    assert not bool((ds.mask & ds.eval_mask).any())


def test_default_mask_excludes_eval_and_nans():
    target = _target()
    target[0, 0, 0] = np.nan  # invalid value
    eval_mask = _eval_mask()
    ds = _make(target=target, eval_mask=eval_mask)
    # reference: valid iff not NaN and not an eval cell
    ref = ~np.isnan(target) & ~eval_mask
    assert torch.equal(ds.mask, torch.as_tensor(ref))
    # NaN cell is masked out
    assert not bool(ds.mask[0, 0, 0])


def test_explicit_mask_is_intersected_with_eval():
    target = _target()
    eval_mask = _eval_mask()
    # a user mask that (wrongly) marks an eval cell valid must still be cleared
    user_mask = np.ones_like(eval_mask, dtype=bool)
    user_mask[2, 0, 0] = False  # an extra invalid cell
    ds = _make(target=target, eval_mask=eval_mask, mask=user_mask)
    ref = ~eval_mask & user_mask
    assert torch.equal(ds.mask, torch.as_tensor(ref))
    assert not bool((ds.mask & ds.eval_mask).any())   # leak guard still holds
    assert not bool(ds.mask[2, 0, 0])                 # user invalid respected


def test_item_hides_eval_in_input_keeps_it_in_target():
    target = _target()
    eval_mask = _eval_mask(cells=((1, 1, 0),))  # eval cell inside sample 0's win
    ds = _make(target=target, eval_mask=eval_mask, window=4, stride=1)
    item = ds[0]
    # sample 0 covers absolute steps 0..3 in both input window and target horizon
    # input carries the mask, which is 0 at the eval cell
    assert 'mask' in item.input and 'x' in item.input
    assert bool(item.input.mask[1, 1, 0]) is False
    # the true value lives on the target side, available for the eval loss
    assert 'y' in item.target
    assert item.target.y[1, 1, 0] == target[1, 1, 0]
    # and eval_mask is exposed (as auxiliary), flagging the cell to score
    assert bool(item.eval_mask[1, 1, 0]) is True


# -- batch-map wiring -------------------------------------------------------

def test_eval_mask_is_auxiliary_horizon_not_input():
    ds = _make()
    # eval_mask must not be fed as model input
    assert 'eval_mask' not in ds.input_map
    assert 'eval_mask' in ds.auxiliary_map
    assert ds.auxiliary_map['eval_mask'].synch_mode is HORIZON
    # mask is part of the input, synchronised with the window
    assert 'mask' in ds.input_map
    assert ds.input_map['mask'].synch_mode is WINDOW


def test_reset_maps_restore_mask_and_eval_wiring():
    ds = _make(eval_mask=_eval_mask(cells=((1, 1, 0),)), window=4, stride=1)
    ds.reset_input_map()
    ds.reset_auxiliary_map()
    # the input mask (WINDOW) and the auxiliary eval_mask (HORIZON) are restored
    assert 'mask' in ds.input_map
    assert ds.input_map['mask'].synch_mode is WINDOW
    assert 'eval_mask' in ds.auxiliary_map
    assert ds.auxiliary_map['eval_mask'].synch_mode is HORIZON
    # eval_mask must NOT leak into the input after a reset
    assert 'eval_mask' not in ds.input_map
    assert 'eval_mask' not in ds[0].input
    # the value-leak guard still holds after a reset: eval values stay hidden
    assert not bool((ds.mask & ds.eval_mask).any())
    assert bool(ds[0].input.mask[1, 1, 0]) is False


def test_set_mask_strips_eval_cells_even_if_caller_forgets():
    ds = _make()
    # pass a genuinely dirty mask (all-ones) with no pre-anding: set_mask
    # itself must strip the eval cells before storing.
    dirty = torch.ones_like(ds.mask, dtype=torch.bool)
    assert bool((dirty & ds.eval_mask).any())  # sanity: eval cells are IN
    ds.set_mask(dirty, add_to_input_map=False)
    assert not bool((ds.mask & ds.eval_mask).any())
