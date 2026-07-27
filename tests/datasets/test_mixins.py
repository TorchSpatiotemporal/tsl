"""Prototype mixins: ``MissingValuesMixin`` (eval/training mask bookkeeping used
by imputation datasets) and ``TabularParsingMixin`` edge cases.

The ``MissingValuesMixin`` checks are leakage-relevant: the *training* mask must
never include cells reserved for *evaluation*.
"""
import numpy as np
import pandas as pd
import pytest

from tsl.datasets.prototypes import TabularDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin

from .helpers import grid_array, grid_dataframe, make_tabular

N_STEPS, N_NODES, N_CHANNELS = 10, 3, 1


class MissingValuesDataset(TabularDataset, MissingValuesMixin):
    """Tabular dataset that also carries an evaluation mask."""


def _make(mask=None):
    return MissingValuesDataset(target=grid_array(N_STEPS, N_NODES, N_CHANNELS),
                                mask=mask)


# -- MissingValuesMixin -----------------------------------------------------


def test_set_eval_mask_without_mask():
    ds = _make()
    eval_mask = np.zeros((N_STEPS, N_NODES, N_CHANNELS), dtype=bool)
    eval_mask[0] = True
    ds.set_eval_mask(eval_mask)
    np.testing.assert_array_equal(ds.eval_mask, eval_mask)


def test_set_eval_mask_intersects_with_mask():
    mask = np.ones((N_STEPS, N_NODES, N_CHANNELS), dtype=bool)
    mask[0, 0, 0] = False  # this cell is not observed
    ds = _make(mask=mask)
    eval_mask = np.ones((N_STEPS, N_NODES, N_CHANNELS), dtype=bool)
    ds.set_eval_mask(eval_mask)
    # eval can only point at observed cells -> eval_mask subset of mask
    assert not bool(ds.eval_mask[0, 0, 0])
    assert np.all(ds.eval_mask <= ds.mask)


def test_training_mask_excludes_eval():
    mask = np.ones((N_STEPS, N_NODES, N_CHANNELS), dtype=bool)
    mask[1, 1, 0] = False
    ds = _make(mask=mask)
    eval_mask = np.zeros((N_STEPS, N_NODES, N_CHANNELS), dtype=bool)
    eval_mask[0] = True
    ds.set_eval_mask(eval_mask)
    expected = (~ds.eval_mask) & mask
    np.testing.assert_array_equal(ds.training_mask, expected)
    # leakage invariant: training and eval cells never overlap
    assert not np.any(ds.training_mask & ds.eval_mask)


def test_training_mask_is_mask_without_eval():
    mask = np.ones((N_STEPS, N_NODES, N_CHANNELS), dtype=bool)
    ds = _make(mask=mask)
    np.testing.assert_array_equal(ds.training_mask, mask)


# -- TabularParsingMixin._value_to_kwargs -----------------------------------


def test_value_to_kwargs_array():
    ds = make_tabular()
    arr = grid_array(N_STEPS, N_NODES, 1)
    out = ds._value_to_kwargs(arr)
    assert set(out) == {'value'}
    assert out['value'] is arr


def test_value_to_kwargs_dataframe():
    ds = make_tabular()
    df = grid_dataframe(N_STEPS, N_NODES, 1)
    out = ds._value_to_kwargs(df)
    assert set(out) == {'value'}


def test_value_to_kwargs_list_and_tuple():
    ds = make_tabular()
    val = grid_array(N_STEPS, N_NODES, 1)
    out = ds._value_to_kwargs([val, 't n f'])
    assert out['pattern'] == 't n f'
    assert ds._value_to_kwargs((val, 't n f'))['pattern'] == 't n f'


def test_value_to_kwargs_mapping():
    ds = make_tabular()
    val = grid_array(N_STEPS, N_NODES, 1)
    out = ds._value_to_kwargs({'value': val, 'pattern': 't n f'})
    assert out['pattern'] == 't n f'


def test_value_to_kwargs_bad_type_raises():
    ds = make_tabular()
    with pytest.raises(TypeError):
        ds._value_to_kwargs(42)


# -- TabularParsingMixin node-subset check ----------------------------------


def test_add_covariate_with_unknown_nodes_raises():
    df = grid_dataframe(N_STEPS, N_NODES, 1)  # nodes 0,1,2
    ds = TabularDataset(target=df)
    # covariate referencing a node (99) absent from the dataset
    cols = pd.MultiIndex.from_product([[0, 1, 99], [0]],
                                      names=['nodes', 'channels'])
    cov = pd.DataFrame(np.ones((N_STEPS, N_NODES)), index=df.index,
                       columns=cols)
    with pytest.raises(AssertionError):
        ds.add_covariate('u', cov, 't n f')
