import pickle

import numpy as np
import pytest

from tsl.data.datamodule.splitters import FixedIndicesSplitter, Splitter

from .helpers import _make_dataset


# -- base Splitter ----------------------------------------------------------

def test_base_fit_not_implemented():
    # The abstract base does not know how to split.
    with pytest.raises(NotImplementedError):
        Splitter().fit(_make_dataset(4, 2))


def test_fresh_splitter_is_unfitted():
    splitter = Splitter()
    assert splitter.fitted is False
    assert splitter.indices == dict(train=None, val=None, test=None)
    assert splitter.train_idxs is None
    assert splitter.val_idxs is None
    assert splitter.test_idxs is None
    assert splitter.train_len is None
    assert splitter.val_len is None
    assert splitter.test_len is None


def test_set_indices_and_lens():
    splitter = Splitter()
    splitter.set_indices(train=np.arange(5), val=np.arange(2), test=np.arange(3))
    assert splitter.train_len == 5
    assert splitter.val_len == 2
    assert splitter.test_len == 3
    assert splitter.lens() == dict(train_len=5, val_len=2, test_len=3)


def test_set_indices_partial_leaves_others_untouched():
    splitter = Splitter()
    splitter.set_indices(val=np.arange(4))
    assert splitter.val_len == 4
    assert splitter.train_idxs is None
    assert splitter.test_idxs is None


def test_track_fit_sets_fitted_and_caches_indices():
    # A subclass whose ``fit`` populates the indices: ``_track_fit`` must flip
    # ``fitted`` to True and ``split`` must then return the cached dict.
    class _Const(Splitter):

        def fit(self, dataset):
            self.set_indices(np.arange(3), np.arange(2), np.arange(1))

    splitter = _Const()
    assert splitter.fitted is False
    out = splitter.fit(_make_dataset(4, 2))
    assert splitter.fitted is True
    # fit returns the indices dict
    assert out is splitter.indices
    # once fitted, split short-circuits to the cached indices (same object)
    assert splitter.split(_make_dataset(4, 2)) is splitter.indices


def test_split_triggers_fit_when_unfitted():
    # A fresh splitter is not fitted, so ``split`` must run ``fit`` (rather than
    # return the empty cached indices).
    class _Const(Splitter):

        def fit(self, dataset):
            self.set_indices(np.arange(3), np.arange(2), np.arange(1))

    splitter = _Const()
    out = splitter.split(_make_dataset(4, 2))
    assert splitter.fitted is True
    assert out is splitter.indices
    assert splitter.train_len == 3 and splitter.val_len == 2


def test_reset_clears_state():
    splitter = Splitter()
    splitter.set_indices(np.arange(5), np.arange(2), np.arange(3))
    splitter._fitted = True
    splitter.reset()
    assert splitter.fitted is False
    assert splitter.indices == dict(train=None, val=None, test=None)


def test_copy_is_deep_and_independent():
    splitter = Splitter()
    splitter.set_indices(np.arange(5), np.arange(2), np.arange(3))
    copy = splitter.copy()
    assert isinstance(copy, Splitter)
    assert copy.train_len == 5 and copy.val_len == 2 and copy.test_len == 3
    # mutating the original must not affect the copy
    splitter.set_indices(train=np.arange(99))
    assert copy.train_len == 5


def test_copy_downcasts_subclass_to_base_but_keeps_indices():
    # ``copy`` always builds a base ``Splitter`` (it hardcodes ``Splitter()``),
    # so a subclass loses its type while its fitted indices are preserved.
    fixed = FixedIndicesSplitter(train_idxs=[0, 1], val_idxs=[2], test_idxs=[3])
    copy = fixed.copy()
    assert type(copy) is Splitter
    assert copy.fitted is True
    assert copy.train_idxs == [0, 1] and copy.test_idxs == [3]


def test_repr_contains_class_name_and_lens():
    splitter = FixedIndicesSplitter(train_idxs=[0, 1, 2],
                                    val_idxs=[3, 4],
                                    test_idxs=[5, 6, 7])
    text = repr(splitter)
    assert text.startswith('FixedIndicesSplitter(')
    assert 'train_len=3' in text
    assert 'val_len=2' in text
    assert 'test_len=3' in text


def test_call_dispatches_to_split():
    dataset = _make_dataset(4, 2)
    splitter = FixedIndicesSplitter(train_idxs=[0, 1], val_idxs=[2], test_idxs=[3])
    assert splitter(dataset) == splitter.split(dataset)


def test_fitted_splitter_pickles():
    # ``__getstate__`` drops the bound ``fit`` so the splitter can be pickled.
    splitter = FixedIndicesSplitter(train_idxs=[0, 1, 2],
                                    val_idxs=[3, 4],
                                    test_idxs=[5])
    restored = pickle.loads(pickle.dumps(splitter))
    assert restored.fitted is True
    assert restored.lens() == splitter.lens()


# -- FixedIndicesSplitter ---------------------------------------------------

def test_fixed_indices_is_fitted_on_construction():
    splitter = FixedIndicesSplitter(train_idxs=[0, 1], val_idxs=[2], test_idxs=[3])
    assert splitter.fitted is True


def test_fixed_indices_returns_given_indices():
    dataset = _make_dataset(4, 2)
    splitter = FixedIndicesSplitter(train_idxs=[0, 1, 2],
                                    val_idxs=[3, 4],
                                    test_idxs=[5, 6, 7])
    out = splitter.split(dataset)
    assert out['train'] == [0, 1, 2]
    assert out['val'] == [3, 4]
    assert out['test'] == [5, 6, 7]


def test_fixed_indices_fit_is_noop():
    dataset = _make_dataset(4, 2)
    splitter = FixedIndicesSplitter(train_idxs=[0, 1], val_idxs=[2], test_idxs=[3])
    before = dict(splitter.indices)
    splitter.fit(dataset)
    assert splitter.indices == before


def test_fixed_indices_partial_args():
    splitter = FixedIndicesSplitter(train_idxs=[0, 1, 2])
    assert splitter.train_idxs == [0, 1, 2]
    assert splitter.val_idxs is None
    assert splitter.test_idxs is None
