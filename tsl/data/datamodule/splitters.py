import functools
from copy import deepcopy
from datetime import datetime
from typing import Callable, Mapping, Optional, Tuple, Union

import numpy as np
from pandas import DatetimeIndex

from tsl.utils.python_utils import ensure_list

from ..spatiotemporal_dataset import SpatioTemporalDataset
from ..synch_mode import SynchMode

__all__ = [
    'Splitter',
    'CustomSplitter',
    'TemporalSplitter',
    'AtTimeStepSplitter',
]

from ...typing import Index


class Splitter:
    r"""Base class for splitter module."""

    def __init__(self):
        self.__indices = dict()
        self._fitted = False
        self.reset()

    def __new__(cls, *args, **kwargs) -> "Splitter":
        obj = super().__new__(cls)

        # track `fit` calls
        obj.fit = cls._track_fit(obj, obj.fit)

        return obj

    @staticmethod
    def _track_fit(obj: "Splitter", fn: callable) -> callable:
        """A decorator to track fit calls.

        When ``splitter.fit(...)`` is called, :obj:`splitter.fitted` is set to
        :obj:`True`.

        Args:
            obj: Object whose function will be tracked.
            fn: Function that will be wrapped.

        Returns:
            Decorated method to track :obj:`fit` calls.
        """

        @functools.wraps(fn)
        def fit(dataset: SpatioTemporalDataset) -> dict:
            fn(dataset)
            obj._fitted = True
            return obj.indices

        return fit

    def __getstate__(self) -> dict:
        # avoids _pickle.PicklingError: Can't pickle <...>: it's not the same
        # object as <...>
        d = self.__dict__.copy()
        del d['fit']
        return d

    def __call__(self, *args, **kwargs):
        return self.split(*args, **kwargs)

    def __repr__(self):
        lens = ", ".join(map(lambda kv: "%s=%s" % kv, self.lens().items()))
        return "%s(%s)" % (self.__class__.__name__, lens)

    @property
    def indices(self):
        return self.__indices

    @property
    def fitted(self):
        return self._fitted

    @property
    def train_idxs(self):
        return self.__indices.get('train')

    @property
    def val_idxs(self):
        return self.__indices.get('val')

    @property
    def test_idxs(self):
        return self.__indices.get('test')

    @property
    def train_len(self):
        return len(self.train_idxs) if self.train_idxs is not None else None

    @property
    def val_len(self):
        return len(self.val_idxs) if self.val_idxs is not None else None

    @property
    def test_len(self):
        return len(self.test_idxs) if self.test_idxs is not None else None

    def set_indices(self, train=None, val=None, test=None):
        if train is not None:
            self.__indices['train'] = train
        if val is not None:
            self.__indices['val'] = val
        if test is not None:
            self.__indices['test'] = test

    def reset(self):
        self.__indices = dict(train=None, val=None, test=None)
        self._fitted = False

    def lens(self) -> dict:
        return dict(train_len=self.train_len,
                    val_len=self.val_len,
                    test_len=self.test_len)

    def copy(self) -> "Splitter":
        copy = Splitter()
        copy.__dict__ = deepcopy(self.__dict__)
        return copy

    def fit(self, dataset: SpatioTemporalDataset):
        raise NotImplementedError

    def split(self, dataset: SpatioTemporalDataset) -> dict:
        if self.fitted:
            return self.indices
        else:
            return self.fit(dataset)


class CustomSplitter(Splitter):
    r"""Create a :class:`~tsl.data.datamodule.splitters.Splitter` using custom
    validation and test sets splitting functions."""

    def __init__(self,
                 val_split_fn: Callable = None,
                 test_split_fn: Callable = None,
                 val_kwargs: Mapping = None,
                 test_kwargs: Mapping = None,
                 mask_test_indices_in_val: bool = True):
        super(CustomSplitter, self).__init__()
        self.val_split_fn = val_split_fn
        self.test_split_fn = test_split_fn
        self.val_kwargs = val_kwargs or dict()
        self.test_kwargs = test_kwargs or dict()
        self.mask_test_indices_in_val = mask_test_indices_in_val

    @property
    def val_policy(self):
        return self.val_split_fn.__name__ if callable(
            self.val_split_fn) else None

    @property
    def test_policy(self):
        return self.test_split_fn.__name__ if callable(
            self.test_split_fn) else None

    def fit(self, dataset: SpatioTemporalDataset):
        _, test_idxs = self.test_split_fn(dataset, **self.test_kwargs)
        val_kwargs = self.val_kwargs
        if self.mask_test_indices_in_val and len(test_idxs):
            val_kwargs = dict(**self.val_kwargs, mask=test_idxs)
        train_idxs, val_idxs = self.val_split_fn(dataset, **val_kwargs)
        self.set_indices(train_idxs, val_idxs, test_idxs)


class FixedIndicesSplitter(Splitter):
    r"""Create a :class:`~tsl.data.datamodule.splitters.Splitter` using fixed
    indices for training, validation and test sets."""

    def __init__(self,
                 train_idxs: Optional[Index] = None,
                 val_idxs: Optional[Index] = None,
                 test_idxs: Optional[Index] = None):
        super(FixedIndicesSplitter, self).__init__()
        self.set_indices(train_idxs, val_idxs, test_idxs)
        self._fitted = True

    def fit(self, dataset: SpatioTemporalDataset):
        pass


class TemporalSplitter(Splitter):
    r"""Split the data sequentially with specified lengths.

    Args:
        val_len (int or float): Length of the validation set.
        test_len (int or float): Length of the test set.
        offset (str): How to size the offset separating the splits so that samples
            do not leak across sets.

            - :obj:`'window'`: separate splits by :obj:`dataset.samples_offset` positions,
              so their lookback windows just touch. This avoids leakage (no target step shared
              across splits) as long as the horizon is short enough relative to the window.
            - :obj:`'sample'`: separate splits by :obj:`ceil(sample_span / stride)` positions,
              so that adjacent splits share no time step in any role, for any window/horizon/delay/stride.

            (default: :obj:`'window'`)
    """

    def __init__(self,
                 val_len: Union[int, float] = None,
                 test_len: Union[int, float] = None,
                 offset: str = 'window'):
        super(TemporalSplitter, self).__init__()
        self._val_len = val_len
        self._test_len = test_len
        self.offset = offset

    def fit(self, dataset: SpatioTemporalDataset):
        idx = np.arange(len(dataset))
        val_len, test_len = self._val_len, self._test_len
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len

        if self.offset == 'window':
            # Separate the closest train and val/test samples by
            # ``samples_offset`` positions, so their windows just touch.
            # This avoids leakage iff that separation also covers the horizon.
            offset = dataset.samples_offset - 1
            assert (offset + 1) * dataset.stride >= dataset.horizon, (
                f"offset='window' separates split horizons by "
                f"{(offset + 1) * dataset.stride} steps < horizon="
                f"{dataset.horizon} (window={dataset.window}, "
                f"stride={dataset.stride}): target steps would be shared across "
                f"splits.")
        elif self.offset == 'sample':
            # Separate the closest split samples by ``ceil(sample_span / stride)``
            # positions: splits share no time step in any role, for any window/horizon/delay/stride.
            offset = int(np.ceil(dataset.sample_span / dataset.stride)) - 1
        else:
            raise ValueError(f"Unknown offset '{self.offset}', must be "
                             "'window' or 'sample'.")

        self.set_indices(idx[:val_start - offset],
                         idx[val_start:test_start - offset],
                         idx[test_start:])


class AtTimeStepSplitter(Splitter):
    r"""Split the data at given time steps (only for
    :class:`~tsl.data.SpatioTemporalDataset` with
    :class:`~pandas.DatetimeIndex` index).

    Each split is defined by a (``first_ts``, ``last_ts``) timestamp range,
    following the chronological order ``train`` -> ``val`` -> ``test``. A split
    is active when at least one of its bounds is given (training is always
    active); the remaining bounds are then inferred:

    - A missing inner boundary is placed ``min_offset`` positions away from the
      adjacent split (e.g. an open-ended :obj:`last_val_ts` ends just before the
      test range, and a missing :obj:`first_test_ts` starts just after
      validation).
    - A missing outer boundary falls back to the edge of the series: training
      defaults to start at the beginning, and the latest split extends to the
      end.
    - A held-out (validation or test) split with no bounds at all is left empty.

    Splits are kept separated so they do not leak across each other. The
    separation is controlled by :obj:`min_offset`, using the same vocabulary as
    :class:`TemporalSplitter`:

    - :obj:`'sample'`: separate the closest splits by at least
      :obj:`ceil(sample_span / stride)` positions, so that they share no time
      step in any role (input window or prediction horizon), for any
      window/horizon/delay/stride.
    - :obj:`'window'`: separate the closest splits by at least
      :obj:`samples_offset` positions, so their lookback windows just touch.
      This avoids leakage (no target step shared across splits) as long as the
      horizon is short enough relative to the window, and raises otherwise.

    (default: :obj:`'sample'`)

    After resolving the ranges, the splits are checked and a
    :class:`ValueError` is raised if any two are closer than :obj:`min_offset`
    (e.g. when explicit boundaries would make the splits overlap).

    Args:
        first_val_ts, last_val_ts (optional): Bounds of the validation range.
        first_test_ts, last_test_ts (optional): Bounds of the test range.
        first_train_ts (optional): Start of the training range. Defaults to the
            beginning of the series.
        last_train_ts (optional): End of the training range. Defaults to the
            last position keeping training separated from the held-out splits.
        min_offset (str): Minimum separation between the closest splits, either
            :obj:`'sample'` or :obj:`'window'`. (default: :obj:`'sample'`)
    """

    def __init__(self,
                 first_val_ts: Union[Tuple, datetime] = None,
                 last_val_ts: Union[Tuple, datetime] = None,
                 first_test_ts: Union[Tuple, datetime] = None,
                 last_test_ts: Union[Tuple, datetime] = None,
                 first_train_ts: Union[Tuple, datetime] = None,
                 last_train_ts: Union[Tuple, datetime] = None,
                 min_offset: str = 'sample'):
        super(AtTimeStepSplitter, self).__init__()
        self.first_val_ts = first_val_ts
        self.last_val_ts = last_val_ts
        self.first_test_ts = first_test_ts
        self.last_test_ts = last_test_ts
        self.first_train_ts = first_train_ts
        self.last_train_ts = last_train_ts
        self.min_offset = min_offset

    def fit(self, dataset: SpatioTemporalDataset):
        if not isinstance(dataset.index, DatetimeIndex):
            raise ValueError(
                "AtTimeStepSplitter requires a SpatioTemporalDataset with a "
                "pandas.DatetimeIndex index, but the dataset's index is "
                f"{type(dataset.index).__name__}.")
        offset = self._min_gap(dataset)
        train_idx, val_idx, test_idx = self._resolve_ranges(dataset, offset)
        self._check_separated(offset,
                              train=train_idx,
                              val=val_idx,
                              test=test_idx)
        return self.set_indices(train_idx, val_idx, test_idx)

    def _min_gap(self, dataset: SpatioTemporalDataset) -> int:
        """Minimum separation (in positions) required between two splits."""
        if self.min_offset == 'sample':
            # Full footprint (window + horizon) disjointness.
            return int(np.ceil(dataset.sample_span / dataset.stride))
        if self.min_offset == 'window':
            # Lookback windows just touch; safe iff the gap also covers horizon.
            offset = dataset.samples_offset
            assert offset * dataset.stride >= dataset.horizon, (
                f"offset='window' separates splits by "
                f"{offset * dataset.stride} steps < horizon={dataset.horizon} "
                f"(window={dataset.window}, stride={dataset.stride}): target "
                f"steps would be shared across splits.")
            return offset
        raise ValueError(f"Unknown offset '{self.min_offset}', must be "
                         "'sample' or 'window'.")

    def _resolve_ranges(self, dataset: SpatioTemporalDataset, offset: int):
        """Resolve the train/val/test position ranges, inferring unspecified
        boundaries from the adjacent splits and the edges of the series."""
        n = dataset.n_samples

        # First/last sample position of a (half-open) timestamp bound, or
        # ``None`` when the bound is unspecified.
        def first_pos(ts):
            if ts is None:
                return None
            idx = indices_between(dataset, first_ts=ts)
            return int(idx.min()) if len(idx) else n

        def last_pos(ts):
            if ts is None:
                return None
            idx = indices_between(dataset, last_ts=ts)
            return int(idx.max()) if len(idx) else -1

        def as_idx(present, start, end):
            return (np.arange(start, end + 1) if present
                    else np.array([], dtype=int))

        val_present = (self.first_val_ts is not None
                       or self.last_val_ts is not None)
        test_present = (self.first_test_ts is not None
                        or self.last_test_ts is not None)
        vf, vl = first_pos(self.first_val_ts), last_pos(self.last_val_ts)
        tf, tl = first_pos(self.first_test_ts), last_pos(self.last_test_ts)
        train_last = last_pos(self.last_train_ts)

        # Validation: a missing end is placed ``offset`` before the test split
        # (chronological order train -> val -> test); edges fall back to the
        # series bounds.
        if val_present and vl is None:
            vl = (tf - offset) if tf is not None else n - 1
        val_start = vf if vf is not None else 0
        val_end = vl if vl is not None else n - 1

        # Test: a missing start is placed ``offset`` after validation. An
        # unspecified test (no bounds) is left empty.
        test_start, test_end = 0, n - 1
        if test_present:
            test_start = tf if tf is not None else \
                (val_end + offset if val_present else 0)
            test_end = tl if tl is not None else n - 1

        # Training: starts at the series beginning and ends ``offset`` before the
        # earliest held-out split, unless an explicit bound is given.
        train_start = first_pos(self.first_train_ts)
        if train_start is None:
            train_start = 0
        if train_last is not None:
            train_end = train_last
        else:
            held_out_starts = ([val_start] if val_present else []) + \
                              ([test_start] if test_present else [])
            train_end = (min(held_out_starts) - offset) if held_out_starts \
                else n - 1

        return (as_idx(True, train_start, train_end),
                as_idx(val_present, val_start, val_end),
                as_idx(test_present, test_start, test_end))

    @staticmethod
    def _check_separated(offset, **splits):
        """Raise if any two splits are closer than ``offset`` positions."""
        splits = {name: np.asarray(idxs)
                  for name, idxs in splits.items() if len(idxs)}
        names = list(splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                lo, hi = sorted((splits[a], splits[b]), key=lambda s: s.min())
                gap = int(hi.min()) - int(lo.max())
                if gap < offset:
                    raise ValueError(
                        f"'{a}' and '{b}' splits are not separated enough: they "
                        f"are {gap} positions apart but at least {offset} are "
                        "required to avoid sharing time steps across splits. "
                        "Adjust the timestamp ranges.")


def indices_between(dataset: SpatioTemporalDataset,
                    first_ts: Union[Tuple, datetime] = None,
                    last_ts: Union[Tuple, datetime] = None):
    if first_ts is not None:
        if not isinstance(first_ts, datetime):
            # first_ts must be (tuple, list) and len(first_ts) >= 3
            first_ts = datetime(*first_ts, tzinfo=dataset.index.tzinfo)
    if last_ts is not None:
        if not isinstance(last_ts, datetime):
            # last_ts must be (tuple, list) and len(last_ts) >= 3
            last_ts = datetime(*last_ts, tzinfo=dataset.index.tzinfo)
    first_day_loc, last_day_loc = dataset.index.slice_locs(first_ts, last_ts)
    first_sample_loc = first_day_loc - dataset.horizon_offset
    last_sample_loc = last_day_loc - dataset.horizon_offset - 1
    indices_after = first_sample_loc <= dataset.indices
    indices_before = dataset.indices < last_sample_loc
    indices = np.nonzero(indices_after & indices_before).ravel()
    return indices


def disjoint_months(dataset, months=None, synch_mode=SynchMode.WINDOW):
    idxs = np.arange(len(dataset))
    months = ensure_list(months)
    # divide indices according to window or horizon
    if synch_mode is SynchMode.WINDOW:
        start = 0
        end = dataset.window - 1
    elif synch_mode is SynchMode.HORIZON:
        start = dataset.horizon_offset
        end = dataset.horizon_offset + dataset.horizon - 1
    else:
        raise ValueError("synch_mode can only be one of "
                         f"{[SynchMode.WINDOW, SynchMode.HORIZON]}")
    # after idxs
    indices = np.asarray(dataset._indices)
    start_in_months = np.in1d(dataset.index[indices + start].month, months)
    end_in_months = np.in1d(dataset.index[indices + end].month, months)
    idxs_in_months = start_in_months & end_in_months
    after_idxs = idxs[idxs_in_months]
    # previous idxs
    months = np.setdiff1d(np.arange(1, 13), months)
    start_in_months = np.in1d(dataset.index[indices + start].month, months)
    end_in_months = np.in1d(dataset.index[indices + end].month, months)
    idxs_in_months = start_in_months & end_in_months
    prev_idxs = idxs[idxs_in_months]
    return prev_idxs, after_idxs


# aliases
temporal = TemporalSplitter
at_ts = AtTimeStepSplitter
