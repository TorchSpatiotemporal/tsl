import functools
from copy import deepcopy
from datetime import datetime
from typing import Mapping, Callable, Union, Tuple, Optional

import numpy as np

from tsl.utils.python_utils import ensure_list
from ..spatiotemporal_dataset import SpatioTemporalDataset
from ..utils import SynchMode

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
        return dict(train_len=self.train_len, val_len=self.val_len,
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

    def __init__(self, val_split_fn: Callable = None,
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

    def __init__(self, train_idxs: Optional[Index] = None,
                 val_idxs: Optional[Index] = None,
                 test_idxs: Optional[Index] = None):
        super(FixedIndicesSplitter, self).__init__()
        self.set_indices(train_idxs, val_idxs, test_idxs)
        self._fitted = True

    def fit(self, dataset: SpatioTemporalDataset):
        pass


class TemporalSplitter(Splitter):

    def __init__(self, val_len: int = None, test_len: int = None):
        super(TemporalSplitter, self).__init__()
        self._val_len = val_len
        self._test_len = test_len

    def fit(self, dataset: SpatioTemporalDataset):
        idx = np.arange(len(dataset))
        val_len, test_len = self._val_len, self._test_len
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        self.set_indices(idx[:val_start - dataset.samples_offset],
                         idx[val_start:test_start - dataset.samples_offset],
                         idx[test_start:])

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--val-len', type=float or int, default=0.1)
        parser.add_argument('--test-len', type=float or int, default=0.2)
        return parser


class AtTimeStepSplitter(Splitter):

    def __init__(self, first_val_ts: Union[Tuple, datetime] = None,
                 first_test_ts: Union[Tuple, datetime] = None):
        super(AtTimeStepSplitter, self).__init__()
        self.first_val_ts = first_val_ts
        self.first_test_ts = first_test_ts

    def fit(self, dataset: SpatioTemporalDataset):
        train_idx, test_idx = split_at_ts(dataset, ts=self.first_test_ts)
        train_idx, val_idx = split_at_ts(dataset, ts=self.first_val_ts,
                                         mask=test_idx)
        return self.set_indices(train_idx, val_idx, test_idx)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--first-val-ts', type=list or tuple, default=None)
        parser.add_argument('--first-test-ts', type=list or tuple, default=None)
        return parser


###


def indices_between(dataset: SpatioTemporalDataset,
                    first_ts: Union[Tuple, datetime] = None,
                    last_ts: Union[Tuple, datetime] = None):
    if first_ts is not None:
        if isinstance(first_ts, datetime):
            pass
        elif isinstance(first_ts, (tuple, list)) and len(first_ts) >= 3:
            first_ts = datetime(*first_ts, tzinfo=dataset.index.tzinfo)
        else:
            raise TypeError("first_ts must be a datetime or a tuple")
    if last_ts is not None:
        if isinstance(last_ts, datetime):
            pass
        elif isinstance(last_ts, (tuple, list)) and len(last_ts) >= 3:
            last_ts = datetime(*last_ts, tzinfo=dataset.index.tzinfo)
        else:
            raise TypeError("last_ts must be a datetime or a tuple")
    first_day_loc, last_day_loc = dataset.index.slice_locs(first_ts, last_ts)
    first_sample_loc = first_day_loc - dataset.horizon_offset
    last_sample_loc = last_day_loc - dataset.horizon_offset - 1
    indices_from_sample = np.where((first_sample_loc <= dataset.indices) & (
            dataset.indices < last_sample_loc))[0]
    return indices_from_sample


def split_at_ts(dataset, ts, mask=None):
    from_day_idxs = indices_between(dataset, first_ts=ts)
    prev_idxs = np.arange(
        from_day_idxs[0] if len(from_day_idxs) else len(dataset))
    if mask is not None:
        from_day_idxs = np.setdiff1d(from_day_idxs, mask)
        prev_idxs = np.setdiff1d(prev_idxs, mask)
    return prev_idxs, from_day_idxs


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
        raise ValueError('synch_mode can only be one of %s'
                         % [SynchMode.WINDOW, SynchMode.HORIZON])
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


# SPLIT FUNCTIONS

def split_function_builder(fn, *args, name=None, **kwargs):
    def wrapper_split_fn(dataset, length=None, mask=None):
        return fn(dataset, length=length, mask=mask, *args, **kwargs)

    wrapper_split_fn.__name__ = name or "wrapped__%s" % fn.__name__
    return wrapper_split_fn


def subset_len(length, set_size, period=None):
    if period is None:
        period = set_size
    if length is None or length <= 0:
        length = 0
    if 0. < length < 1.:
        length = max(int(length * period), 1)
    elif period <= length < set_size:
        length = int(length / set_size * period)
    elif length > set_size:
        raise ValueError("Provided length of %i is greater than set_size %i" % (
            length, set_size))
    return length


def tail_of_period(iterable, length, mask=None, period=None):
    size = len(iterable)
    period = period or size
    if mask is None:
        mask = []
    indices = np.arange(size)
    length = subset_len(length, size, period)

    prev_idxs, after_idxs = [], []
    for batch_idxs in [indices[i:i + period] for i in range(0, size, period)]:
        batch_idxs = np.setdiff1d(batch_idxs, mask)
        prev_idxs.extend(batch_idxs[:len(batch_idxs) - length])
        after_idxs.extend(batch_idxs[len(batch_idxs) - length:])

    return np.array(prev_idxs), np.array(after_idxs)


def random(iterable, length, mask=None):
    size = len(iterable)
    if mask is None:
        mask = []
    indices = np.setdiff1d(np.arange(size), mask)
    np.random.shuffle(indices)
    split_at = len(indices) - subset_len(length, size)
    res = [np.sort(indices[:split_at]), np.sort(indices[split_at:])]
    return res


def past_pretest_days(dataset, length, mask):
    # get the first day of testing, as the first step of the horizon
    keep_until = np.min(mask)
    first_testing_day_idx = dataset._indices[keep_until]
    first_testing_day = dataset.index[
        first_testing_day_idx + dataset.lookback + dataset.delay]

    # extract samples before first day of testing through the years
    tz_info = dataset.index.tzinfo
    years = sorted(set(dataset.index.year))
    yearly_testing_loc = []
    for year in years:
        ftd_year = datetime(year, first_testing_day.month,
                            first_testing_day.day, tzinfo=tz_info)
        yearly_testing_loc.append(dataset.index.slice_locs(ftd_year)[0])
    yearly_train_samples = [
        np.where(dataset._indices < ytl - dataset.lookback - dataset.delay)[0]
        for ytl in yearly_testing_loc]
    # filter the years in which there are no such samples
    yearly_train_samples = [yts for yts in yearly_train_samples if len(yts) > 0]

    # for each year excluding the last take the last "val_len // n_years" samples
    yearly_val_len = length // len(yearly_train_samples)
    yearly_val_lens = [min(yearly_val_len, len(yts)) for yts in
                       yearly_train_samples[:-1]]
    # for the last year, take the remaining number of samples needed to reach val_len
    # this value is always greater or equals to the other so we have at least the same number of validation samples
    # coming from the last year than the maximum among all the other years.
    yearly_val_lens.append(length - sum(yearly_val_lens))
    # finally extracts the validation samples
    val_idxs = [idxs[-val_len:] for idxs, val_len in
                zip(yearly_train_samples, yearly_val_lens)]
    val_idxs = np.concatenate(val_idxs)

    # recompute training and test indices
    all_idxs = np.arange(len(dataset))
    train_idxs = np.setdiff1d(all_idxs, val_idxs)

    return train_idxs, val_idxs


def last_month(dataset, mask=None):
    if mask is not None:
        keep_until = np.min(mask)
        last_day_idx = dataset._indices[keep_until]
        last_day = dataset.index[last_day_idx]
    else:
        last_day = dataset.index[-1]
    split_day = (last_day.year, last_day.month, 1)
    return split_at_ts(dataset, split_day, mask)


# aliases
temporal = TemporalSplitter
at_ts = AtTimeStepSplitter
