from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import Index

from ...ops.framearray import framearray_shape, framearray_to_numpy
from ...ops.pattern import check_pattern, infer_pattern
from ...typing import FrameArray
from ...utils.python_utils import ensure_list
from . import casting


class TabularParsingMixin:

    def _parse_target(self, obj: FrameArray) -> FrameArray:
        # if target is DataFrame
        if isinstance(obj, pd.DataFrame):
            casting.to_nodes_channels_columns(obj)
            obj = casting.convert_precision_df(obj, precision=self.precision)
        # if target is array-like
        else:
            obj = np.asarray(obj)
            # reshape to [time, nodes, features]
            while obj.ndim < 3:
                obj = obj[..., None]
            assert obj.ndim == 3, \
                "Target signal must be 3-dimensional with pattern 't n f'."
            obj = casting.convert_precision_numpy(obj,
                                                  precision=self.precision)
        return obj

    def _parse_covariate(self, obj: FrameArray, pattern: Optional[str] = None) \
            -> Tuple[FrameArray, str]:
        # compute object shape
        shape = framearray_shape(obj)

        # infer pattern if it is None, otherwise sanity check
        if pattern is None:
            pattern = infer_pattern(shape, t=self.length, n=self.n_nodes)
        else:
            # check that pattern and shape match
            pattern = check_pattern(pattern, ndim=len(shape))

        dims = pattern.split(' ')  # 't n f' -> ['t', 'n', 'f']

        if isinstance(obj, pd.DataFrame):
            assert self.is_target_dataframe, \
                "Cannot add DataFrame covariates if target is ndarray."
            # check covariate index matches steps or nodes in the dataset
            # according to the dim token
            index = self._token_to_index(dims[0], obj.index)
            obj = obj.reindex(index=index)

            # todo check when columns is not multiindex
            #  add 1 dummy feature dim always?
            for lvl, tkn in enumerate(dims[1:]):
                columns = self._token_to_index(tkn, obj.columns.unique(lvl))
                if isinstance(obj.columns, pd.MultiIndex):
                    obj.reindex(columns=columns, level=lvl)
                else:
                    obj.reindex(columns=columns)
            obj = casting.convert_precision_df(obj, precision=self.precision)
        else:
            obj = np.asarray(obj)
            # check shape
            for d, s in zip(dims, obj.shape):
                self._token_to_index(d, s)
            obj = casting.convert_precision_numpy(obj,
                                                  precision=self.precision)

        return obj, pattern

    def _token_to_index(self, token, index_or_size: Union[int, Index]):
        no_index = isinstance(index_or_size, int)
        if token == 't':
            if no_index:
                assert index_or_size == len(self.index)
            return self.index if self.force_synchronization else index_or_size
        if token == 'n':
            if no_index:
                assert index_or_size == len(self.nodes)
            else:
                assert set(index_or_size).issubset(self.nodes), \
                    "You are trying to add a covariate dataframe with " \
                    "nodes that are not in the dataset."
            return self.nodes
        if token in ['c', 'f'] and not no_index:
            return index_or_size

    def _columns_multiindex(self, nodes=None, channels=None):
        nodes = nodes if nodes is not None else self.nodes
        channels = channels if channels is not None else self.channels
        return pd.MultiIndex.from_product([nodes, channels],
                                          names=['nodes', 'channels'])

    def _value_to_kwargs(self, value: Union[FrameArray, List, Tuple, Mapping]):
        keys = ['value', 'pattern']
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            return dict(value=value)
        if isinstance(value, (list, tuple)):
            return dict(zip(keys, value))
        elif isinstance(value, Mapping):
            assert set(value.keys()).issubset(keys)
            return value
        else:
            raise TypeError('Invalid type for value "{}"'.format(type(value)))


class TemporalFeaturesMixin:

    def __check_temporal_index(self):
        if not casting.is_datetime_like_index(self.index):
            raise NotImplementedError("This method can be used only with "
                                      "datetime-like index.")

    def datetime_encoded(self, units: Union[str, List]) -> pd.DataFrame:
        r"""Transform dataset's temporal index into covariates using sinusoidal
        transformations. Each temporal unit is used as period to compute the
        operations, obtaining two feature (:math:`\sin` and :math:`\cos`) for
        each unit."""
        self.__check_temporal_index()
        units = ensure_list(units)
        index_nano = self.index.view(np.int64)
        datetime = dict()
        for unit in units:
            nano_unit = casting.time_unit_to_nanoseconds(unit)
            nano_sec = index_nano * (2 * np.pi / nano_unit)
            datetime[unit + '_sin'] = np.sin(nano_sec)
            datetime[unit + '_cos'] = np.cos(nano_sec)
        return pd.DataFrame(datetime, index=self.index, dtype=np.float32)

    def datetime_onehot(self, units: Union[str, List]) -> pd.DataFrame:
        r"""Transform dataset's temporal index into one-hot-encodings for each
        temporal unit specified. Internally, this function calls
        :func:`pandas.get_dummies`."""
        self.__check_temporal_index()
        units = ensure_list(units)
        datetime = dict()
        for unit in units:
            # check that unit is a valid datetime unit
            casting.check_time_unit(unit, include_onehot=True)
            datetime[unit] = getattr(self.index, unit)
        dummies = pd.get_dummies(pd.DataFrame(datetime, index=self.index),
                                 columns=units)
        return dummies

    def holidays_onehot(self, country, subdiv=None) -> pd.DataFrame:
        """Returns a DataFrame to indicate if dataset timestamps is holiday.
        See https://python-holidays.readthedocs.io/en/latest/.

        Args:
            country (str): country for which holidays have to be checked, e.g.,
                "CH" for Switzerland.
            subdiv (dict, optional): optional country sub-division (state,
                region, province, canton), e.g., "TI" for Ticino, Switzerland.

        Returns:
            pandas.DataFrame: DataFrame with one column ("holiday") as one-hot
                encoding (1 if the timestamp is in a holiday, 0 otherwise).
        """
        self.__check_temporal_index()
        try:
            import holidays
        except ModuleNotFoundError:
            raise RuntimeError("You should install optional dependency "
                               "'holidays' to call 'datetime_holidays'.")

        years = np.unique(self.index.year.values)
        h = holidays.country_holidays(country, subdiv=subdiv, years=years)

        # label all the timestamps, whether holiday or not
        out = pd.DataFrame(0,
                           dtype=np.uint8,
                           index=self.index.normalize(),
                           columns=['holiday'])
        for date in h.keys():
            try:
                out.loc[[date]] = 1
            except KeyError:
                pass
        out.index = self.index

        return out


class MissingValuesMixin:

    def set_eval_mask(self, eval_mask: FrameArray):
        eval_mask = self._parse_target(eval_mask)
        eval_mask = framearray_to_numpy(eval_mask).astype(bool)
        eval_mask = eval_mask & self.mask
        self.add_covariate('eval_mask', eval_mask, 't n f')

    @property
    def training_mask(self):
        if hasattr(self, 'eval_mask') and self.eval_mask is not None:
            return self.mask & ~self.eval_mask
        return self.mask
