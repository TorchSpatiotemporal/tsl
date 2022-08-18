from typing import Union, Optional, List, Tuple, Mapping

import numpy as np
import pandas as pd
from . import casting
from ...ops.framearray import framearray_shape, framearray_to_numpy
from ...ops.pattern import check_pattern
from ...typing import FrameArray, DataArray
from ...utils.python_utils import ensure_list


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
            obj = casting.convert_precision_numpy(obj, precision=self.precision)
        return obj

    def _parse_covariate(self, obj: FrameArray, pattern: Optional[str] = None) \
            -> Tuple[FrameArray, str]:
        # compute object shape
        shape = framearray_shape(obj)
        # infer pattern if it is None
        if pattern is None:
            pattern = self._infer_pattern(shape)
        # check that pattern and shape match
        pattern = check_pattern(pattern)
        dims = pattern.strip().split(' ')

        if isinstance(obj, pd.DataFrame):
            assert self.is_target_dataframe, \
                "Cannot add DataFrame covariates if target is ndarray."
            obj = obj.reindex(index=casting.token_to_index_df(
                self, dims[0], obj.index))
            for lvl, tkn in enumerate(dims[1:]):
                columns = casting.token_to_index_df(
                    self, tkn, obj.columns.unique(lvl))
                obj.reindex(columns=columns, level=lvl)
            obj = casting.convert_precision_df(obj, precision=self.precision)
        else:
            obj = np.asarray(obj)
            # check shape
            for d, s in zip(dims, obj.shape):
                casting.token_to_index_array(self, d, s)
            obj = casting.convert_precision_numpy(obj, precision=self.precision)

        return obj, pattern

    def _columns_multiindex(self, nodes=None, channels=None):
        nodes = nodes if nodes is not None else self.nodes
        channels = channels if channels is not None else self.channels
        return pd.MultiIndex.from_product([nodes, channels],
                                          names=['nodes', 'channels'])

    def _infer_pattern(self, shape: tuple):
        out = []
        for dim in shape:
            if dim == self.length:
                out.append('t')
            elif dim == self.n_nodes:
                out.append('n')
            else:
                out.append('f')
        pattern = ' '.join(out)
        try:
            pattern = check_pattern(pattern)
        except RuntimeError:
            raise RuntimeError(f"Cannot infer pattern from shape: {shape}.")
        return pattern

    def _value_to_kwargs(self, value: Union[DataArray, List, Tuple, Mapping]):
        keys = ['value', 'pattern']
        if isinstance(value, DataArray.__args__):
            return dict(value=value)
        if isinstance(value, (list, tuple)):
            return dict(zip(keys, value))
        elif isinstance(value, Mapping):
            assert set(value.keys()).issubset(keys)
            return value
        else:
            raise TypeError('Invalid type for value "{}"'.format(type(value)))


class TemporalFeaturesMixin:

    def datetime_encoded(self, units):
        if not casting.is_datetime_like_index(self.index):
            raise NotImplementedError("This method can be used only with "
                                      "datetime-like index.")
        units = ensure_list(units)
        mapping = {un: pd.to_timedelta('1' + un).delta
                   for un in ['day', 'hour', 'minute', 'second',
                              'millisecond', 'microsecond', 'nanosecond']}
        mapping['week'] = pd.to_timedelta('1W').delta
        mapping['year'] = 365.2425 * 24 * 60 * 60 * 10 ** 9
        index_nano = self.index.view(np.int64)
        datetime = dict()
        for unit in units:
            if unit not in mapping:
                raise ValueError()
            nano_sec = index_nano * (2 * np.pi / mapping[unit])
            datetime[unit + '_sin'] = np.sin(nano_sec)
            datetime[unit + '_cos'] = np.cos(nano_sec)
        return pd.DataFrame(datetime, index=self.index, dtype=np.float32)

    def datetime_onehot(self, units):
        if not casting.is_datetime_like_index(self.index):
            raise NotImplementedError("This method can be used only with "
                                      "datetime-like index.")
        units = ensure_list(units)
        datetime = dict()
        for unit in units:
            if hasattr(self.index.__dict__, unit):
                raise ValueError()
            datetime[unit] = getattr(self.index, unit)
        dummies = pd.get_dummies(pd.DataFrame(datetime, index=self.index),
                                 columns=units)
        return dummies

    def holidays_onehot(self, country, subdiv=None):
        """Returns a DataFrame to indicate if dataset timestamps is holiday.
        See https://python-holidays.readthedocs.io/en/latest/

        Args:
            country (str): country for which holidays have to be checked, e.g.,
                "CH" for Switzerland.
            subdiv (dict, optional): optional country sub-division (state,
                region, province, canton), e.g., "TI" for Ticino, Switzerland.

        Returns: 
            pandas.DataFrame: DataFrame with one column ("holiday") as one-hot
                encoding (1 if the timestamp is in a holiday, 0 otherwise).
        """
        if not casting.is_datetime_like_index(self.index):
            raise NotImplementedError("This method can be used only with "
                                      "datetime-like index.")
        try:
            import holidays
        except ModuleNotFoundError:
            raise RuntimeError("You should install optional dependency "
                               "'holidays' to call 'datetime_holidays'.")

        years = np.unique(self.index.year.values)
        h = holidays.country_holidays(country, subdiv=subdiv, years=years)

        # label all the timestamps, whether holiday or not
        out = pd.DataFrame(0, dtype=np.uint8,
                           index=self.index.normalize(), columns=['holiday'])
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
