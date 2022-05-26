import numpy as np
import pandas as pd

from tsl.ops.dataframe import to_numpy
from . import checks
from ...typing import FrameArray
from ...utils.python_utils import ensure_list


class PandasParsingMixin:

    def _parse_dataframe(self, df: pd.DataFrame, node_level: bool = True):
        assert checks.is_datetime_like_index(df.index)
        if node_level:
            df = checks.to_nodes_channels_columns(df)
        else:
            df = checks.to_channels_columns(df)
        df = checks.cast_df(df, precision=self.precision)
        return df

    def _to_indexed_df(self, array: np.ndarray):
        if array.ndim == 1:
            array = array[..., None]
        # check shape equivalence
        time, channels = array.shape
        if time != self.length:
            raise ValueError("Cannot match temporal dimensions {} and {}"
                             .format(time, self.length))
        return pd.DataFrame(array, self.index)

    def _to_primary_df_schema(self, array: np.ndarray):
        array = np.asarray(array)
        while array.ndim < 3:
            array = array[..., None]
        # check shape equivalence
        time, nodes, channels = array.shape
        if time != self.length:
            raise ValueError("Cannot match temporal dimensions {} and {}"
                             .format(time, self.length))
        if nodes != self.n_nodes:
            raise ValueError("Cannot match nodes dimensions {} and {}"
                             .format(nodes, self.n_nodes))
        array = array.reshape(time, nodes * channels)
        columns = self.columns(channels=pd.RangeIndex(channels))
        return pd.DataFrame(array, self.index, columns)

    def _synch_with_primary(self, df: pd.DataFrame):
        assert hasattr(self, 'df'), \
            "Cannot call this method before setting primary dataframe."
        if df.columns.nlevels == 2:
            nodes = set(df.columns.unique(0))
            channels = list(df.columns.unique(1))
            assert nodes.issubset(self.nodes), \
                "You are trying to add an exogenous dataframe with nodes that" \
                " are not in the dataset."
            columns = self.columns(channels=channels)
            df = df.reindex(index=self.index, columns=columns)
        elif df.columns.nlevels == 1:
            df = df.reindex(index=self.index)
        else:
            raise ValueError("Input dataframe must have either 1 ('nodes' or "
                             "'channels') or 2 ('nodes', 'channels') column "
                             "levels.")
        return df

    def _check_name(self, name: str, check_type: str):
        assert check_type in ['exogenous', 'attribute']
        invalid_names = set(dir(self))
        if check_type == 'exogenous':
            invalid_names.update(self._attributes)
        else:
            invalid_names.update(self._exogenous)
        if name in invalid_names:
            raise ValueError(f"Cannot set {check_type} with name '{name}', "
                             f"{self.__class__.__name__} contains already an "
                             f"attribute named '{name}'.")


class TemporalFeaturesMixin:

    def datetime_encoded(self, units):
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
    eval_mask: np.ndarray

    def set_eval_mask(self, eval_mask: FrameArray):
        if isinstance(eval_mask, pd.DataFrame):
            eval_mask = to_numpy(self._parse_dataframe(eval_mask))
        if eval_mask.ndim == 2:
            eval_mask = eval_mask[..., None]
        assert eval_mask.shape == self.shape
        eval_mask = eval_mask.astype(self.mask.dtype) & self.mask
        self.eval_mask = eval_mask

    @property
    def training_mask(self):
        if hasattr(self, 'eval_mask') and self.eval_mask is not None:
            return self.mask & (1 - self.eval_mask)
        return self.mask
