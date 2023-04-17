from typing import Union

import numpy as np
import pandas as pd
import pandas.tseries.frequencies as pd_freq

from tsl import logger
from tsl.utils.python_utils import precision_stoi


def to_nodes_channels_columns(df: pd.DataFrame,
                              inplace: bool = True) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    if df.columns.nlevels == 1:
        nodes = list(df.columns)
        columns = pd.MultiIndex.from_product([nodes, pd.RangeIndex(1)],
                                             names=['nodes', 'channels'])
        df.columns = columns
        logger.debug('Inferred input data-format: [time, nodes]')
    elif df.columns.nlevels == 2:
        # # make tabular
        cols = [df.columns.unique(i) for i in range(2)]
        cols = pd.MultiIndex.from_product(cols, names=['nodes', 'channels'])
        if not df.columns.equals(cols):
            df = df.reindex(columns=cols)
        else:
            df.columns.names = ['nodes', 'channels']
    else:
        raise ValueError("Input dataframe must have either 1 ('nodes') "
                         "or 2 ('nodes', 'channels') column levels.")
    return df


def convert_precision_df(df: pd.DataFrame,
                         precision: Union[int, str] = None,
                         inplace: bool = True) -> pd.DataFrame:
    if precision is None:
        return df
    precision = precision_stoi(precision)
    if not inplace:
        df = df.copy()
    # float to float{precision}
    to_dtype = f'float{precision}'
    from_dtypes = {'float16', 'float32', 'float64'}.difference({to_dtype})
    float_cols = df.select_dtypes(include=from_dtypes).columns
    df[float_cols] = df[float_cols].astype(to_dtype)
    # int to int{precision}
    to_dtype = f'int{precision}'
    from_dtypes = {'int16', 'int32', 'int64'}.difference({to_dtype})
    int_cols = df.select_dtypes(include=from_dtypes).columns
    df[int_cols] = df[int_cols].astype(to_dtype)
    return df


def convert_precision_numpy(arr: np.ndarray,
                            precision: Union[int, str] = None) -> np.ndarray:
    if precision is None:
        return arr
    precision = precision_stoi(precision)
    # float to float{precision}
    if arr.dtype.name.startswith('float'):
        return arr.astype(f'float{precision}')
    # int to int{precision}
    if arr.dtype.name.startswith('int'):
        return arr.astype(f'int{precision}')
    return arr


def to_pandas_freq(freq):
    try:
        freq = pd_freq.to_offset(freq)
    except ValueError:
        raise ValueError(f"Value '{freq}' is not a valid frequency.")
    return freq


def is_datetime_like_index(index):
    return isinstance(index,
                      (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex))


def check_time_unit(time_unit: str, include_onehot: bool = False):
    allowed_units = {
        'year', 'week', 'day', 'hour', 'minute', 'second', 'millisecond',
        'microsecond', 'nanosecond'
    }
    if include_onehot:
        allowed_units.update(
            {'weekday', 'day_of_week', 'dayofweek', 'weekofyear'})
    if time_unit not in allowed_units:
        raise RuntimeError(f"'{time_unit}' is not a valid time unit. "
                           f"Allowed units are {', '.join(allowed_units)}.")


def time_unit_to_nanoseconds(time_unit: str):
    check_time_unit(time_unit)
    if time_unit == 'year':
        return 365.2425 * 24 * 60 * 60 * 10**9
    elif time_unit == 'week':
        time_unit = 'W'
    return pd.Timedelta('1' + time_unit).value
