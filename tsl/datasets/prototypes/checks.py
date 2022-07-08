import re
from typing import Union

import numpy as np
import pandas as pd
import pandas.tseries.frequencies as pd_freq

PATTERN_MATCH = re.compile('^(t?){2}(n?){2}[cf]*$')


def to_nodes_channels_columns(df: pd.DataFrame,
                              inplace: bool = True) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    if df.columns.nlevels == 1:
        nodes = list(df.columns)
        columns = pd.MultiIndex.from_product([nodes, pd.RangeIndex(1)],
                                             names=['nodes', 'channels'])
        df.columns = columns
    elif df.columns.nlevels == 2:
        df.columns.names = ['nodes', 'channels']
        # make tabular
        df = df.reindex(columns=df.columns.unique(0), level=0)
        df = df.reindex(columns=df.columns.unique(1), level=1)
    else:
        raise ValueError("Input dataframe must have either 1 ('nodes') "
                         "or 2 ('nodes', 'channels') column levels.")
    return df


def to_channels_columns(df: pd.DataFrame,
                        inplace: bool = True) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    if df.columns.nlevels == 1:
        df.columns.name = 'channels'
    else:
        raise ValueError("Input dataframe must have 1 ('nodes') column levels.")
    return df


def precision_stoi(precision: Union[int, str]):
    if isinstance(precision, str):
        precision = dict(half=16, full=32, double=64).get(precision)
    assert precision in [16, 32, 64], \
        "precision must be one of 16 (or 'half'), 32 (or 'full') or 64 " \
        f"(or 'double'). Default is 32, invalid input '{precision}'."
    return precision


def convert_precision_df(df: pd.DataFrame, precision: Union[int, str] = None,
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
    df.loc[:, float_cols] = df[float_cols].astype(to_dtype)
    # int to int{precision}
    to_dtype = f'int{precision}'
    from_dtypes = {'int16', 'int32', 'int64'}.difference({to_dtype})
    int_cols = df.select_dtypes(include=from_dtypes).columns
    df.loc[:, int_cols] = df[int_cols].astype(to_dtype)
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


def check_pattern(pattern: str) -> str:
    pattern_squeezed = pattern.replace(' ', '')
    # check 'c'/'f' follows 'n', 'n' follows 't'
    # allow for duplicate 'n' or 't' dims (e.g., 'n n', 't t n f')
    # allow for limitless 'c'/'f' dims (e.g., 't n f f')
    if not PATTERN_MATCH.match(pattern_squeezed):
        raise RuntimeError(f'Pattern "{pattern}" not allowed.')
    return ' '.join(pattern_squeezed)


def to_pandas_freq(freq):
    try:
        freq = pd_freq.to_offset(freq)
    except ValueError:
        raise ValueError(f"Value '{freq}' is not a valid frequency.")
    return freq


def is_datetime_like_index(index):
    return isinstance(index, (pd.DatetimeIndex,
                              pd.PeriodIndex,
                              pd.TimedeltaIndex))
