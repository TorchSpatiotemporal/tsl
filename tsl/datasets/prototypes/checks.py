from typing import Union

import pandas as pd
import pandas.tseries.frequencies as pd_freq


def to_nodes_channels_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.columns.nlevels == 1:
        nodes = list(df.columns)
        columns = pd.MultiIndex.from_product([nodes, pd.RangeIndex(1)],
                                             names=['nodes', 'channels'])
        df.columns = columns
    elif df.columns.nlevels == 2:
        df.columns.names = ['nodes', 'channels']
    else:
        raise ValueError("Input dataframe must have either 1 ('nodes') "
                         "or 2 ('nodes', 'channels') column levels.")
    return df


def to_channels_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.columns.nlevels == 1:
        df.columns.name = 'channels'
    else:
        raise ValueError("Input dataframe must have 1 ('nodes') column levels.")
    return df


def cast_df(df: pd.DataFrame, precision: Union[int, str] = 32) -> pd.DataFrame:
    if isinstance(precision, str):
        precision = dict(half=16, full=32, double=64).get(precision)
    assert precision in [16, 32, 64], \
        "precision must be one of 16 (or 'half'), 32 (or 'full') or 64 " \
        f"(or 'double'). Default is 32, invalid input '{precision}'."
    df = df.copy()
    # float to float{precision}
    dtypes = ['float16', 'float32', 'float64']
    float_cols = df.select_dtypes(include=dtypes).columns
    df.loc[:, float_cols] = df[float_cols].astype(f'float{precision}')
    # int to int{precision}
    dtypes = ['int16', 'int32', 'int64']
    int_cols = df.select_dtypes(include=dtypes).columns
    df.loc[:, int_cols] = df[int_cols].astype(f'int{precision}')
    return df


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
