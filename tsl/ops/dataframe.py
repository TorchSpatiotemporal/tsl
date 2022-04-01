from typing import Union, Callable

import numpy as np
import pandas as pd

from tsl.typing import Index


def to_numpy(df):
    if df.columns.nlevels == 1:
        return df.to_numpy()
    cols = [df.columns.unique(i) for i in range(df.columns.nlevels)]
    cols = pd.MultiIndex.from_product(cols)
    df = df.reindex(columns=cols)
    return df.values.reshape((-1, *cols.levshape))


def aggregate(df: pd.DataFrame, node_index: Index, aggr_fn: Callable = np.sum):
    """Aggregate nodes in MultiIndexed DataFrames.

    Args:
        df (pd.DataFrame): MultiIndexed DataFrame to be aggregated. Columns must
            be a :class:`~pandas.MultiIndex` object with :obj:`nodes` in first
            level and :obj:`channels` in second.
        node_index (Index): A sequence of :obj:`cluster_id` with length equal to
            number of nodes in :obj:`df`. The i-th node will be mapped to
            cluster at i-th position in :obj:`node_index`.
        aggr_fn (Callable): Function to be used for cluster aggregation.
    """
    assert df.columns.nlevels == 2,\
        "This function currently supports only MultiIndexed DataFrames."
    channels = df.columns.unique(1).values
    grouper = pd.MultiIndex.from_product([node_index, channels],
                                         names=df.columns.names)
    df = df.groupby(grouper, axis=1).aggregate(aggr_fn)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=grouper.names)
    return df


def compute_mean(x: Union[pd.DataFrame, np.ndarray],
                 index: pd.DatetimeIndex = None
                 ) -> Union[pd.DataFrame, np.ndarray]:
    """Compute the mean values for each row.

    The mean is first computed hourly over the week of the year. Further
    :obj:`NaN` values are imputed using hourly mean over the same month through
    the years. If other :obj:`NaN` are present, they are replaced with the mean
    of the sole hours. Remaining missing values are filled with :obj:`ffill` and
    :obj:`bfill`.

    Args:
        x (np.array | pd.Dataframe): Array-like with missing values.
        index (pd.DatetimeIndex | pd.PeriodIndex | pd.TimedeltaIndex, optional):
            Temporal index if x is not a :obj:'~pandas.Dataframe' with a
            temporal index. Must have same length as :obj:`x`.
            (default :obj:`None`)
    """
    if index is not None:
        if not isinstance(index, pd.DatetimeIndex):
            # try casting
            index = pd.to_datetime(index)
        assert len(index) == len(x)
        if isinstance(x, pd.DataFrame):
            # override index of x
            df_mean = x.copy().set_index(index)
        else:
            # try casting to np.ndarray
            x = np.asarray(x)
            shape = x.shape
            # x can be N-dimensional, we flatten all but the first dimensions
            x = x.reshape((shape[0], -1))
            df_mean = pd.DataFrame(x, index=index)
    elif isinstance(x, pd.DataFrame):
        df_mean = x.copy()
    else:
        raise TypeError("`x` must be a pd.Dataframe or a np.ndarray.")
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week,
             df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean


# todo convert to pandas function with holidays
def holidays(index):
    """Return a binary dataframe that takes value: 1 if the day is a holiday, 0
    otherwise.

    Args:
        index (pd.DateTimeIndex): The datetime-like index.
    """
    assert isinstance(index, pd.DatetimeIndex)
    holidays = (index.month == 1) & (index.day == 1)  # new year
    holidays |= (index.month == 1) & (index.day == 6)  # epiphany
    holidays |= (index.month == 5) & (index.day == 21)  # ascension day
    holidays |= (index.month == 12) & (index.day == 24)  # Christmas' eve
    holidays |= (index.month == 12) & (index.day == 25)  # Christmas
    holidays |= (index.month == 12) & (index.day == 26)  # Saint Steven
    holidays |= index.weekday == 6  # sundays
    df = pd.DataFrame(holidays, index=index, columns='holiday').astype('uint8')
    return df
