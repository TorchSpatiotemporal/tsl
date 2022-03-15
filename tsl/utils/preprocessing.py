import numpy as np
import pandas as pd

import tsl


def aggregate(dataframe, idx=None, type='sum'):
    # todo: make it work with multindex dataframes
    aggregation_fn = getattr(np, type)
    if idx is None:
        aggr = aggregation_fn(dataframe.values, axis=1)
        return pd.DataFrame(aggr, index=dataframe.index, columns=['seq'])

    else:
        ids = np.unique(idx)
        aggregates = []
        x = dataframe.values.T
        for i in ids:
            aggregates.append(aggregation_fn(x[idx == i], axis=0))

        cols = ['seq' + '_' + str(i) for i in ids]
        return pd.DataFrame(dict(zip(cols, aggregates)), index=dataframe.index)


def get_trend(df, period='week', train_len=None, valid_mask=None):
    """
    Perform detrending on a time series by subtrating from each value of the input dataframe
    the average value computed over the training dataset for each hour/weekday
    :param df: dataframe
    :param period: period of the trend ('day', 'week', 'month')
    :param train_len: train length,
    :return:
        - the detrended datasets
        - the trend values that has to be added back after computing the prediction
    """
    df = df.copy()
    if train_len is not None:
        df[train_len:] = np.nan
    if valid_mask is not None:
        df[~valid_mask] = np.nan
    idx = [df.index.hour, df.index.minute]
    if period == 'week':
        idx = [df.index.weekday, ] + idx
    elif period == 'month':
        idx = [df.index.month, df.index.weekday] + idx
    elif period != 'day':
        raise NotImplementedError("Period must be in ('day', 'week', 'month')")

    means = df.groupby(idx).transform(np.nanmean)
    return means


def normalize_by_group(df, by):
    """
    Normalizes a dataframe using mean and std of a specified group.

    :param df: the data
    :param by: used to determine the groups for the groupby
    :return: the normalized df
    """
    groups = df.groupby(by)
    # computes group-wise mean/std,
    # then auto broadcasts to size of group chunk
    mean = groups.transform(np.nanmean)
    std = groups.transform(np.nanstd) + tsl.epsilon  # add epsilon to avoid division by zero
    return (df[mean.columns] - mean) / std
