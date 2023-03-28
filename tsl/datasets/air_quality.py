import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.data.synch_mode import HORIZON
from tsl.datasets.prototypes import DatetimeDataset
from tsl.datasets.prototypes.mixin import MissingValuesMixin
from tsl.utils import download_url, extract_zip


def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1
    if it is present in the DataFrame and absent in the :obj:`infer_from` month.

    Args:
        df (pd.Dataframe): The DataFrame.
        infer_from (str): Denotes from which month the evaluation value must be
            inferred. Can be either :obj:`previous` or :obj:`next`.

    Returns:
        pd.DataFrame: The evaluation mask for the DataFrame.
    """
    mask = (~df.isna()).astype('uint8')
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns,
                             data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('`infer_from` can only be one of {}'.format(
            ['previous', 'next']))
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    length = len(months)
    for i in range(length):
        j = (i + offset) % length
        year_i, month_i = months[i]
        year_j, month_j = months[j]
        cond_j = (mask.index.year == year_j) & (mask.index.month == month_j)
        mask_j = mask[cond_j]
        offset_i = 12 * (year_i - year_j) + (month_i - month_j)
        mask_i = mask_j.shift(1, pd.DateOffset(months=offset_i))
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        i_idx = mask_i.index
        eval_mask.loc[i_idx] = ~mask_i.loc[i_idx] & mask.loc[i_idx]
    return eval_mask


class AirQualitySplitter(Splitter):

    def __init__(self,
                 val_len: int = None,
                 test_months: Sequence = (3, 6, 9, 12)):
        super(AirQualitySplitter, self).__init__()
        self._val_len = val_len
        self.test_months = test_months

    def fit(self, dataset):
        nontest_idxs, test_idxs = disjoint_months(dataset,
                                                  months=self.test_months,
                                                  synch_mode=HORIZON)
        # take equal number of samples before each month of testing
        val_len = self._val_len
        if val_len < 1:
            val_len = int(val_len * len(nontest_idxs))
        val_len = val_len // len(self.test_months)
        # get indices of first day of each testing month
        delta = np.diff(test_idxs)
        delta_idxs = np.flatnonzero(delta > delta.min())
        end_month_idxs = test_idxs[1:][delta_idxs]
        if len(end_month_idxs) < len(self.test_months):
            end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
        # expand month indices
        month_val_idxs = [
            np.arange(v_idx - val_len, v_idx) - dataset.window
            for v_idx in end_month_idxs
        ]
        val_idxs = np.concatenate(month_val_idxs) % len(dataset)
        # remove overlapping indices from training set
        ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs,
                                                  val_idxs,
                                                  synch_mode=HORIZON,
                                                  as_mask=True)
        train_idxs = nontest_idxs[~ovl_idxs]
        self.set_indices(train_idxs, val_idxs, test_idxs)


class AirQuality(DatetimeDataset, MissingValuesMixin):
    r"""Measurements of pollutant :math:`PM2.5` collected by 437 air quality
    monitoring stations spread across 43 Chinese cities from May 2014 to April
    2015.

    The dataset contains also a smaller version :obj:`AirQuality(small=True)`
    with only the subset of nodes containing the 36 sensors in Beijing.

    Data collected inside the `Urban Air
    <https://www.microsoft.com/en-us/research/project/urban-air/>`_ project.

    Dataset size:
        + Time steps: 8760
        + Nodes: 437
        + Channels: 1
        + Sampling rate: 1 hour
        + Missing values: 25.67%

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """
    url = "https://drive.switch.ch/index.php/s/W0fRqotjHxIndPj/download"

    similarity_options = {'distance'}

    def __init__(self,
                 root: str = None,
                 impute_nans: bool = True,
                 small: bool = False,
                 test_months: Sequence = (3, 6, 9, 12),
                 infer_eval_from: str = 'next',
                 freq: Optional[str] = None,
                 masked_sensors: Optional[Sequence] = None):
        self.root = root
        self.small = small
        self.test_months = test_months
        self.infer_eval_from = infer_eval_from  # [next, previous]
        if masked_sensors is None:
            self.masked_sensors = []
        else:
            self.masked_sensors = list(masked_sensors)
        df, mask, eval_mask, dist = self.load(impute_nans=impute_nans)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         default_splitting_method='air_quality',
                         name='AQI36' if self.small else 'AQI')
        self.add_covariate('dist', dist, pattern='n n')
        self.set_eval_mask(eval_mask)

    @property
    def raw_file_names(self) -> List[str]:
        return ['full437.h5', 'small36.h5']

    @property
    def required_file_names(self) -> List[str]:
        return self.raw_file_names + ['aqi_dist.npy']

    def download(self):
        path = download_url(self.url, self.root_dir, 'data.zip')
        extract_zip(path, self.root_dir)
        os.unlink(path)

    def build(self):
        self.maybe_download()
        # compute distances from latitude and longitude degrees
        path = os.path.join(self.root_dir, 'full437.h5')
        stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(st_coord, to_rad=True).values
        np.save(os.path.join(self.root_dir, 'aqi_dist.npy'), dist)

    def load_raw(self):
        self.maybe_build()
        dist = np.load(os.path.join(self.root_dir, 'aqi_dist.npy'))
        if self.small:
            path = os.path.join(self.root_dir, 'small36.h5')
            eval_mask = pd.read_hdf(path, 'eval_mask')
            dist = dist[:36, :36]
        else:
            path = os.path.join(self.root_dir, 'full437.h5')
            eval_mask = None
        df = pd.read_hdf(path, 'pm25')
        return pd.DataFrame(df), dist, eval_mask

    def load(self, impute_nans=True):
        # load readings and stations metadata
        df, dist, eval_mask = self.load_raw()
        # compute the masks:
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is valid
        if eval_mask is None:
            eval_mask = infer_mask(df, infer_from=self.infer_eval_from)
        # 1 if value is ground-truth for imputation
        eval_mask = eval_mask.values.astype('uint8')
        if len(self.masked_sensors):
            eval_mask[:, self.masked_sensors] = mask[:, self.masked_sensors]
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            from tsl.ops.framearray import temporal_mean
            df = df.fillna(temporal_mean(df))
        return df, mask, eval_mask, dist

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'air_quality':
            val_len = kwargs.get('val_len')
            return AirQualitySplitter(test_months=self.test_months,
                                      val_len=val_len)

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            from tsl.ops.similarities import gaussian_kernel

            # use same theta for both air and air36
            theta = np.std(self.dist[:36, :36])
            return gaussian_kernel(self.dist, theta=theta)
