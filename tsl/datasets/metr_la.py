import os

import numpy as np
import pandas as pd

from tsl import logger

from ..ops.similarities import gaussian_kernel
from ..utils import download_url, extract_zip
from .prototypes import DatetimeDataset


class MetrLA(DatetimeDataset):
    r"""Traffic readings collected from 207 loop detectors on
    highways in Los Angeles County, aggregated in 5 minutes intervals over four
    months between March 2012 and June 2012.

    A benchmark dataset for traffic forecasting as described in
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting" <https://arxiv.org/abs/1707.01926>`_.

    Dataset information:
        + Time steps: 34272
        + Nodes: 207
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 8.11%

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """
    url = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"

    similarity_options = {'distance'}

    def __init__(self, root=None, impute_zeros=True, freq=None):
        # set root path
        self.root = root
        # load dataset
        df, dist, mask = self.load(impute_zeros=impute_zeros)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="MetrLA")
        self.add_covariate('dist', dist, pattern='n n')

    @property
    def raw_file_names(self):
        return [
            'metr_la.h5', 'distances_la.csv', 'sensor_locations_la.csv',
            'sensor_ids_la.txt'
        ]

    @property
    def required_file_names(self):
        return ['metr_la.h5', 'metr_la_dist.npy']

    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)

    def build(self) -> None:
        self.maybe_download()
        # Build distance matrix
        logger.info('Building distance matrix...')
        raw_dist_path = os.path.join(self.root_dir, 'distances_la.csv')
        distances = pd.read_csv(raw_dist_path)
        ids_path = os.path.join(self.root_dir, 'sensor_ids_la.txt')
        with open(ids_path) as f:
            ids = f.read().strip().split(',')
        num_sensors = len(ids)
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
        # Builds sensor id to index map.
        sensor_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}
        # Fills cells in the matrix with distances.
        for row in distances.values:
            if row[0] not in sensor_to_ind or row[1] not in sensor_to_ind:
                continue
            dist[sensor_to_ind[row[0]], sensor_to_ind[row[1]]] = row[2]
        # Save to built directory
        path = os.path.join(self.root_dir, 'metr_la_dist.npy')
        np.save(path, dist)
        # Remove raw data
        self.clean_downloads()

    def load_raw(self):
        self.maybe_build()
        # load traffic data
        traffic_path = os.path.join(self.root_dir, 'metr_la.h5')
        df = pd.read_hdf(traffic_path)
        # add missing values
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0],
                                   datetime_idx[-1],
                                   freq='5T')
        df = df.reindex(index=date_range)
        # load distance matrix
        path = os.path.join(self.root_dir, 'metr_la_dist.npy')
        dist = np.load(path)
        return df, dist

    def load(self, impute_zeros=True):
        df, dist = self.load_raw()
        mask = (df.values != 0.).astype('uint8')
        if impute_zeros:
            df = df.replace(to_replace=0., method='ffill')
        return df, dist, mask

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
