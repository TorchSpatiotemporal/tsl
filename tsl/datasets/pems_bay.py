import os

import numpy as np
import pandas as pd

from tsl import logger
from tsl.ops.similarities import gaussian_kernel
from tsl.utils import download_url, extract_zip

from .prototypes import DatetimeDataset


class PemsBay(DatetimeDataset):
    r"""The dataset contains 6 months of traffic readings from 01/01/2017 to
    05/31/2017 collected every 5 minutes by 325 traffic sensors in San Francisco
    Bay Area.

    The measurements are provided by California Transportation Agencies
    (CalTrans) Performance Measurement System (PeMS). A benchmark dataset for
    traffic forecasting as described in
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting" <https://arxiv.org/abs/1707.01926>`_.

    Dataset information:
        + Time steps: 52128
        + Nodes: 325
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 0.02%

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """

    url = "https://drive.switch.ch/index.php/s/5NPcgGFAIJ4oFcT/download"

    similarity_options = {'distance', 'stcn'}

    def __init__(self, mask_zeros: bool = True, root=None, freq=None):
        # Set root path
        self.root = root
        self.mask_zeros = mask_zeros
        # load dataset
        df, dist, mask = self.load(mask_zeros)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="PemsBay")
        self.add_covariate('dist', dist, pattern='n n')

    @property
    def raw_file_names(self):
        return ['pems_bay.h5', 'distances_bay.csv', 'sensor_locations_bay.csv']

    @property
    def required_file_names(self):
        return ['pems_bay.h5', 'pems_bay_dist.npy']

    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)

    def build(self) -> None:
        self.maybe_download()
        # Build distance matrix
        path = os.path.join(self.root_dir, 'pems_bay.h5')
        ids = list(pd.read_hdf(path).columns)
        self.build_distance_matrix(ids)
        # Remove raw data
        self.clean_downloads()

    def load_raw(self):
        self.maybe_build()
        # load traffic data
        traffic_path = os.path.join(self.root_dir, 'pems_bay.h5')
        df = pd.read_hdf(traffic_path)
        # add missing values (index is sorted)
        date_range = pd.date_range(df.index[0], df.index[-1], freq='5T')
        df = df.reindex(index=date_range)
        # load distance matrix
        path = os.path.join(self.root_dir, 'pems_bay_dist.npy')
        dist = np.load(path)
        return df.astype('float32'), dist

    def load(self, mask_zeros: bool = True):
        df, dist = self.load_raw()
        mask = ~np.isnan(df.values)
        if mask_zeros:
            mask &= df.values != 0
        df.fillna(method='ffill', axis=0, inplace=True)
        return df, dist, mask

    def build_distance_matrix(self, ids):
        logger.info('Building distance matrix...')
        raw_dist_path = os.path.join(self.root_dir, 'distances_bay.csv')
        distances = pd.read_csv(raw_dist_path)
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
        path = os.path.join(self.root_dir, 'pems_bay_dist.npy')
        np.save(path, dist)
        return dist

    def compute_similarity(self, method: str, **kwargs):
        if method == 'distance':
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
        elif method == 'stcn':
            sigma = 10
            return gaussian_kernel(self.dist, sigma)
