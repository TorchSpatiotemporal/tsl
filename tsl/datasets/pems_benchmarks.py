import os
from pathlib import Path

import numpy as np
import pandas as pd

from tsl import logger
from tsl.datasets.prototypes import DatetimeDataset
from tsl.ops.similarities import gaussian_kernel
from tsl.utils import download_url, extract_zip


class _PeMS(DatetimeDataset):
    r"""Abstract class for PeMSD datasets."""

    url: None
    start_date: None
    similarity_options = {'distance', 'stcn', 'binary'}
    num_sensors: None
    name: None

    def __init__(self, mask_zeros: bool = False, root=None, freq=None):
        # Set root path
        self.root = root
        self.mask_zeros = mask_zeros
        # load dataset
        flow, occupancy, speed, dist, mask = self.load(mask_zeros)
        super().__init__(target=flow,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name=self.name)
        # todo : remove this hack
        if occupancy is not None:
            occupancy.columns = self.target.columns
            self.add_covariate('occupancy', occupancy, pattern='t n f')
        if speed is not None:
            speed.columns = self.target.columns
            self.add_covariate('speed', speed, pattern='t n f')
        self.add_covariate('dist', dist, pattern='n n')

    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)

    def build(self) -> None:
        # Build dataset
        self.maybe_download()
        self.build_distance_matrix(self.num_sensors)
        self.clean_downloads()

    def load_raw(self):
        self.maybe_build()
        fp = np.load(self.raw_files_paths[0])
        data = fp['data']
        fp.close()
        index = pd.date_range(start=self.start_date,
                              periods=len(data),
                              freq='5T')

        df_flow = pd.DataFrame(data=data[..., 0],
                               index=index).astype('float32')

        if data.shape[-1] > 1:
            df_occ = pd.DataFrame(data=data[..., 1],
                                  index=index).astype('float32')

            df_speed = pd.DataFrame(data=data[..., 2],
                                    index=index).astype('float32')
        else:
            df_occ = df_speed = None

        # load distance matrix
        path = os.path.join(self.root_dir, 'distance_matrix.npy')
        dist = np.load(path)
        return df_flow, \
            df_occ, \
            df_speed, \
            dist

    def load(self, mask_zeros: bool = True):
        *dfs, dist = self.load_raw()
        mask = None
        if mask_zeros:
            mask = dfs[0].values != 0
        return *dfs, dist, mask

    def build_distance_matrix(self, num_sensors):
        logger.info('Building distance matrix...')
        distances = pd.read_csv(self.raw_files_paths[1])
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
        # Fills cells in the matrix with distances.
        for row in distances.values:
            dist[int(row[0]), int(row[1])] = row[2]
        # Save to built directory
        path = os.path.join(self.root_dir, 'distance_matrix.npy')
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
        elif method == 'binary':
            return (~np.isinf(self.dist)).astype('float32')


class PeMS03(_PeMS):
    r"""The dataset contains 3 months of traffic readings from 09/01/2018 to
    11/30/2018 collected every 5 minutes by 358 traffic sensors.

    The measurements are provided by California Transportation Agencies
    (CalTrans) Performance Measurement System (PeMS). A benchmark dataset for
    traffic forecasting as described in the paper `"Learning Dynamics and
    Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting"
    <https://ieeexplore.ieee.org/document/9346058>`_ (Guo et al., 2021).

    Dataset information:
        + Time steps: 26208
        + Nodes: 358
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 0% (already imputed in the dataset)

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """
    name = 'PeMS03'
    start_date = '09-01-2018 00:00'
    num_sensors = 358
    url = 'https://drive.switch.ch/index.php/s/B5xDMtNs4M7pzsn/download'

    @property
    def raw_file_names(self):
        return ['pems03.npz', 'distances.csv', 'index.txt']

    @property
    def required_file_names(self):
        return ['pems03.npz', 'distance_matrix.npy', 'index.txt']

    def build_distance_matrix(self, num_sensors):
        logger.info('Building distance matrix...')
        raw_dist_path = os.path.join(self.root_dir, self.raw_files_paths[1])
        distances = pd.read_csv(raw_dist_path)
        ids = Path(os.path.join(self.root_dir,
                                'index.txt')).read_text().splitlines()
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
        sensor_to_idx = {int(sensor_id): i for i, sensor_id in enumerate(ids)}
        for row in distances.values:
            if row[0] not in sensor_to_idx or row[1] not in sensor_to_idx:
                continue
            dist[sensor_to_idx[row[0]], sensor_to_idx[row[1]]] = row[2]
        path = os.path.join(self.root_dir, 'distance_matrix.npy')
        np.save(path, dist)
        return dist


class PeMS04(_PeMS):
    r"""The dataset contains 2 months of traffic readings from 01/01/2018 to
    02/28/2018 collected every 5 minutes by 307 traffic sensors in San Francisco
    Bay Area.

    The measurements are provided by California Transportation Agencies
    (CalTrans) Performance Measurement System (PeMS). A benchmark dataset for
    traffic forecasting as described in the paper `"Learning Dynamics and
    Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting"
    <https://ieeexplore.ieee.org/document/9346058>`_ (Guo et al., 2021).

    The target variable is the total flow (number of detected vehicles).

    Dataset information:
        + Time steps: 16992
        + Nodes: 307
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 0% (already imputed in the dataset)

    Covariates:
        + :obj:`occupancy`: :math:`T \times N \times 1` Time series associated
          to the occupancy of the lanes.
        + :obj:`speed`: :math:`T \times N \times 1` Time series associated to
          average speed of the detected vehicles.

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """
    name = 'PeMS04'
    start_date = '01-01-2018 00:00'
    num_sensors = 307
    url = 'https://drive.switch.ch/index.php/s/swNbaB5rPrBmAZQ/download'

    @property
    def raw_file_names(self):
        return ['pems04.npz', 'distance.csv']

    @property
    def required_file_names(self):
        return ['pems04.npz', 'distance_matrix.npy']


class PeMS07(_PeMS):
    r"""The dataset contains 4 months of traffic readings from 05/01/2017 to
    08/31/2017 collected every 5 minutes by 883 traffic sensors.

    The measurements are provided by California Transportation Agencies
    (CalTrans) Performance Measurement System (PeMS). A benchmark dataset for
    traffic forecasting as described in the paper `"Learning Dynamics and
    Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting"
    <https://ieeexplore.ieee.org/document/9346058>`_ (Guo et al., 2021).

    Dataset information:
        + Time steps: 28224
        + Nodes: 883
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 0% (already imputed in the dataset)

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """
    name = 'PeMS07'
    start_date = '05-01-2017 00:00'
    num_sensors = 883
    url = 'https://drive.switch.ch/index.php/s/VcyirewUufrN57h/download'

    @property
    def raw_file_names(self):
        return ['pems07.npz', 'distance.csv']

    @property
    def required_file_names(self):
        return ['pems07.npz', 'distance_matrix.npy']


class PeMS08(_PeMS):
    r"""The dataset contains 2 months of traffic readings from 07/01/2016 to
    08/31/2016 collected every 5 minutes by 170 traffic sensors in San
    Bernardino.

    The measurements are provided by California Transportation Agencies
    (CalTrans) Performance Measurement System (PeMS). A benchmark dataset for
    traffic forecasting as described in the paper `"Learning Dynamics and
    Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting"
    <https://ieeexplore.ieee.org/document/9346058>`_ (Guo et al., 2021).

    Dataset information:
        + Time steps: 17856
        + Nodes: 170
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 0% (already imputed in the dataset)

    Covariates:
        + :obj:`occupancy`: :math:`T \times N \times 1` Time series associated
          to the occupancy of the lanes.
        + :obj:`speed`: :math:`T \times N \times 1` Time series associated to
          average speed of the detected vehicles.

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.
    """
    name = 'PeMS08'
    start_date = '07-01-2016 00:00'
    num_sensors = 170
    url = 'https://drive.switch.ch/index.php/s/AUGNn9Rx9zMz3vg/download'

    @property
    def raw_file_names(self):
        return ['pems08.npz', 'distance.csv']

    @property
    def required_file_names(self):
        return ['pems08.npz', 'distance_matrix.npy']
