import os

import numpy as np
import pandas as pd

from tsl import logger
from tsl.ops.similarities import gaussian_kernel
from tsl.utils import download_url, extract_zip
from .prototypes import PandasDataset


class PemsBay(PandasDataset):
    """A benchmark dataset for traffic forecasting as described in
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_

    The dataset contains 6 months of traffic readings from 01/01/2017 to
    05/31/2017 collected every 5 minutes by 325 traffic sensors in San Francisco
    Bay Area. The measurements are provided by California Transportation
    Agencies (CalTrans) Performance Measurement System (PeMS).
    """

    url = "https://drive.switch.ch/index.php/s/5NPcgGFAIJ4oFcT/download"

    similarity_options = {'distance', 'stcn'}
    temporal_aggregation_options = {'mean', 'nearest'}
    spatial_aggregation_options = None

    def __init__(self, root=None, freq='5T'):
        # Set root path
        self.root = root
        # load dataset
        df, dist, mask = self.load()
        super().__init__(dataframe=df,
                         mask=mask,
                         attributes=dict(dist=dist),
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="PemsBay")

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
        # add missing values
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5T')
        df = df.reindex(index=date_range)
        # load distance matrix
        path = os.path.join(self.root_dir, 'pems_bay_dist.npy')
        dist = np.load(path)
        return df.astype('float32'), dist

    def load(self):
        df, dist = self.load_raw()
        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        return df, dist, mask.astype('uint8')

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

    # Old methods, still to be checked

    def get_datetime_dummies(self):
        df = self.dataframe()
        df['day'] = df.index.weekday
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        dummies = pd.get_dummies(df[['day', 'hour', 'minute']],
                                 columns=['day', 'hour', 'minute'])
        return dummies.values

    def get_datetime_features(self):
        df = pd.DataFrame(index=self.df.index)
        idx: pd.DatetimeIndex = df.index

        # encode hour of the day
        second_of_the_day = idx.hour * 60 * 60 + idx.minute * 60 + idx.second
        seconds_per_day = 24 * 60 * 60
        df['day_sin'] = np.sin(
            second_of_the_day * (2 * np.pi / seconds_per_day))
        df['day_cos'] = np.cos(
            second_of_the_day * (2 * np.pi / seconds_per_day))

        return df[['day_sin', 'day_cos']].values

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window],
                idx[test_start:]]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = PemsBay()
    plt.imshow(dataset.mask, aspect='auto')
    plt.show()
