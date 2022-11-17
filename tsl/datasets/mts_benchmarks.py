import pandas as pd
import os

import tsl
from tsl.utils import download_url
from tsl.datasets.prototypes import DatetimeDataset


class _MTSBenchmarkDataset(DatetimeDataset):
    """Abstract class for loading datasets from
    https://github.com/laiguokun/multivariate-time-series-data.

    Args:
        root: Root folder for data download.
        freq: Resampling frequency.
    """
    url = None
    default_similarity_score = None
    default_spatial_aggregation = None
    default_temporal_aggregation = None
    default_freq = None
    start_date = None

    def __init__(self,
                 root=None,
                 freq=None):
        self.root = root
        df, mask = self.load()
        super().__init__(target=df, mask=mask, freq=freq,
                         similarity_score=self.default_similarity_score,
                         temporal_aggregation=self.default_temporal_aggregation,
                         spatial_aggregation=self.default_spatial_aggregation,
                         name=self.__class__.__name__)

    @property
    def required_file_names(self):
        return [f'{self.__class__.__name__}.h5']

    def download(self) -> None:
        download_url(self.url, self.root_dir)

    def build(self):
        # Build dataset
        self.maybe_download()
        tsl.logger.info(f"Building the {self.__class__.__name__} dataset...")
        df = pd.read_csv(self.raw_files_paths[0],
                         index_col=False,
                         header=None,
                         sep=',',
                         compression='gzip')
        index = pd.date_range(start=self.start_date, periods=len(df),
                              freq=self.default_freq)
        df = df.set_index(index)
        path = os.path.join(self.root_dir, f'{self.__class__.__name__}.h5')
        df.to_hdf(path, key='raw')
        self.clean_downloads()

    def load_raw(self) -> pd.DataFrame:
        self.maybe_build()
        df = pd.read_hdf(self.required_files_paths[0])
        return df

    def load(self):
        df = self.load_raw()
        tsl.logger.info('Loaded raw dataset.')
        mask = (df.values != 0.).astype('uint8')
        return df, mask


class ElectricityBenchmark(_MTSBenchmarkDataset):
    """Electricity consumption (in kWh) measured hourly by 321 sensors from
    2012 to 2014.

    Imported from https://github.com/laiguokun/multivariate-time-series-data.
    The `original dataset <https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014>`_
    records values in kW for 370 nodes starting from 2011, with part of the
    nodes with missing values before 2012. For the original dataset refer to
    :class:`~tsl.datasets.Elergone`.

    Dataset information:
        + Time steps: 26304
        + Nodes: 321
        + Channels: 1
        + Sampling rate: 1 hour
        + Missing values: 1.09%
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/electricity/electricity.txt.gz?raw=true'

    similarity_options = None

    default_similarity_score = None
    default_temporal_aggregation = 'sum'
    default_spatial_aggregation = 'sum'
    default_freq = '1H'
    start_date = '01-01-2012 00:00'

    @property
    def raw_file_names(self):
        return ['electricity.txt.gz']


class TrafficBenchmark(_MTSBenchmarkDataset):
    """A collection of hourly road occupancy rates (between 0 and 1) measured
    by 862 sensors for 48 months (2015-2016) on San Francisco Bay Area freeways.

    Imported from https://github.com/laiguokun/multivariate-time-series-data,
    raw data at `California Department of Transportation <https://pems.dot.ca.gov>`_.

    Dataset information:
        + Time steps: 17544
        + Nodes: 862
        + Channels: 1
        + Sampling rate: 1 hour
        + Missing values: 0.90%
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/traffic/traffic.txt.gz?raw=true'

    similarity_options = None

    default_similarity_score = None
    default_temporal_aggregation = 'mean'
    default_spatial_aggregation = 'mean'
    default_freq = '1H'
    start_date = '01-01-2015 00:00'

    @property
    def raw_file_names(self):
        return ['traffic.txt.gz']


class SolarBenchmark(_MTSBenchmarkDataset):
    """Solar power production records in the year of 2006, is sampled every 10
    minutes from 137 synthetic PV farms in Alabama State.
    The mask denotes 55.10% of data corresponding to daily hours with nonzero
    power production.

    Imported from https://github.com/laiguokun/multivariate-time-series-data,
    raw data at https://www.nrel.gov/grid/solar-power-data.html.

    Dataset information:
        + Time steps: 52560
        + Nodes: 137
        + Channels: 1
        + Sampling rate: 10 minutes
        + Missing values: 0.00%
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/solar-energy/solar_AL.txt.gz?raw=true'

    similarity_options = None

    default_similarity_score = None
    default_temporal_aggregation = 'mean'
    default_spatial_aggregation = 'sum'
    default_freq = '10T'
    start_date = '01-01-2006 00:00'

    @property
    def raw_file_names(self):
        return ['solar_AL.txt.gz']


class ExchangeBenchmark(_MTSBenchmarkDataset):
    """The collection of the daily exchange rates of eight foreign countries
    including Australia, British, Canada, Switzerland, China, Japan, New
    Zealand and Singapore ranging from 1990 to 2016.

    Imported from https://github.com/laiguokun/multivariate-time-series-data.

    Dataset information:
        + Time steps: 7588
        + Nodes: 8
        + Channels: 1
        + Sampling rate: 1 day
        + Missing values: 0.00%
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/exchange_rate/exchange_rate.txt.gz?raw=true'

    similarity_options = None

    default_similarity_score = None
    default_temporal_aggregation = 'mean'
    default_spatial_aggregation = None
    default_freq = '1D'
    start_date = '01-01-1990'

    @property
    def raw_file_names(self):
        return ['exchange_rate.txt.gz']
