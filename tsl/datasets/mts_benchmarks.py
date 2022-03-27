
import pandas as pd
import os

import tsl
from tsl.utils import download_url
from tsl.datasets.prototypes import PandasDataset


class _MTSBenchmarkDataset(PandasDataset):
    """
    Abstract class for loading datasets from https://github.com/laiguokun/multivariate-time-series-data
    """
    url = None
    default_similarity_score = None
    default_spatial_aggregation = None
    default_temporal_aggregation = None
    default_freq = None
    start_date = '01-01-12 00:00'

    def __init__(self,
                 root=None,
                 freq=None):
        """

        Args:
            root: Root folder for data download.
            freq: Resampling frequency.
        """
        self.root = root
        df, mask = self.load()
        super().__init__(dataframe=df,
                         mask=mask,
                         freq=freq,
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
        index = pd.date_range(start=self.start_date, periods=len(df), freq=self.default_freq)
        df = df.set_index(index)
        path = os.path.join(self.root_dir, f'{self.__class__.__name__}.h5')
        df.to_hdf(path, key='raw')
        self.clean_downloads()
        return df

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
    r"""
    From https://github.com/laiguokun/multivariate-time-series-data :

    The raw dataset is in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.
    It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014.
    Because the some dimensions are equal to 0. So we eliminate the records in 2011.
    Final we get data contains electricity consumption of 321 clients from 2012 to 2014.
    And we converted the data to reflect hourly consumption.
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/electricity/electricity.txt.gz?raw=true'

    similarity_options = None
    temporal_aggregation_options = {'sum'}
    spatial_aggregation_options = {'sum'}

    default_similarity_score = None
    default_temporal_aggregation = 'sum'
    default_spatial_aggregation = 'sum'
    default_freq = '1H'
    start_date = '01-01-2001 00:00'

    @property
    def raw_file_names(self):
        return ['electricity.txt.gz']


class TrafficBenchmark(_MTSBenchmarkDataset):
    r"""
    From https://github.com/laiguokun/multivariate-time-series-data :

    The raw data is in http://pems.dot.ca.gov. The data in this repo is a collection of 48 months (2015-2016) hourly
    data from the California Department of Transportation.
    The data describes the road occupancy rates (between 0 and 1) measured by different sensors on San Francisco Bay area
    freeways.
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/traffic/traffic.txt.gz?raw=true'

    similarity_options = None
    temporal_aggregation_options = {'mean'}
    spatial_aggregation_options = {'mean'}

    default_similarity_score = None
    default_temporal_aggregation = 'mean'
    default_spatial_aggregation = 'mean'
    default_freq = '1H'
    start_date = '01-01-2015 00:00'

    @property
    def raw_file_names(self):
        return ['traffic.txt.gz']


class SolarBenchmark(_MTSBenchmarkDataset):
    r"""
    From https://github.com/laiguokun/multivariate-time-series-data :

    The raw data is in http://www.nrel.gov/grid/solar-power-data.html .
    It contains the solar power production records in the year of 2006, which is sampled every 10 minutes from 137 PV
    plants in Alabama State.
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/solar-energy/solar_AL.txt.gz?raw=true'

    similarity_options = None
    temporal_aggregation_options = {'mean'}
    spatial_aggregation_options = {'sum'}

    default_similarity_score = None
    default_temporal_aggregation = 'mean'
    default_spatial_aggregation = 'sum'
    default_freq = '10T'
    start_date = '01-01-2006 00:00'

    @property
    def raw_file_names(self):
        return ['solar_AL.txt.gz']


class ExchangeBenchmark(_MTSBenchmarkDataset):
    r"""
    From https://github.com/laiguokun/multivariate-time-series-data :

    The collection of the daily exchange rates of eight foreign countries including Australia, British, Canada,
    Switzerland, China, Japan, New Zealand and Singapore ranging from 1990 to 2016.
    """
    url = 'https://github.com/TorchSpatiotemporal/multivariate-time-series-data/blob/master/exchange_rate/exchange_rate.txt.gz?raw=true'

    similarity_options = None
    temporal_aggregation_options = {'mean'}
    spatial_aggregation_options = None

    default_similarity_score = None
    default_temporal_aggregation = 'mean'
    default_spatial_aggregation = None
    default_freq = '1D'

    @property
    def raw_file_names(self):
        return ['exchange_rate.txt.gz']

