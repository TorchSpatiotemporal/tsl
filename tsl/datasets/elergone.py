import os
from typing import Optional

import numpy as np
import pandas as pd

import tsl
from tsl.datasets.prototypes import DatetimeDataset, casting
from tsl.ops import similarities as sims
from tsl.utils import download_url, extract_zip


class Elergone(DatetimeDataset):
    """Load profiles of 370 points collected every 15 minutes from 2011 to 2014.

    Raw data at
    https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014.
    The :obj:`load` method loads the values in kWh, computes the mask for the
    zero values and pads the missing steps.

    From the original description:

        Values in the original dataframe are in kW of each 15 min. To convert
        values in kWh values must be divided by 4. Each column represent one
        client. Some clients were created after 2011. In these cases
        consumption were considered zero. All time labels report to Portuguese
        hour. However, all days present 96 measures (24*4). Every year in March
        time change day (which has only 23 hours) the values between 1:00 am
        and 2:00 am are zero for all points. Every year in October time change
        day (which has 25 hours) the values between 1:00 am and 2:00 am
        aggregate the consumption of two hours.

    Dataset size:
        + Time steps: 140256
        + Nodes: 370
        + Channels: 1
        + Sampling rate: 15 minutes
        + Missing values: 20.15%

    Args:
        root: Root folder for data download.
        freq: Resampling frequency.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"  # noqa

    similarity_options = {'correntropy', 'pearson'}

    def __init__(self, root=None, freq=None):
        self.root = root
        df, mask = self.load()
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score='correntropy',
                         temporal_aggregation='sum',
                         spatial_aggregation='sum',
                         name='Electricity')

    @property
    def raw_file_names(self):
        return ['LD2011_2014.csv']

    @property
    def required_file_names(self):
        return ['elergone.h5']

    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)
        path = path.replace('.zip', '')
        os.rename(path, path.replace('.txt', '.csv'))
        self.clean_root_dir()

    def build(self):
        # Build dataset
        self.maybe_download()
        tsl.logger.info("Building the electricity dataset...")
        path = os.path.join(self.root_dir, 'LD2011_2014.csv')
        df = pd.read_csv(path,
                         sep=';',
                         index_col=0,
                         parse_dates=True,
                         decimal=',')

        df.index.freq = df.index.inferred_freq
        path = os.path.join(self.root_dir, 'elergone.h5')
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
        # idx = sorted(df.index)
        # start, end = idx[0], idx[-1]
        # idx = pd.date_range(start, end, freq='15T')
        # df = df.reindex(index=idx)
        df /= 4.  # kW -> kWh
        # drop duplicates
        df = df[~df.index.duplicated(keep='first')]
        df = df.fillna(0.)
        mask = (df.values != 0.).astype('uint8')
        return df, mask

    def compute_similarity(self,
                           method: str,
                           gamma=10,
                           trainlen=None,
                           **kwargs) -> Optional[np.ndarray]:
        train_df = self.dataframe()
        mask = self.mask
        if trainlen is not None:
            train_df = self.dataframe().iloc[:trainlen]
            mask = mask[:trainlen]

        x = np.asarray(train_df) * mask[..., -1]
        if method == 'correntropy':
            period = casting.to_pandas_freq('1D').nanos // self.freq.nanos
            x = (x - x.mean()) / x.std()
            sim = sims.correntropy(x, period=period, mask=mask, gamma=gamma)
        else:
            raise NotImplementedError
        return sim
