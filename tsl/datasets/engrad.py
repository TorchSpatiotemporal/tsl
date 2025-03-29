import math
import os
from typing import List, Optional, Union, Literal

import numpy as np
import pandas as pd
from tsl.data import Splitter
from tsl.data.datamodule.splitters import indices_between
from tsl.datasets.prototypes import DatetimeDataset
from tsl.utils import download_url, extract_zip, ensure_list


class EngRadSplitter(Splitter):

    def __init__(self,
                 val_len: int = None,
                 val_seq_len: int = 7,
                 first_val_step=(2019, 1, 1),
                 first_test_step=(2020, 1, 1)):
        super(EngRadSplitter, self).__init__()
        self._val_len = val_len
        self.val_seq_len = val_seq_len
        self.first_val_step = first_val_step
        self.first_test_step = first_test_step

    def fit(self, dataset):
        # Get test indices
        test_idxs = indices_between(dataset, first_ts=self.first_test_step)
        # Get validation indices
        if self.first_val_step is not None:
            val_idxs = indices_between(dataset, first_ts=self.first_val_step,
                                       last_ts=self.first_test_step)
        else:
            val_idxs = np.setdiff1d(np.arange(len(dataset)), test_idxs)
        # Remove validation indices overlapping with test indices
        ovl_idxs, _ = dataset.overlapping_indices(val_idxs,
                                                  test_idxs,
                                                  synch_mode='window',
                                                  as_mask=True)
        val_idxs = val_idxs[~ovl_idxs]
        ovl_idxs, _ = dataset.overlapping_indices(val_idxs,
                                                  test_idxs,
                                                  synch_mode='horizon',
                                                  as_mask=True)
        val_idxs = val_idxs[~ovl_idxs]
        # Sparsify validation set according to val_len
        val_len = self._val_len
        if val_len < 1:
            val_len = int(val_len * len(val_idxs))
        # Take sparse sequences of self.val_seq_len
        num_seq = math.ceil(val_len / self.val_seq_len)
        seq_len = len(val_idxs) // num_seq
        val_seq_start = seq_len - self.val_seq_len
        seq_start_idx = val_idxs[val_seq_start::seq_len]
        val_seq_idxs = np.ravel(seq_start_idx[:, None] +
                                np.arange(self.val_seq_len))
        # Remove possibly out-of-bounds indices
        val_idxs = np.intersect1d(val_seq_idxs, val_idxs)
        # Use all other indices for training
        train_idxs = np.arange(val_idxs[-1])
        ovl_idxs, _ = dataset.overlapping_indices(train_idxs,
                                                  val_idxs,
                                                  synch_mode='window',
                                                  as_mask=True)
        train_idxs = train_idxs[~ovl_idxs]
        ovl_idxs, _ = dataset.overlapping_indices(train_idxs,
                                                  val_idxs,
                                                  synch_mode='horizon',
                                                  as_mask=True)
        train_idxs = train_idxs[~ovl_idxs]
        self.set_indices(train_idxs, val_idxs, test_idxs)


class EngRad(DatetimeDataset):
    r"""The EngRAD dataset from the paper `"Graph-based Forecasting with
    Missing Data through Spatiotemporal Downsampling"
    <https://arxiv.org/abs/2402.10634>`_ (Marisca et al., ICML 2024).
    
    The dataset consists of weather measurements collected hourly in 722 cities 
    spread across England from 2018 to 2020. The dataset is available through
    `Zenodo <https://zenodo.org/records/12760772>`_.

    Data provider: https://open-meteo.com/

    Dataset size:
        + Time steps: 26304
        + Nodes: 487
        + Channels: 5
        + Sampling rate: 1 hour
        + Missing values: 0.00%

    Channels:
        + ``temperature_2m``: Air temperature at 2 meters above ground (°C).
          Instant.
        + ``relative_humidity_2m``: Relative humidity at 2 meters above ground
          (%). Instant.
        + ``precipitation``: Total precipitation (rain, showers, snow) sum of
          the preceding hour (mm). Preceding hour sum.
        + ``cloud_cover``: Total cloud cover as an area fraction (%). Instant.
        + ``shortwave_radiation``: Global horizontal irradiation (GHI) (W/m²).
          Preceding hour mean.

    Static attributes:
        + :obj:`metadata`: information associated to the locations.
        + :obj:`distances`: :math:`N \times N` matrix of pairwise distances
          between the locations.
    """
    url = "https://zenodo.org/records/12760772/files/data.h5?download=1"

    similarity_options = {'distance', 'grid'}

    def __init__(self,
                 root: str = None,
                 target_channels: Optional[Union[str, List[str]]] = 'all',
                 covariate_channels: Optional[Union[str, List[str]]] = None,
                 mask_zero_radiance: bool = False,
                 precipitation_unit: Literal["mm", "cm"] = "mm",
                 freq: Optional[str] = None):
        self.root = root
        self.mask_zero_radiance = mask_zero_radiance
        self.precipitation_unit = precipitation_unit
        # Load data
        df, metadata, dist, mask = self.load(self.mask_zero_radiance)
        # Set covariates
        covariates = dict(metadata=(metadata, 'n f'),
                          distances=(dist, 'n n'))
        # Optionally filter channels
        target = df
        if target_channels is not None and target_channels != 'all':
            target_channels = ensure_list(target_channels)
            nodes = metadata.index
            columns = pd.MultiIndex.from_product([nodes, target_channels])
            target = df.loc[:, columns]
            if mask is not None:
                mask = mask.loc[:, columns]
        # Optionally add covariates
        if covariate_channels == 'all':
            covariates['u'] = (df, 't n f')
        elif covariate_channels is not None:
            covariate_channels = ensure_list(covariate_channels)
            nodes = metadata.index
            columns = pd.MultiIndex.from_product([nodes, covariate_channels])
            covariates['u'] = (df.loc[:, columns], 't n f')

        super().__init__(target=target,
                         mask=mask,
                         covariates=covariates,
                         freq=freq,
                         similarity_score='distance',
                         temporal_aggregation='mean',
                         spatial_aggregation='mean',
                         name='EngRad')

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.h5']

    @property
    def required_file_names(self) -> dict[str, str]:
        return {'data': self.raw_file_names[0], 'distances': 'dist.npy'}

    def download(self):
        download_url(self.url, self.root_dir, 'data.h5')

    def build(self):
        self.maybe_download()
        # compute distances from latitude and longitude degrees
        path = self.required_files_paths['data']
        metadata = pd.DataFrame(pd.read_hdf(path, 'metadata'))
        coords = metadata.loc[:, ['lat', 'lon']]
        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(coords, to_rad=True).values
        np.save(self.required_files_paths['distances'], dist)

    def load_raw(self):
        self.maybe_build()
        df = pd.read_hdf(self.required_files_paths['data'], 'data')
        metadata = pd.read_hdf(self.required_files_paths['data'], 'metadata')
        dist = np.load(self.required_files_paths['distances'])
        return pd.DataFrame(df), metadata, dist

    def load(self, mask_zero_radiance: bool = False):
        df, metadata, dist = self.load_raw()
        if mask_zero_radiance:
            mask = pd.DataFrame(True, index=df.index, columns=df.columns)
            swr = df.loc[:, (slice(None), 'shortwave_radiation')] > 0
            mask.loc[swr.index, swr.columns] = swr
        else:
            mask = None
        if self.precipitation_unit == 'cm':
            df.loc[:, (slice(None), 'precipitation')] /= 10
        return df, metadata, dist, mask

    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'engrad':
            return EngRadSplitter(**kwargs)

    def compute_similarity(self, method: str, **kwargs):
        from tsl.ops.similarities import gaussian_kernel
        if method == "distance":
            theta = kwargs.get('theta', np.std(self.distances))
            return gaussian_kernel(self.distances, theta=theta)
        if method == "grid":
            dist = self.distances.copy()
            dist[dist > 16] = np.inf  # keep only grid edges
            theta = kwargs.get('theta', 20)
            return gaussian_kernel(dist, theta=theta)