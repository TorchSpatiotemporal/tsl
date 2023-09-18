import os
from typing import Dict, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..utils import download_url, extract_zip
from .prototypes import DatetimeDataset
from .prototypes.casting import to_pandas_freq

__base_url__ = "https://drive.switch.ch/index.php/s/nJgK7ca28hk7AMU/download"
__subsets__ = ["CA", "GBA", "GLA", "SD"]
SubsetType = Literal["CA", "GBA", "GLA", "SD"]


class LargeST(DatetimeDataset):
    r"""LargeST is a large-scale traffic forecasting dataset containing 5 years
    of traffic readings from 01/01/2017 to 12/31/2021 collected every 5 minutes
    by 8600 traffic sensors in California.

    Given the large number of sensors in the dataset, there are 3 subsets of
    sensors that can be selected:

    + :obj:`GLA` (Greater Los Angeles)
        + Nodes: 3834
        + Edges: 98703
        + District: 7, 8, 12

    + :obj:`GBA` (Greater Bay Area)
        + Nodes: 2352
        + Edges: 61246
        + District: 4

    + :obj:`SD` (San Diego)
        + Nodes: 716
        + Edges: 17319
        + District: 11

    By default, the full dataset :obj:`CA` is loaded, corresponding to the
    whole California.

    The measurements are provided by California Transportation Agencies
    (CalTrans) Performance Measurement System (PeMS). Introduced in the paper
    `"LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting"
    <https://arxiv.org/abs/2306.08259>`_ (Liu et al., 2023),
    where only readings from 2019 are considered, aggregated into 15-minutes
    intervals.

    Dataset information:
        + Time steps: 525888
        + Nodes: 8600
        + Edges: 201363
        + Channels: 1
        + Sampling rate: 5 minutes
        + Missing values: 1.51%

    Static attributes:
        + :obj:`metadata`: storing for each node:
            + ``lat``: latitude of the sensor;
            + ``lon``: longitude of the sensor;
            + ``district``: California's district where sensor is located (one
              of ``3``, ``4``, ``5``, ``6``, ``7``, ``8``, ``10``, ``11``,
              ``12``);
            + ``county``: California's county where sensor is located;
            + ``fwy_id``: id of highway where a sensor is located;
            + ``n_lanes``: the number of lanes in correspondence to the sensor
              (max 8);
            + ``direction``: direction of the highway measured by the sensor
              (one of ``N``, ``S``, ``E``, ``W``).
        + :obj:`adj`: weighted adjacency matrix
          :math:`\mathbf{A} \in \mathbb{R}^{N \times N}` built using road
          distances.

    Args:
        root (str, optional): The root directory where data will be downloaded
            and stored. If :obj:`None`, then defaults to :obj:`.storage` folder
            inside :tsl:`null` tsl's root directory.
            (default: :obj:`None`)
        subset (str): The subset to be loaded. Must be one of :obj:`"CA"`,
            :obj:`"GLA"`, :obj:`"GBA"`, :obj:`"SD"`.
            (default: :obj:`"CA"`)
        year (int or list): The year(s) to be loaded. Must be (a list) in
            :obj:`[2017, 2021]`. Note that raw data are divided by year and
            only requested years are downloaded.
            (default: :obj:`2019`)
        imputation_mode (str, optional): How to impute missing values. If
            :obj:`"nearest"`, then use nearest observation; if :obj:`"zero"`,
            fill missing values with :obj:`0`; if :obj:`None`, do not impute
            (leave :obj:`nan`).
            (default: :obj:`"zero"`)
        freq (str): The sampling rate used for resampling (e.g., :obj:`"15T"`
            for 15-minutes intervals resampling).
            (default: :obj:`"15T"`)
        precision (int or str): The float precision of the dataset.
            (default: :obj:`32`)
    """
    base_url = __base_url__
    url = {
        "2017": __base_url__ + "?path=%2F2017&files=data.h5",
        "2018": __base_url__ + "?path=%2F2018&files=data.h5",
        "2019": __base_url__ + "?path=%2F2019&files=data.h5",
        "2020": __base_url__ + "?path=%2F2020&files=data.h5",
        "2021": __base_url__ + "?path=%2F2021&files=data.h5",
        "sensors": __base_url__ + "?files=sensors.zip",
    }

    similarity_options = {"precomputed"}

    def __init__(self,
                 root: str = None,
                 subset: SubsetType = "CA",
                 year: Optional[Union[int, Sequence[int]]] = 2019,
                 imputation_mode: Literal["nearest", "zero", None] = "zero",
                 freq: str = "15T",
                 precision: Union[int, str] = 32):
        # set root path
        self.root = root

        subset = subset.upper()
        if subset not in __subsets__:
            raise ValueError(
                f"Incorrect choice for 'subset' ({subset}). "
                f"Available options are {', '.join(__subsets__)}.")
        self.subset = subset

        view_years = years_set = set(range(2017,
                                           2022))  # between 2017 and 2021
        if year is not None:
            year = {year} if isinstance(year, int) else set(year)
            view_years = view_years.intersection(year)
            if not len(view_years):
                raise ValueError(f"Incorrect choice for 'year' ({year}). "
                                 f"Must be a subset of {years_set}.")
        self.years = sorted(view_years)

        self.imputation_mode = imputation_mode
        assert imputation_mode in ["nearest", "zero", None]

        # Set dataset frequency here to resample when loading
        if freq is not None:
            freq = to_pandas_freq(freq)
        self.freq = freq

        # load dataset
        readings, mask, metadata, adj = self.load()
        covariates = {"metadata": (metadata, 'n f'), "adj": (adj, 'n n')}
        super().__init__(target=readings,
                         freq=freq,
                         mask=mask,
                         covariates=covariates,
                         similarity_score="precomputed",
                         temporal_aggregation="mean",
                         spatial_aggregation="mean",
                         name=f"LargeST-{subset}",
                         precision=precision)

    @property
    def raw_file_names(self) -> Dict[str, str]:
        out = {
            str(year): os.path.join(str(year), "data.h5")
            for year in self.years
        }
        out["metadata"] = os.path.join("sensors", "metadata.csv")
        out["adj"] = os.path.join("sensors", "adj.npz")
        return out

    def download(self) -> None:
        for key, filepath in self.raw_files_paths.items():
            # download only required data that are missing
            if not os.path.exists(filepath):
                # "metadata" and "adj" are inside single .zip file
                if key in ["metadata", "adj"]:
                    sub_dir = os.path.dirname(filepath)
                    os.makedirs(sub_dir, exist_ok=True)
                    # download, extract, and remove .zip file
                    in_dir = download_url(self.url["sensors"],
                                          sub_dir,
                                          filename="sensors.zip")
                    extract_zip(in_dir, sub_dir)
                    os.unlink(in_dir)
                else:  # download directly .h5 file containing readings per year
                    sub_dir, filename = os.path.split(filepath)
                    os.makedirs(sub_dir, exist_ok=True)
                    download_url(self.url[key], sub_dir, filename)

    def load_raw(self):
        self.maybe_download()

        filenames = self.required_files_paths

        # load sensors information
        metadata = pd.read_csv(filenames["metadata"], index_col=0)
        max_nodes = len(metadata)

        # possibly select subset, "CA" stands for no subset (whole California)
        node_mask = slice(None)
        if self.subset == "GLA":  # Greater Los Angeles
            node_mask = ((metadata.District == 7) | (metadata.District == 8) |
                         (metadata.District == 12)).values
        elif self.subset == "GBA":  # Greater Bay Area
            node_mask = (metadata.District == 4).values
        elif self.subset == "SD":  # San Diego
            node_mask = (metadata.District == 11).values
        metadata = metadata.loc[node_mask]

        # load traffic data only for requested years
        readings = []
        for year in self.years:
            data_path = filenames[str(year)]
            data_df = pd.read_hdf(data_path, key="readings")
            data_df = data_df.loc[:, node_mask]  # filter subset
            # resample here to aggregate only valid observations and
            # align to authors' preprocessing
            if self.freq is not None:
                data_df = data_df.resample(self.freq).mean()
                # in authors' code: data_df.resample('15T').mean().round(0)
            readings.append(data_df)

        readings = (
            readings[0] if len(readings) == 1  # avoid useless
            else pd.concat(readings, axis=0))  # computations

        # load adjacency
        edge_index, edge_weight = np.load(filenames["adj"]).values()
        # build square adj from coo to add adj as covariate
        adj = np.eye(max_nodes, dtype=np.float32)
        adj[tuple(edge_index)] = edge_weight
        adj = adj[node_mask][:, node_mask]

        return readings, metadata, adj

    def load(self):
        readings, metadata, adj = self.load_raw()
        # impute missing observations using last observed values
        # in authors' code: readings = readings.fillna(0)
        mask = ~readings.isna().values
        if self.imputation_mode == "nearest":
            readings = readings.ffill().bfill()
        elif self.imputation_mode == "zero":
            readings = readings.fillna(0)
        return readings, mask, metadata, adj

    def compute_similarity(self, method: str, **kwargs):
        if method == "precomputed":
            # load precomputed adjacency matrix based on road distance
            return self.adj
