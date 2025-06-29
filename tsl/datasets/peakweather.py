from typing import Literal, Optional, Union, Sequence, List

import numpy as np
import pandas as pd
from tsl.datasets.prototypes import DatetimeDataset
from tsl.ops.similarities import geographical_distance, gaussian_kernel
from tsl.utils import ensure_list

from peakweather import PeakWeatherDataset


class PeakWeather(DatetimeDataset):
    """The PeakWeather dataset from the paper `"PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning" <https://arxiv.org/abs/2506.13652>`_ (Zambon et al., 2025)

    The dataset consists of surface weather observations collected every 10 minutes over more than 8 years from the ground stations of the Federal Office of Meteorology and Climatology MeteoSwiss's measurement network. The dataset is available on `HuggingFace. <https://huggingface.co/datasets/MeteoSwiss/PeakWeather>`_

        Args:
            root (str, optional): Location of the dataset. 
                If not provided, the data will be downloaded in the current working directory. Defaults to None.
            target_channels (Union[str, List[str]], optional): Defines which channels (variables) are considered targets. 
                Defaults to "all".
            covariate_channels (Optional[Union[str, List[str]]], optional): Defines which channels (variables) are considered covariates. 
                Defaults to None.
            years (Optional[Union[int, Sequence[int]]], optional): Specifies the years to load. If not provided, all years are used. 
                Defaults to None.
            extended_topo_vars (Optional[Union[str, Sequence[str]]], optional): Specifies which static topographical variables to include. 
                Defaults to "none".
            imputation_method (Literal["locf", "zero", None], optional): Method used to impute missing values. 
                Defaults to "zero".
            interpolation_method (str, optional): Spatial interpolation method for topographical variables. 
                Defaults to "nearest".
            freq (str, optional): Frequency for resampling observations. If not provided, the original 10-minute resolution is used. 
                Defaults to None.
            station_type (Optional[Literal['rain_gauge', 'meteo_station']], optional): Type of stations to load. 
                If not provided, loads both rain gauge and meteorological station data. Defaults to None.
            extended_nwp_vars (Optional[List[str]], optional): Defines the NWP model baseline variables to include in the dataset. 
                Defaults to None.

    Dataset size:
        + Time steps: 433728
        + Nodes: 302
        + Channels: 10 (8 + 2 derivate)
        + Sampling rate: 10 min
        + Missing values: 0.00%

    Channels:
        + ``humidity``: Relative air humidity 2m above the ground. Current value. Unit: %.
        + ``precipitation``: Precipitation. Ten minutes total. Unit: mm.
        + ``pressure``: Atmospheric pressure at barometric altitude (QFE). Current value. Unit: hPa.
        + ``sunshine``: Sunshine duration. Ten minutes total. Unit: min.
        + ``temperature``: Air temperature 2m above the ground. Current value. Unit: °C.
        + ``wind_direction``: Wind direction. Ten minutes mean. Unit: °.
        + ``wind_gust``: Gust peak (one second). Ten minute maximum. Unit: m/s.
        + ``wind_speed``: Wind speed scalar. Ten minutes mean. Unit: m/s.
        + ``wind_u``: Eastward wind component (derivate from wind_speed and wind_direction). Ten minutes mean. Unit: m/s.
        + ``wind_v``: Northward wind component (derivate from wind_speed and wind_direction). Ten minutes mean. Unit: m/s.
        
    
    """ 
    
    base_url = PeakWeatherDataset.base_url
    available_years = PeakWeatherDataset.available_years
    available_topography = PeakWeatherDataset.available_topography
    similarity_options = {"distance"}

    available_channels = (
        *PeakWeatherDataset.available_parameters.keys(),
        "wind_u",
        "wind_v",
    )

    def __init__(self,
                 root: str = None,
                 target_channels: Union[str, List[str]] = "all",
                 covariate_channels: Optional[Union[str, List[str]]] = None,
                 years: Optional[Union[int, Sequence[int]]] = None,
                 extended_topo_vars: Optional[Union[str, Sequence[str]]] = "none",
                 imputation_method: Literal["locf", "zero", None] = "zero",
                 interpolation_method: str = "nearest",
                 freq: str = None,
                 station_type: Optional[Literal['rain_gauge', 'meteo_station']] = None,
                 extended_nwp_vars: Optional[List[str]] = None):

        channels = None
        if target_channels != "all" and covariate_channels != "other":
            channels = ensure_list(target_channels)
            if covariate_channels is not None:
                channels += ensure_list(covariate_channels)

        if not isinstance(extended_nwp_vars, list) or len(extended_nwp_vars)==0: 
            extended_nwp_vars = "none"

        ds = PeakWeatherDataset(root=root,
                               pad_missing_variables=True,
                               parameters=channels,
                               years=years,
                               extended_topo_vars=extended_topo_vars,
                               imputation_method=imputation_method,
                               interpolation_method=interpolation_method,
                               compute_uv=True,
                               freq=freq,
                               station_type=station_type,
                               extended_nwp_vars=extended_nwp_vars)
        covariates = {
            "stations_table": (ds.stations_table, "n f"),
            "installation_table": (ds.installation_table, "f f"),
            "parameters_table": (ds.parameters_table, "f f"),
        }

        ds.observations.index = ds.observations.index.astype("datetime64[ns, UTC]")

        # Optionally filter channels
        target = ds.observations
        mask = ds.mask

        if target_channels == "all":
            target_channels = ds.parameters
        target_params = pd.Index(ensure_list(target_channels))

        assert target_params.isin(ds.parameters).all(), \
            (f"Target channels {target_params.difference(ds.parameters)} not "
             f"in dataset parameters {ds.parameters}")

        if covariate_channels is None:
            covar_params = pd.Index([])
        elif covariate_channels == "other":
            covar_params = ds.parameters.difference(target_params)
            wind_params = {'wind_direction', 'wind_speed', 'wind_u', 'wind_v'}
            if target_params.isin(wind_params).any():
                covar_params = covar_params.difference(wind_params)
        else:
            covar_params = pd.Index(ensure_list(covariate_channels))

        assert covar_params.isin(ds.parameters).all(), \
            (f"Covariate channels {covar_params.difference(ds.parameters)} not "
             f"in dataset parameters {ds.parameters}")
        assert not target_params.isin(covar_params).any(), \
            (f"Covariate channels {covar_params.intersection(target_params)} "
             f"are also in target channels {target_params}")

        target_cols = pd.MultiIndex.from_product([ds.stations, target_params])
        target = target.loc[:, target_cols]
        mask = mask.loc[:, target_cols]

        if len(covar_params):
            covar_cols = pd.MultiIndex.from_product([ds.stations, covar_params])
            self.covariates_id = list(covar_params)
            covariates["u"] = (ds.observations.loc[:, covar_cols], "t n f")
            covariates["u_mask"] = (ds.mask.loc[:, covar_cols], "t n f")

        super(DatetimeDataset, self).__init__(target=target,
                                              mask=mask,
                                              covariates=covariates,
                                              similarity_score="distance",
                                              temporal_aggregation="mean",
                                              spatial_aggregation="mean",
                                              default_splitting_method="at_ts",
                                              force_synchronization=True,
                                              name=ds.__class__.__name__,
                                              precision=32)
        self.icon_data = None
        if isinstance(extended_nwp_vars, list):
            self.icon_data = {c: ds.get_icon_data(c) for c in extended_nwp_vars}

    def compute_similarity(self, method: str, **kwargs) -> Optional[np.ndarray]:
        if method == "distance":
            coords = self.stations_table.loc[:, ['latitude', 'longitude']]
            distances = geographical_distance(coords, to_rad=True).values
            theta = kwargs.get('theta', np.std(distances))
            return gaussian_kernel(distances, theta=theta)