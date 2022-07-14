import os
from typing import Union, List

import pandas as pd

from .prototypes import PandasDataset
from ..ops.similarities import gaussian_kernel, geographical_distance
from ..utils import download_url
from ..utils.python_utils import ensure_list


class PvUS(PandasDataset):
    r"""Simulated solar power production from more than 5,000 photovoltaic
    plants in the US.

    Data are provided by `National Renewable Energy Laboratory (NREL)
    <https://www.nrel.gov/>`_'s `Solar Power Data for Integration Studies
    <https://www.nrel.gov/grid/solar-power-data.html>`_. Original raw data
    consist of 1 year (2006) of 5-minute solar power (in MW) for approximately
    5,000 synthetic PV plants in the United States.

    Preprocessed data are resampled in 10-minutes intervals taking the average.
    The entire dataset contains 5016 plants, divided in two macro zones (east
    and west). The "east" zone contains 4084 plants, the "west" zone has 1082
    plants. Some states appear in both zones, with plants at same geographical
    position. When loading the entire datasets, duplicated plants in "east" zone
    are dropped.

    Dataset information:
        + Time steps: 52560
        + Nodes: 5016
        + Channels: 1
        + Sampling rate: 10 minutes
        + Missing values: 0.00%

    Static attributes:
        + :obj:`metadata`: table of PV farm information
            + "capacity": total capacity of the farm (in MW)
            + "state_id": id of the state in which the farm is located
            + "state": complete name of state in which the farm is located
            + "lat": latitude of the farm
            + "lon": longitude of the farm
            + "type": type of the panel
                - UPV: Utility scale PV
                - DPV: Distributed PV
              From NREL: "The practical difference between UPV and DPV is in
              the configurations (UPV has single axis tracking while DPV is
              fixed tilt equaling to latitude) and the smoothing (both are run
              through a low-pass filter, the DPV will have more of the high
              frequency variability smoothed out)."
    """
    url = {
        'east': 'https://drive.switch.ch/index.php/s/ZUORMr4uzBSr04b/download',
        'west': 'https://drive.switch.ch/index.php/s/HRPNJdeAzeQLA1f/download'
    }

    available_zones = ['east', 'west']
    similarity_options = {'distance', 'correntropy'}

    def __init__(self, zones: Union[str, List] = None,
                 root: str = None, freq: str = None):
        # set root path
        if zones is None:
            zones = self.available_zones
        else:
            zones = ensure_list(zones)
            if not set(zones).issubset(self.available_zones):
                invalid_zones = set(zones).difference(self.available_zones)
                raise ValueError(f"Invalid zones {invalid_zones}. "
                                 f"Allowed zones are {self.available_zones}.")
        self.zones = zones
        self.root = root
        # load dataset
        actual, metadata = self.load()
        super().__init__(target=actual, freq=freq,
                         similarity_score="distance",
                         spatial_aggregation="sum",
                         temporal_aggregation="mean",
                         name="PV-US")
        self.add_covariate('metadata', metadata, pattern='n f')

    @property
    def raw_file_names(self):
        return [f'{zone}.h5' for zone in self.zones]

    @property
    def required_file_names(self):
        return self.raw_file_names

    def download(self) -> None:
        for zone in self.zones:
            filename = f'{zone}.h5'
            download_url(self.url[zone], self.root_dir, filename)

    def load_raw(self):
        self.maybe_download()
        actual, metadata = [], []
        for zone in self.zones:
            # load zone data
            zone_path = os.path.join(self.root_dir, f'{zone}.h5')
            actual.append(pd.read_hdf(zone_path, key='actual'))
            metadata.append(pd.read_hdf(zone_path, key='metadata'))
        # concat zone and sort by plant id
        actual = pd.concat(actual, axis=1).sort_index(axis=1, level=0)
        metadata = pd.concat(metadata, axis=0).sort_index()
        # drop duplicated farms when loading whole dataset
        if len(self.zones) == 2:
            duplicated_farms = metadata.index[[s_id.endswith('-east')
                                               for s_id in metadata.state_id]]
            metadata = metadata.drop(duplicated_farms, axis=0)
            actual = actual.drop(duplicated_farms, axis=1, level=0)
        return actual, metadata

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            # compute distances from latitude and longitude degrees
            loc_coord = self.metadata.loc[:, ['lat', 'lon']]
            dist = geographical_distance(loc_coord, to_rad=True).values
            return gaussian_kernel(dist)
