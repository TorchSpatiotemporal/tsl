# Interfaces
from .prototypes import Dataset, TabularDataset, DatetimeDataset
from .prototypes import classes as prototype_classes
# Datasets
from .air_quality import AirQuality
from .elergone import Elergone
from .metr_la import MetrLA
from .mts_benchmarks import (
    ElectricityBenchmark,
    TrafficBenchmark,
    SolarBenchmark,
    ExchangeBenchmark
)
from .pems_bay import PemsBay
from .pems_benchmarks import (
    PeMS03,
    PeMS04,
    PeMS07,
    PeMS08
)
from .synthetic import GaussianNoiseSyntheticDataset

dataset_classes = [
    'AirQuality',
    'Elergone',
    'MetrLA',
    'PemsBay',
    'PeMS03',
    'PeMS04',
    'PeMS07',
    'PeMS08',
    'ElectricityBenchmark',
    'TrafficBenchmark',
    'SolarBenchmark',
    'ExchangeBenchmark',
    'GaussianNoiseSyntheticDataset'
]

__all__ = prototype_classes + dataset_classes
