# Interfaces
from .prototypes import Dataset, TabularDataset, PandasDataset
from .prototypes import classes as prototype_classes
# Datasets
from .air_quality import AirQuality
from .elergone import Elergone
from .metr_la import MetrLA
from .pems_bay import PemsBay
from .mts_benchmarks import (
    ElectricityBenchmark,
    TrafficBenchmark,
    SolarBenchmark,
    ExchangeBenchmark
)

dataset_classes = [
    'AirQuality',
    'Elergone',
    'MetrLA',
    'PemsBay',
    'ElectricityBenchmark',
    'TrafficBenchmark',
    'SolarBenchmark',
    'ExchangeBenchmark'
]

__all__ = prototype_classes + dataset_classes
