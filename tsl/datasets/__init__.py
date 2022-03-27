# Interfaces
from .prototypes import Dataset, PandasDataset
# Datasets
from .air_quality import AirQuality
from .elergone import Elergone
from .metr_la import MetrLA
from .pems_bay import PemsBay
from .mts_benchmarks import ElectricityBenchmark, TrafficBenchmark, SolarBenchmark, ExchangeBenchmark

__all__ = [
    'Dataset',
    'PandasDataset',
    'AirQuality',
    'Elergone',
    'MetrLA',
    'PemsBay',
    'ElectricityBenchmark',
    'TrafficBenchmark',
    'SolarBenchmark',
    'ExchangeBenchmark'
]

prototype_classes = __all__[:2]
dataset_classes = __all__[2:]
