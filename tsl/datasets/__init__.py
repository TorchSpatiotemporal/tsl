# Interfaces
# Datasets
from .air_quality import AirQuality
from .elergone import Elergone
from .gpvar import GPVARDataset, GPVARDatasetAZ
from .metr_la import MetrLA
from .mts_benchmarks import (
    ElectricityBenchmark,
    ExchangeBenchmark,
    SolarBenchmark,
    TrafficBenchmark,
)
from .pems_bay import PemsBay
from .pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from .prototypes import Dataset, DatetimeDataset, TabularDataset
from .prototypes import classes as prototype_classes
from .synthetic import GaussianNoiseSyntheticDataset

dataset_classes = [
    "AirQuality",
    "Elergone",
    "MetrLA",
    "PemsBay",
    "PeMS03",
    "PeMS04",
    "PeMS07",
    "PeMS08",
    "ElectricityBenchmark",
    "TrafficBenchmark",
    "SolarBenchmark",
    "ExchangeBenchmark",
    "GaussianNoiseSyntheticDataset",
    "GPVARDataset",
    "GPVARDatasetAZ",
]

__all__ = prototype_classes + dataset_classes
