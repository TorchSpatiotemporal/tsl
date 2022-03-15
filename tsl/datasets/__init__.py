# Interfaces
from .prototypes import Dataset, PandasDataset
# Datasets
from .air_quality import AirQuality
from .electricity import Electricity
from .metr_la import MetrLA
from .pems_bay import PemsBay

__all__ = [
    'Dataset',
    'PandasDataset',
    'AirQuality',
    'Electricity',
    'MetrLA',
    'PemsBay'
]

prototype_classes = __all__[:2]
dataset_classes = __all__[2:]
