from ._logger import logger
from .config import Config
from .lazy_loader import LazyLoader

data = LazyLoader('data', globals(), 'tsl.data')
datasets = LazyLoader('datasets', globals(), 'tsl.datasets')
nn = LazyLoader('nn', globals(), 'tsl.nn')
engines = LazyLoader('engines', globals(), 'tsl.engines')
metrics = LazyLoader('metrics', globals(), 'tsl.metrics')

__version__ = '1.0.0'

epsilon = 5e-8
config = Config()

__all__ = [
    '__version__',
    'config',
    'epsilon',
    'logger',
    'data',
    'datasets',
    'nn',
    'engines',
    'metrics',
]
