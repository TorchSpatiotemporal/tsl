from .config import Config
from .lazy_loader import LazyLoader
from .logger import logger

data = LazyLoader('data', globals(), 'tsl.data')
datasets = LazyLoader('datasets', globals(), 'tsl.datasets')
nn = LazyLoader('nn', globals(), 'tsl.nn')
inference = LazyLoader('inference', globals(), 'tsl.inference')

__version__ = '0.1.1'

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
    'inference'
]
