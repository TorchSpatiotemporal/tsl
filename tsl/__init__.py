import os

import tsl.global_scope
from tsl.global_scope import *

data = LazyLoader('data', globals(), 'tsl.data')
datasets = LazyLoader('datasets', globals(), 'tsl.datasets')
nn = LazyLoader('nn', globals(), 'tsl.nn')
predictors = LazyLoader('predictors', globals(), 'tsl.predictors')
imputers = LazyLoader('imputers', globals(), 'tsl.imputers')

__version__ = '0.1.0'

epsilon = 5e-8
config = Config()

config_file = os.path.join(config.curr_dir, 'tsl_config.yaml')
if os.path.exists(config_file):
    config.load_config_file(config_file)

__all__ = [
    '__version__',
    'config',
    'epsilon',
    'logger',
    'tsl',
    'data',
    'datasets',
    'nn',
    'predictors',
    'imputers'
]
