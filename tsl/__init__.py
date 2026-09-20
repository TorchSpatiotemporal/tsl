from importlib.util import find_spec

_torch_dependencies = {
    'torch': 'PyTorch',
    'torch_geometric': 'PyG (torch-geometric)',
    'torch_scatter': 'torch-scatter',
    'torch_sparse': 'torch-sparse',
}
_missing_torch_dependencies = [
    display_name
    for module_name, display_name in _torch_dependencies.items()
    if find_spec(module_name) is None
]
if _missing_torch_dependencies:
    missing = ', '.join(_missing_torch_dependencies)
    raise ImportError(
        f'tsl requires {missing} to be installed before tsl. Install a compatible'
        'PyTorch/PyG stack by following the tsl quickstart: '
        'https://torch-spatiotemporal.readthedocs.io/en/latest/usage/quickstart.html.'
    )

from ._logger import logger
from .config import Config
from .lazy_loader import LazyLoader

data = LazyLoader('data', globals(), 'tsl.data')
datasets = LazyLoader('datasets', globals(), 'tsl.datasets')
nn = LazyLoader('nn', globals(), 'tsl.nn')
engines = LazyLoader('engines', globals(), 'tsl.engines')
metrics = LazyLoader('metrics', globals(), 'tsl.metrics')

__version__ = '0.9.6'

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
