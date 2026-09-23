"""Import smoke tests for public modules and optional dependencies."""

import os
import subprocess
import sys
from pathlib import Path

PUBLIC_MODULES = [
    'tsl',
    'tsl.data',
    'tsl.data.datamodule',
    'tsl.data.loader',
    'tsl.data.preprocessing',
    'tsl.datasets',
    'tsl.datasets.prototypes',
    'tsl.engines',
    'tsl.experiment',
    'tsl.metrics',
    'tsl.metrics.numpy',
    'tsl.metrics.torch',
    'tsl.nn',
    'tsl.nn.blocks',
    'tsl.nn.layers',
    'tsl.nn.models',
    'tsl.ops',
    'tsl.ops.graph_generators',
    'tsl.transforms',
    'tsl.utils',
]


def test_public_modules_import_without_optional_extensions():
    modules = repr(PUBLIC_MODULES)
    script = f'''\
import builtins
import importlib

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch_sparse" or name.startswith("torch_sparse.") or name == "torch_scatter" or name.startswith("torch_scatter."):
        raise ModuleNotFoundError(f"optional dependency {{name}} is unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
for module in {modules}:
    importlib.import_module(module)
'''
    environment = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    environment['PYTHONPATH'] = os.pathsep.join(
        filter(None, [root, environment.get('PYTHONPATH')])
    )
    subprocess.run([sys.executable, '-c', script], check=True, env=environment)
