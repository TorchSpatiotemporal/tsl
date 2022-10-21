import importlib
from importlib.util import find_spec


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    From ``pytorch_lightning.utilities.imports``."""
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def _module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    From ``pytorch_lightning.utilities.imports``."""
    module_names = module_path.split(".")
    if not _package_available(module_names[0]):
        return False
    try:
        module = importlib.import_module(module_names[0])
    except AttributeError:
        # Python 3.6
        return False
    except ImportError:
        return False
    for name in module_names[1:]:
        if not hasattr(module, name):
            return False
        module = getattr(module, name)
    return True


_HYDRA_AVAILABLE = _package_available("hydra")
_NEPTUNE_AVAILABLE = _package_available("neptune")
_FAST_TRANSFORMER_AVAILABLE = _package_available("pytorch_fast_transformers")
