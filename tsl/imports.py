import importlib
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Optional


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


def optional_import(module_name: str) -> Optional[ModuleType]:
    """Import an optional module when it is available.

    Args:
        module_name (str): Fully qualified name of the module to import.

    Returns:
        ModuleType or None: The imported module, or :obj:`None` when its top-level
        package is not installed.

    Raises:
        ImportError: If the package is installed but the module cannot be imported.
    """
    if not _package_available(module_name.split('.')[0]):
        return None
    return importlib.import_module(module_name)


def require_optional_dependency(
    module_name: str, install_name: str = None
) -> ModuleType:
    """Import an optional dependency or raise an actionable error.

    Args:
        module_name (str): Fully qualified name of the module to import.
        install_name (str, optional): Distribution name to show in the installation
            command. Defaults to the top-level package name.

    Returns:
        ModuleType: The imported module.

    Raises:
        ImportError: If the dependency is not installed or cannot be imported.
    """
    module = optional_import(module_name)
    if module is None:
        install_name = install_name or module_name.split('.')[0]
        raise ImportError(
            f"Optional dependency '{install_name}' is required for this functionality."
        )
    return module


def is_optional_instance(obj: Any, module_name: str, class_name: str) -> bool:
    """Check whether an object is an instance from an optional dependency.

    Args:
        obj (Any): Object to inspect.
        module_name (str): Module exporting the class.
        class_name (str): Name of the class in :obj:`module_name`.

    Returns:
        bool: Whether :obj:`obj` is an instance of the requested optional class.
    """
    module = optional_import(module_name)
    return module is not None and isinstance(obj, getattr(module, class_name))
