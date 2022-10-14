import inspect
import os
import sys
from functools import wraps
from typing import Optional, Any, Callable

import torch
from pytorch_lightning import seed_everything

from tsl import logger

try:
    _hydra_installed = True
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf, flag_override
    from omegaconf.errors import ConfigAttributeError
except ModuleNotFoundError:
    _hydra_installed = False

_UNSPECIFIED_: Any = object()


def get_cli_arg(key: str, delete: bool = False):
    try:
        key_idx = [arg.split("=")[0] for arg in sys.argv].index(key)
        arg = sys.argv[key_idx].split("=")[1]
        if delete:
            del sys.argv[key_idx]
        return arg
    except ValueError:
        return None


def _pre_experiment_routine(cfg: DictConfig):
    hconf = HydraConfig.get()

    seed = cfg.get('seed', None)
    seed = seed_everything(seed)
    with flag_override(cfg, 'struct', False):
        cfg.run = DictConfig(dict(seed=seed,
                                  name=hconf.job.name,
                                  # dir=hconf.runtime.output_dir,
                                  dir=hconf.run.dir))

    if not cfg.get('structure_config', True):
        OmegaConf.set_struct(cfg, False)

    if 'num_threads' in cfg:
        torch.set_num_threads(1)

    logger.info("\n**** Experiment config ****\n" +
                OmegaConf.to_yaml(cfg, resolve=True))

    return cfg


class Experiment:
    r"""Simple class to handle the routines used to run experiments.

    This class relies heavily on the `Hydra <https://hydra.cc/>`_ framework,
    check `Hydra docs <https://hydra.cc/docs/intro/>`_ for usage information.

    Hydra is an optional dependency of tsl, to install it using pip:

    .. code-block:: bash

        pip install hydra-core

    Args:
        run_fn (callable): Python function that actually runs the experiment
            when called. The run function must accept a single argument,
            being the experiment configuration.
        config_path (str, optional): Path to configuration files.
            If not specified the default will be used.
        config_name (str, optional): Name of the configuration file in
            :attr:`config_path` to be used. The :obj:`.yaml` extension can be
            omitted.
    """

    def __init__(self, run_fn: Callable,
                 config_path: Optional[str] = _UNSPECIFIED_,
                 config_name: Optional[str] = None):
        if not _hydra_installed:
            raise RuntimeError("Install optional dependency 'hydra-core'"
                               f" to use {self.__class__.__name__}.")
        self.cfg: Optional[DictConfig] = None
        override_config_path = get_cli_arg('config_path', delete=True)
        config_path = override_config_path or config_path
        if not os.path.isabs(config_path):
            root_path = os.path.dirname(inspect.getfile(run_fn))
            config_path = os.path.join(root_path, config_path)
        self.config_path = config_path
        override_config_name = get_cli_arg('config', delete=True)
        self.config_name = override_config_name or config_name
        self.run_fn = self._decorate_run_fn(run_fn)

    def _decorate_run_fn(self, run_fn: Callable):
        args = inspect.getfullargspec(run_fn).args
        if len(args) > 1:
            raise RuntimeError("run_fn must have a single 'cfg' parameter.")

        def run_fn_decorator(func: Callable) -> Callable:
            @wraps(func)
            def decorated_run_fn(cfg: DictConfig):
                _pre_experiment_routine(cfg)
                self.cfg = cfg
                return func(cfg)  # pass cfg to decorated function

            return decorated_run_fn

        return hydra.main(config_path=self.config_path,
                          config_name=self.config_name,
                          version_base=None)(run_fn_decorator(run_fn))

    def __repr__(self):
        return "{}(config_path={}, config_name={}, run_fn={})".format(
            self.__class__.__name__, self.config_path,
            self.config_name, self.run_fn.__name__
        )

    @property
    def run_dir(self):
        """Directory of the current run, where logs and artifacts are stored."""
        if self.cfg is not None:
            try:
                return self.cfg.run.dir
            except ConfigAttributeError:
                return None
        return None

    def run(self):
        """Run the experiment routine."""
        return self.run_fn()
