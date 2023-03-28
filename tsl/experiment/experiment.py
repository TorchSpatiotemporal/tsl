import inspect
import os
import os.path as osp
import random
import sys
from contextlib import contextmanager
from functools import wraps
from typing import Callable, List, Optional, Union

import torch
from pytorch_lightning import seed_everything

from tsl import config, logger
from tsl.imports import _HYDRA_AVAILABLE
from tsl.utils.python_utils import ensure_list

if _HYDRA_AVAILABLE:
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf, flag_override
    from omegaconf.errors import ConfigAttributeError

    from .resolvers import register_resolvers

    register_resolvers()
else:
    hydra = DictConfig = None


def get_hydra_cli_arg(key: str, delete: bool = False):
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

    # set the seed for the run
    # if None, then use random positive int32 (for logging compatibilities)
    seed = cfg.get("seed")
    if seed is None:
        seed = random.randrange(0, 10**9)
    seed = seed_everything(seed)

    # add run args to cfg
    run_args = dict(
        seed=seed,
        # name=hconf.job.name,
        dir=hconf.runtime.output_dir,
    )
    if hconf.get("output_subdir") is not None:
        run_args["tsl_subdir"] = osp.join(cfg.run.dir, hconf.output_subdir)
        # remove hydra conf from logging
        os.unlink(osp.join(run_args["tsl_subdir"], "hydra.yaml"))
    # set run name
    run_args["name"] = "${now:%Y-%m-%d_%H-%M-%S}_${run.seed}"
    with flag_override(cfg, "struct", False):
        cfg.run = DictConfig(run_args)

    # override data_dir in tsl config
    config.data_dir = cfg.get("data_dir", config.data_dir)

    # if True, then allow for adding new args to the cfg
    if cfg.get("allow_config_extension", False):
        OmegaConf.set_struct(cfg, False)

    # set the PyTorch num_threads from here
    if "num_threads" in cfg:
        torch.set_num_threads(cfg.num_threads)

    logger.info(
        "\n**** Experiment config ****\n" + OmegaConf.to_yaml(cfg, resolve=True)
    )

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
        pre_run_hooks (list): Ordered list of functions to call on
            :meth:`~tsl.experiment.Experiment.run` before the :attr:`run_fn`.
            Every hook must accept a single argument, being the experiment
            configuration, and act in-place on the configuration.
    """

    def __init__(
        self,
        run_fn: Callable,
        config_path: Optional[str] = None,
        config_name: Optional[str] = None,
        pre_run_hooks: Union[Callable, List[Callable]] = None,
    ):
        if not _HYDRA_AVAILABLE:
            raise RuntimeError(
                "Install optional dependency 'hydra-core'"
                f" to use {self.__class__.__name__}."
            )

        # store the run configuration
        self.cfg: Optional[DictConfig] = None

        # default config is cd/config
        if config_path is None:
            config_path = config.config_dir
        # allow override of config_path as Hydra cli arg:
        # config_path={config_path} same as --config-path {config_path}
        override_config_path = get_hydra_cli_arg("config_path", delete=True)
        config_path = override_config_path or config_path
        if not osp.isabs(config_path):
            root_path = osp.dirname(inspect.getfile(run_fn))
            config_path = osp.abspath(osp.join(root_path, config_path))
        self.config_path = config_path
        # store config_dir in tsl config
        config.config_dir = self.config_path

        # allow override of config_name as Hydra cli arg:
        # config={config_name} same as --config-name {config_name}
        override_config_name = get_hydra_cli_arg("config", delete=True)
        self.config_name = override_config_name or config_name

        sys.argv.insert(1, "hydra.output_subdir=null")

        self._pre_run_hooks = [_pre_experiment_routine]
        if pre_run_hooks is not None:
            pre_run_hooks = ensure_list(pre_run_hooks)
            for hook in pre_run_hooks:
                self.register_pre_run_hook(hook)

        self.run_fn = self.register_run_function(run_fn)
        self.run_output = None

    def register_pre_run_hook(self, hook: Callable):
        self._pre_run_hooks.append(hook)

    def register_run_function(self, run_fn: Callable) -> Callable:
        args = inspect.getfullargspec(run_fn).args
        if len(args) > 1:
            raise RuntimeError("run_fn must have a single 'cfg' parameter.")

        def run_fn_decorator(func: Callable) -> Callable:
            @wraps(func)
            def decorated_run_fn(cfg: DictConfig):
                # execute pre-run hooks
                for hook in self._pre_run_hooks:
                    hook(cfg)
                # store final config
                self.cfg = cfg
                self.log_config()

                self.run_output = func(cfg)
                return self.run_output

            return decorated_run_fn

        return hydra.main(
            config_path=self.config_path,
            config_name=self.config_name,
            version_base=None,
        )(run_fn_decorator(run_fn))

    def __repr__(self):
        return "{}(config_path={}, config_name={}, run_fn={})".format(
            self.__class__.__name__,
            self.config_path,
            self.config_name,
            self.run_fn.__name__,
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

    def log_config(self) -> None:
        """Save config as ``.yaml`` file in
        :meth:`~tsl.experiment.Experiment.run_dir`."""
        with open(osp.join(self.run_dir, "config.yaml"), "w") as fp:
            fp.write(OmegaConf.to_yaml(self.cfg, resolve=True))

    def get_config_dict(self) -> dict:
        return OmegaConf.to_object(self.cfg)

    @contextmanager
    def edit_config(self):
        with flag_override(self.cfg, "struct", False) as cfg:
            yield cfg

    def run(self):
        """Run the experiment routine."""
        self.run_fn()
        return self.run_output
