import os
import warnings
from argparse import ArgumentParser

import numpy as np

import tsl
from tsl.utils import parser_utils
from tsl.utils.python_utils import ensure_list


class TslExperiment:
    r"""
    Simple class to handle the routines used to run experiments.

    Args:
        run_fn: Python function that actually runs the experiment when called.
            The run function must accept single argument being the experiment
            hyperparameters.
        parser: Parser used to read the hyperparameters for the experiment.
        debug: Whether to run the experiment in debug mode.
        config_path: Path to configuration files, if not specified the default
            will be used.
    """

    def __init__(self,
                 run_fn,
                 parser: ArgumentParser,
                 debug=False,
                 config_path=None):
        self.run_fn = run_fn
        self.parser = parser
        self.debug = debug
        if config_path is not None:
            self.config_root = config_path
        else:
            self.config_root = tsl.config.config_dir

    def _check_config(self, hparams):
        config_file = hparams.__dict__.get('config', None)
        if config_file is not None:
            # read config file
            import yaml

            config_file = os.path.join(self.config_root, config_file)
            with open(config_file, 'r') as fp:
                experiment_config = yaml.load(fp, Loader=yaml.FullLoader)

            # update hparams
            hparams = parser_utils.update_from_config(hparams,
                                                      experiment_config)
            if hasattr(self.parser, 'parsed_args'):
                self.parser.parsed_args.update(experiment_config)
        return hparams

    def make_run_dir(self):
        """Create directory to store run logs and artifacts."""
        raise NotImplementedError

    def run(self):
        hparams = self.parser.parse_args()
        hparams = self._check_config(hparams)

        return self.run_fn(hparams)

    def run_many_times_sequential(self, n):
        hparams = self.parser.parse_args()
        hparams = self._check_config(hparams)
        warnings.warn('Running multiple times. Make sure that randomness is '
                      'handled properly.')
        for i in range(n):
            print(f"**************Trial n.{i}**************")
            np.random.seed()
            self.run_fn(hparams)

    def run_search_sequential(self, n):
        hparams = self.parser.parse_args()
        hparams = self._check_config(hparams)
        for i, h in enumerate(hparams.trials(n)):
            print(f'**************Trial n.{i}**************')
            try:
                np.random.seed()
                self.run_fn(h)
            except RuntimeError as err:
                print(f'Trial n. {i} failed due to a Runtime error: {err}')

    def run_search_parallel(self, n, workers, gpus=None):
        hparams = self.parser.parse_args()
        hparams = self._check_config(hparams)
        if gpus is None:
            hparams.optimize_parallel_cpu(self.run_fn,
                                          nb_trials=n,
                                          nb_workers=workers)
        else:
            gpus = ensure_list(gpus)
            hparams.optimize_parallel_gpu(self.run_fn,
                                          max_nb_trials=n,
                                          gpu_ids=gpus)
