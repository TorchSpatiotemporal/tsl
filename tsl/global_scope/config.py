import os
from typing import Mapping, Optional


class Config(dict):
    """Manage the package configuration from a single object.

    With a :obj:`Config` object you can edit settings within the tsl scope, like
    directory in which you store configuration files for experiments
    (:obj:`config_dir`), logs (:obj:`log_dir`), and data (:obj:`data_dir`).
    """

    def __init__(self, **kwargs):
        super(Config, self).__init__()
        # configure paths for config files and logs
        self.config_dir = kwargs.pop('config_dir', 'config')
        self.log_dir = kwargs.pop('log_dir', 'log')
        # set 'data_dir' as directory for data loading and downloading
        # defaults to '{tsl_path}/.storage'
        default_storage = os.path.join(self.root_dir, '.storage')
        self.data_dir = kwargs.pop('data_dir', default_storage)
        self.update(**kwargs)

    def __setitem__(self, key: str, value):
        # when adding a directory, transform it to an absolute path (if it is
        # not already) considering the path relative to the current directory
        if key.endswith('_dir') and value is not None:
            if not os.path.isabs(value):
                value = os.path.join(self.curr_dir, value)
        super(Config, self).__setitem__(key, value)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __delattr__(self, item):
        del self[item]

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        for name, value in sorted(self.items()):
            arg_strings.append('%s=%r' % (name, value))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

    @property
    def root_dir(self):
        """Path to tsl installation."""
        return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    @property
    def curr_dir(self):
        """System current directory."""
        return os.getcwd()

    def update(self, mapping: Optional[Mapping] = None, **kwargs) -> None:
        mapping = dict(mapping or {}, **kwargs)
        for k, v in mapping.items():
            self[k] = v

    def disable_logging(self):
        from .logger import logger
        logger.disabled = True

    def load_config_file(self, filename: str):
        """Load a configuration from a json or yaml file."""
        with open(filename, 'r') as fp:
            if filename.endswith('.json'):
                import json
                data = json.load(fp)
            elif filename.endswith('.yaml') or filename.endswith('.yml'):
                import yaml
                data = yaml.load(fp, Loader=yaml.FullLoader)
            else:
                raise RuntimeError('Config file format not supported.')
        self.update(data)
        return self

    @classmethod
    def from_config_file(cls, filename: str):
        """Create new configuration from a json or yaml file."""
        config = cls()
        config.load_config_file(filename)
        return config
