import os
from typing import Mapping, Optional


class Config(dict):

    def __init__(self, **kwargs):
        super(Config, self).__init__()
        if 'config_dir' in kwargs:
            self.config_dir = kwargs.pop('config_dir')
        else:
            self.config_dir = None
        if 'log_dir' in kwargs:
            self.log_dir = kwargs.pop('log_dir')
        else:
            self.log_dir = os.path.join(self.curr_dir, 'log_dir')
        self.update(**kwargs)

    def __setitem__(self, key: str, value):
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
        return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    @property
    def curr_dir(self):
        return os.getcwd()

    @property
    def store_dir(self):
        return os.path.join(self.root_dir, '.storage')

    def update(self, mapping: Optional[Mapping] = None, **kwargs) -> None:
        mapping = dict(mapping or {}, **kwargs)
        for k, v in mapping.items():
            self[k] = v

    def disable_logging(self):
        from .logger import logger
        logger.disabled = True

    def load_config_file(self, filename: str):
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
        config = cls()
        config.load_config_file(filename)
        return config
