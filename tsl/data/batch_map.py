from typing import Iterator
from typing import (Optional, Union, List, Tuple, Mapping)

from tsl.utils.python_utils import ensure_list
from .utils import SynchMode


class BatchMapItem:
    def __init__(self, keys: Union[List, str],
                 synch_mode: SynchMode = SynchMode.WINDOW,
                 preprocess: bool = True,
                 cat_dim: Optional[int] = -1,
                 n_channels: Optional[int] = None):
        super(BatchMapItem, self).__init__()
        self.keys = ensure_list(keys)
        assert isinstance(synch_mode, SynchMode)
        self.synch_mode = synch_mode
        self.preprocess = preprocess
        if len(self.keys) > 1:
            assert cat_dim is not None, \
                '"cat_dim" cannot be None with multiple keys.'
        self.cat_dim = cat_dim
        self.n_channels = n_channels

    def __repr__(self):
        return "([{}], {})".format(', '.join(self.keys), self.synch_mode.name)

    def kwargs(self):
        return self.__dict__


class BatchMap(Mapping):

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key: str, value: Union[BatchMapItem, Tuple, Mapping]):
        # cast item
        if isinstance(value, BatchMapItem):
            pass
        elif isinstance(value, Tuple):
            value = BatchMapItem(*value)
        elif isinstance(value, (List, str)):
            value = BatchMapItem(value)
        elif isinstance(value, Mapping):
            value = BatchMapItem(**value)
        else:
            raise TypeError('Invalid type for InputMap item "{}"'
                            .format(type(value)))
        self.__dict__[key] = value

    def __getitem__(self, k):
        return self.__dict__[k]

    def __len__(self) -> int:
        return len(self.__dict__)

    def __iter__(self) -> Iterator:
        return iter(self.__dict__)

    def __repr__(self):
        s = ['({}={}, {})'.format(key, value.keys, value.synch_mode.name)
             for key, value in self.items()]
        return "{}[{}]".format(self.__class__.__name__, ', '.join(s))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def by_synch_mode(self, synch_mode: SynchMode):
        return {k: v for k, v in self.items() if
                v.synch_mode is synch_mode}
