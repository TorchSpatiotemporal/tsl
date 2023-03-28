from typing import Iterator, List, Mapping, Optional, Tuple, Union

from tsl.utils.python_utils import ensure_list

from .synch_mode import STATIC, WINDOW, SynchMode


class BatchMapItem:

    def __init__(self,
                 keys: Union[List, str],
                 synch_mode: Optional[Union[SynchMode, str]] = None,
                 preprocess: bool = True,
                 cat_dim: Optional[int] = -1,
                 pattern: Optional[str] = None,
                 shape: Optional[tuple] = None):
        super(BatchMapItem, self).__init__()
        # store keys to be mapped
        self.keys = ensure_list(keys)

        # store cat_dim, i.e., the dimension along which keys must be
        # concatenated
        if len(self.keys) > 1 and cat_dim is None:
            raise RuntimeError("'cat_dim' cannot be None with multiple keys.")
        self.cat_dim = cat_dim

        # store synch_mode, i.e., where to put (sliced) data in the batch
        if isinstance(synch_mode, str):
            synch_mode = getattr(SynchMode, synch_mode.upper())
        self.synch_mode = synch_mode

        self.preprocess = preprocess
        self.pattern = pattern
        self.shape = None if shape is None else tuple(shape)

    def __setattr__(self, key, value):
        super(BatchMapItem, self).__setattr__(key, value)
        if key == 'pattern' and value is not None and self.synch_mode is None:
            synch_mode = WINDOW if 't' in value else STATIC
            super(BatchMapItem, self).__setattr__('synch_mode', synch_mode)

    def __repr__(self):
        return "([{}], pattern='{}', shape={})".format(', '.join(self.keys),
                                                       self.pattern,
                                                       self.shape)

    def kwargs(self):
        return self.__dict__


class BatchMap(Mapping):

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key: str, value: Union[BatchMapItem, Tuple,
                                                 Mapping]):
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
            raise TypeError('Invalid type for InputMap item "{}"'.format(
                type(value)))
        self.__dict__[key] = value

    def __getitem__(self, k):
        return self.__dict__[k]

    def __len__(self) -> int:
        return len(self.__dict__)

    def __iter__(self) -> Iterator:
        return iter(self.__dict__)

    def __repr__(self):
        s = [
            "'{}': {}".format(key, repr(value)) for key, value in self.items()
        ]
        return "{}(\n  {}\n)".format(self.__class__.__name__, ',\n  '.join(s))

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def by_synch_mode(self, synch_mode: SynchMode):
        return {k: v for k, v in self.items() if v.synch_mode is synch_mode}
