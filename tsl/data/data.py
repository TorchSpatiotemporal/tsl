import copy
from typing import Iterator, Callable, Dict
from typing import (Optional, Any, List, Iterable, Tuple,
                    Mapping)

from torch import Tensor
from torch_geometric.data.data import Data as PyGData, size_repr
from torch_geometric.data.storage import BaseStorage
from torch_geometric.data.view import KeysView, ValuesView, ItemsView
from torch_sparse import SparseTensor

from tsl.utils.python_utils import ensure_list


class StorageView(BaseStorage):

    def __init__(self, store, keys: Optional[Iterable] = None):
        self.__keys = tuple()
        super(StorageView, self).__init__()
        self._mapping = store
        self._keys = keys

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [size_repr(k, v) for k, v in self.items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __setattr__(self, key, value):
        if key == '_keys':
            if value is None:
                keys = []
            else:
                keys = ensure_list(value)
            self.__keys = tuple(keys)
        else:
            super(StorageView, self).__setattr__(key, value)

    def __getitem__(self, item: str) -> Any:
        if item in self._keys:
            return self._mapping[item]
        else:
            raise KeyError(item)

    def __setitem__(self, key, value):
        super(StorageView, self).__setitem__(key, value)
        self.add_keys(key)

    def __delitem__(self, key):
        super(StorageView, self).__delitem__(key)
        self.del_keys(key)

    def __iter__(self) -> Iterator:
        return iter(self.values())

    # Override methods to account for filtering keys  #########################

    def _filter_keys(self, args: Tuple):
        keys = self._keys
        if len(args):
            keys = [arg for arg in args if arg in self._keys]
        return keys

    def keys(self, *args: List[str]) -> KeysView:
        keys = self._filter_keys(args)
        if len(keys) > 0:
            return super(StorageView, self).keys(*keys)
        return KeysView({})

    def values(self, *args: List[str]) -> ValuesView:
        keys = self._filter_keys(args)
        if len(keys) > 0:
            return super(StorageView, self).values(*keys)
        return ValuesView({})

    def items(self, *args: List[str]) -> ItemsView:
        keys = self._filter_keys(args)
        if len(keys) > 0:
            return super(StorageView, self).items(*keys)
        return ItemsView({})

    def apply_(self, func: Callable, *args: List[str]):
        keys = self._filter_keys(args)
        if len(keys) > 0:
            return super(StorageView, self).apply_(func, *keys)
        return self

    def apply(self, func: Callable, *args: List[str]):
        keys = self._filter_keys(args)
        if len(keys) > 0:
            return super(StorageView, self).apply(func, *keys)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return copy.copy({k: self._mapping[k] for k in self._keys})

    def numpy(self, *args: List[str]):
        r"""Transform all tensors to numpy arrays, either for all
        attributes or only the ones given in :obj:`*args`."""
        self.detach().cpu()
        return self.apply(lambda x: x.numpy(), *args)

    @property
    def _keys(self) -> tuple:
        return tuple(k for k in self.__keys if k in self._mapping)

    @_keys.setter
    def _keys(self, value):
        pass

    def add_keys(self, *keys):
        keys = set(keys).difference(self.__keys)
        self.__keys = tuple([*self.__keys, *keys])

    def del_keys(self, *keys):
        keys = tuple(k for k in self.__keys if k not in keys)
        self.__keys = keys


class DataView(StorageView):
    pass


class Data(PyGData):
    input: DataView
    target: DataView
    pattern: dict

    def __init__(self, input: Optional[Mapping] = None,
                 target: Optional[Mapping] = None,
                 mask: Optional[Tensor] = None,
                 transform: Optional[Mapping] = None,
                 pattern: Optional[Mapping] = None,
                 **kwargs):
        input = input if input is not None else dict()
        target = target if target is not None else dict()
        super(Data, self).__init__(**input, **target, **kwargs)
        # Set 'input' as view on input keys
        self.__dict__['input'] = DataView(self._store, input.keys())
        # Set 'target' as view on input keys
        self.__dict__['target'] = DataView(self._store, target.keys())
        # Add mask
        self.mask = mask
        # Add transform modules
        transform = transform if transform is not None else dict()
        self.transform: dict = transform
        # Add patterns
        pattern = pattern if pattern is not None else dict()
        self.__dict__['pattern'] = pattern

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        inputs = [size_repr(k, v) for k, v in self.input.items()]
        inputs = 'input:{{{}}}'.format(', '.join(inputs))
        targets = [size_repr(k, v) for k, v in self.target.items()]
        targets = 'target:{{{}}}'.format(', '.join(targets))
        info = [inputs, targets, "has_mask={}".format(self.has_mask)]
        if self.has_transform:
            info += ["transform=[{}]".format(', '.join(self.transform.keys()))]
        return '{}({})'.format(cls, ', '.join(info))

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        elif key.startswith('edge_'):
            return 0
        else:
            return -2

    def stores_as(self, data: 'Data'):
        self.input._keys = list(data.input.keys())
        self.target._keys = list(data.target.keys())
        return self

    @property
    def has_transform(self):
        return 'transform' in self and len(self.transform) > 0

    @property
    def has_mask(self):
        return self.get('mask') is not None

    @property
    def edge_weight(self) -> Any:
        return self.get('edge_weight')

    def numpy(self, *args: List[str]):
        r"""Transform all tensors to numpy arrays, either for all
        attributes or only the ones given in :obj:`*args`."""
        self.detach().cpu()
        return self.apply(lambda x: x.numpy(), *args)
