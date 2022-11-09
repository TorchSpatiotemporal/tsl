import copy
from typing import Iterator, Callable, Dict
from typing import (Optional, Any, List, Iterable, Tuple,
                    Mapping)

import torch
from einops import rearrange
from torch import Tensor
from torch_geometric.data.data import Data as PyGData, size_repr
from torch_geometric.data.storage import BaseStorage
from torch_geometric.data.view import KeysView, ValuesView, ItemsView
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor

from tsl.utils.python_utils import ensure_list


class StorageView(BaseStorage):

    def __init__(self, store, keys: Optional[Iterable] = None):
        self.__keys = tuple()
        super(StorageView, self).__init__()
        self._mapping = store
        self._keys = keys  # noqa

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

    def add_keys(self, *keys):
        keys = set(keys).difference(self.__keys)
        self.__keys = tuple([*self.__keys, *keys])

    def del_keys(self, *keys):
        keys = tuple(k for k in self.__keys if k not in keys)
        self.__keys = keys


class Data(PyGData):
    r"""A data object describing a spatiotemporal graph, i.e., a graph with time
    series of equal length associated with every node.


    The data object extends :class:`~torch_geometric.data.Data`, thus preserving
    all its functionalities (see the :class:`documentation
    <torch_geometric.data.Data>` and `the accompanying
    tutorial <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    introduction.html#data-handling-of-graphs>`_).

    Args:
        input (Mapping, optional): Named mapping of :class:`~torch.Tensor` to be
            used as input to the model.
            (default: :obj:`None`)
        target (Mapping, optional): Named mapping of :class:`~torch.Tensor` to be
            used as target of the task.
            (default: :obj:`None`)
        mask (Tensor, optional): The optional mask associated with the target.
            (default: :obj:`None`)
        transform (Mapping, optional): Named mapping of
            :class:`~tsl.data.preprocessing.Scaler` associated with entries in
            :attr:`input` or :attr:`output`.
            (default: :obj:`None`)
        pattern (Mapping, optional): Map of the pattern of each entry in
            :attr:`input` or :attr:`output`.
            (default: :obj:`None`)
        **kwargs: Any keyword argument for :class:`~torch_geometric.data.Data`.
    """

    input: StorageView
    target: StorageView
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
        self.__dict__['input'] = StorageView(self._store, input.keys())
        # Set 'target' as view on target keys
        self.__dict__['target'] = StorageView(self._store, target.keys())
        # Add mask
        self.mask = mask  # noqa
        # Add transform modules
        transform = transform if transform is not None else dict()
        self.transform: dict = transform  # noqa
        # Add patterns
        pattern = pattern if pattern is not None else dict()
        self.__dict__['pattern'] = pattern

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        inputs = [size_repr(k, v) for k, v in self.input.items()]
        inputs = 'input=({})'.format(', '.join(inputs))
        targets = [size_repr(k, v) for k, v in self.target.items()]
        targets = 'target=({})'.format(', '.join(targets))
        info = [inputs, targets, "has_mask={}".format(self.has_mask)]
        if self.has_transform:
            info += ["transform=[{}]".format(', '.join(self.transform.keys()))]
        return '{}(\n  {}\n)'.format(cls, ',\n  '.join(info))

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in self.pattern:
            if 'n' in self.pattern[key]:  # cat along node dimension
                return self.pattern[key].split(' ').index('n')
            elif 'e' in self.pattern[key]:  # cat along edge dimension
                return self.pattern[key].split(' ').index('e')
            else:  # stack on batch dimension
                return None
        return super(Data, self).__cat_dim__(key, value, *args, **kwargs)

    def stores_as(self, data: 'Data'):
        # copy input and target keys in self with no check that keys are in self
        # used when batching Data objects
        self.input._keys = data.input._keys  # noqa
        self.target._keys = data.target._keys  # noqa
        self.__dict__['pattern'] = data.pattern
        return self

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def mask(self) -> Any:
        return self['mask'] if 'mask' in self._store else None

    @property
    def transform(self) -> Any:
        return self['transform'] if 'transform' in self._store else None

    @property
    def has_transform(self):
        return 'transform' in self._store and len(self.transform) > 0

    @property
    def has_mask(self):
        return self['mask'] is not None if 'mask' in self._store else False

    def numpy(self, *args: List[str]):
        r"""Transform all tensors to numpy arrays, either for all
        attributes or only the ones given in :obj:`*args`."""
        self.detach().cpu()
        return self.apply(lambda x: x.numpy(), *args)

    def rearrange_element(self, key: str, pattern: str, **axes_lengths):
        r"""Rearrange key in Data according to the provided patter
         using `einops.rearrange <https://einops.rocks/api/rearrange/>`_."""
        key_pattern = self.pattern[key]
        if '->' in pattern:
            start_pattern, end_pattern = pattern.split('->')
            start_pattern = start_pattern.strip()
            end_pattern = end_pattern.strip()
            if key_pattern != start_pattern:
                raise RuntimeError(f"Starting pattern {start_pattern} does not "
                                   f"match with key patter {key_pattern}.")
        else:
            end_pattern = pattern
            pattern = key_pattern + ' -> ' + pattern
        self[key] = rearrange(self[key], pattern, **axes_lengths)
        self.pattern[key] = end_pattern
        if key in self.transform:
            self.transform[key] = self.transform[key].rearrange(end_pattern)

    def rearrange(self, patterns: Mapping):
        r"""Rearrange all keys in Data according to the provided patter
         using `einops.rearrange <https://einops.rocks/api/rearrange/>`_."""
        for key, pattern in patterns.items():
            self.rearrange_element(key, pattern)
        return self

    def subgraph(self, node_index):
        if isinstance(self.edge_index, SparseTensor):
            self.edge_index = self.edge_index[node_index, node_index]
        elif isinstance(self.edge_index, Tensor):
            node_subgraph = subgraph(node_index, self.edge_index,
                                     self.edge_weight,
                                     num_nodes=self.num_nodes,
                                     relabel_nodes=True)
            self.edge_index, self.edge_weight = node_subgraph
        for key, value in self.items():
            if key == 'edge_index' or key == 'edge_weight':
                continue
            if key in self.pattern and 'n' in self.pattern[key]:
                node_dim = self.pattern[key].split(' ').index('n')
                self[key] = torch.index_select(value, node_dim, node_index)
                if key in self.transform:
                    scaler = self.transform[key]
                    self.transform[key] = scaler.slice(node_index=node_index)
