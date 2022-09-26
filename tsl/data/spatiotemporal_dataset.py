from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Mapping, List, Union, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DatetimeIndex
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.typing import Adj
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor

from tsl.typing import DataArray, TemporalIndex, IndexSlice, TensArray, \
    SparseTensArray
from .batch_map import BatchMap, BatchMapItem
from .data import Data
from .mixin import DataParsingMixin
from .preprocessing.scalers import Scaler, ScalerModule
from .casting import SynchMode, WINDOW, HORIZON, STATIC
from ..ops.pattern import outer_pattern, broadcast, take

_WINDOWING_KEYS = ['target', 'window', 'delay', 'horizon', 'stride']


class SpatioTemporalDataset(Dataset, DataParsingMixin):
    r"""Base class for structures that are bridges between Datasets and Models.

    A :class:`SpatioTemporalDataset` takes as input a
    :class:`~tsl.datasets.Dataset` and
    build a proper structure to feed deep models.

    Args:
        target (DataArray): Data relative to the primary channels.
        index (TemporalIndex, optional): Temporal indices for the data.
            (default: :obj:`None`)
        mask (DataArray, optional): Boolean mask denoting if signal in data is
            valid (1) or not (0).
            (default: :obj:`None`)
        connectivity (SparseTensArray, tuple, optional): The adjacency matrix
            defining nodes' relational information. It can be either a
            dense/sparse matrix :math:`\mathbf{A} \in \mathbb{R}^{N \times N}`
            or an (:obj:`edge_index` :math:`\in \mathbb{N}^{2 \times E}`,
            :obj:`edge_weight` :math:`\in \mathbb{R}^{E})` tuple. The input
            layout will be preserved (e.g., a sparse matrix will be stored as a
            :class:`torch_sparse.SparseTensor`). In any case, the connectivity
            will be stored in the attribute :obj:`edge_index`, and the weights
            will be eventually stored as :obj:`edge_weight`.
            (default: :obj:`None`)
        covariates (dict, optional): Dictionary of exogenous channels with label.
            An :obj:`exogenous` element is a temporal array with node- or graph-
            level channels which are covariates to the main signal. The temporal
            dimension must be equal to the temporal dimension of data, as well
            as the number of nodes if the exogenous is node-level.
            (default: :obj:`None`)
        input_map (BatchMap or dict, optional): Defines how data, exogenous and
            attributes are mapped to the input of dataset samples. Keys in the
            mapping are keys in :obj:`item.input`, while values are
            :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        target_map (BatchMap or dict, optional): Defines how data, exogenous and
            attributes are mapped to the target of dataset samples. Keys in the
            mapping are keys in :obj:`item.target`, while values are
            :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        trend (DataArray, optional): Trend paired with main signal. Must be of
            the same shape of `data`.
            (default: :obj:`None`)
        scalers (Mapping or None): Dictionary of scalers that must be used for
            data preprocessing.
            (default: :obj:`None`)
        window (int): Length (in number of steps) of the lookback window.
        horizon (int): Length (in number of steps) of the prediction horizon.
        delay (int): Offset (in number of steps) between end of window and start
            of horizon.
        stride (int): Offset (in number of steps) between a sample and the next
            one.
        window_lag (int): Sampling frequency (in number of steps) in lookback
            window.
        horizon_lag (int): Sampling frequency (in number of steps) in prediction
            horizon.
    """

    def __init__(self, target: DataArray,
                 index: Optional[TemporalIndex] = None,
                 mask: Optional[DataArray] = None,
                 connectivity: Optional[
                     Union[SparseTensArray, Tuple[DataArray]]] = None,
                 covariates: Optional[Mapping[str, DataArray]] = None,
                 input_map: Optional[Union[Mapping, BatchMap]] = None,
                 target_map: Optional[Union[Mapping, BatchMap]] = None,
                 scalers: Optional[Mapping[str, Scaler]] = None,
                 trend: Optional[DataArray] = None,
                 window: int = 12,
                 horizon: int = 1,
                 delay: int = 0,
                 stride: int = 1,
                 window_lag: int = 1,
                 horizon_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):
        super(SpatioTemporalDataset, self).__init__()
        # Set name
        self.name = name if name is not None else self.__class__.__name__
        self.precision = precision

        # Store windowing information
        self.window = window
        self.delay = delay
        self.horizon = horizon
        self.stride = stride
        self.window_lag = window_lag
        self.horizon_lag = horizon_lag

        # Initialize private data holders
        self._covariates = dict()
        self._indices = None
        self.input_map = BatchMap()
        self.target_map = BatchMap()

        # Store time information
        # if index is not explicitly passed and data is a dataframe, defaults to
        # data's index
        if index is None and isinstance(target, pd.DataFrame):
            if isinstance(target.index, DatetimeIndex):
                index = target.index
        self.index = index

        # Set dataset's target signals
        self.target: Tensor = self._parse_target(target)

        # Store mask
        self.mask: Optional[Tensor] = None
        self.set_mask(mask)

        # Store adj
        self.edge_index: Optional[Adj] = None
        self.edge_weight: Optional[Tensor] = None
        self.set_connectivity(connectivity)

        # Store covariates (e.g., exogenous and attributes)
        self.reset_input_map()
        if covariates is not None:
            for name, value in covariates.items():
                self.add_covariate(name, **self._value_to_kwargs(value))

        # Updated input map (i.e., how to map data, exogenous and attribute
        # inside item.input)
        if input_map is not None:
            self.set_input_map(input_map)

        # Updated target map (i.e., how to map data, exogenous and attribute
        # inside item.target)
        self.reset_target_map()
        if target_map is not None:
            self.set_target_map(target_map)

        # Store preprocessing options
        # A scaler is a module that transforms data with a linear operation
        self.scalers: dict = dict()
        self._batch_scalers: dict = dict()
        if scalers is not None:
            for k, v in scalers.items():
                self.add_scaler(k, v)

        # Target's trend (i.e., 't n f' tensor to be removed when target
        # is preprocessed)
        self.trend: Optional[Tensor] = None
        # handle trend as bias in target scaler, so cache actual one (if any)
        # to restore after trend update/deletion
        self.__target_bias: Optional[Tensor] = None
        if trend is not None:
            self.set_trend(trend)

    def __repr__(self):
        return "{}(n_samples={}, n_nodes={}, n_channels={})" \
            .format(self.name, len(self), self.n_nodes, self.n_channels)

    def __getitem__(self, item) -> Data:
        if isinstance(item, int) and item < 0:
            # compute item's actual index
            item = self._indices.size(0) + item
        else:
            # convert slice to indexes
            item = self._get_time_index(item, layout='index')
        return self.get(item)

    def __contains__(self, item) -> bool:
        return item in self.keys

    def __len__(self) -> int:
        return len(self._indices)

    def __getattr__(self, item):
        if '_covariates' in self.__dict__ and item in self._covariates:
            return self._covariates[item]['value']
        if 'input_map' in self.__dict__ and item in self.input_map:
            return self.collate_item_elem(item)[0]
        raise AttributeError(f"'{self.__class__.__name__}' object has no "
                             f"attribute '{item}'")

    def __setattr__(self, key, value):
        super(SpatioTemporalDataset, self).__setattr__(key, value)
        if key in _WINDOWING_KEYS and all([hasattr(self, attr)
                                           for attr in _WINDOWING_KEYS]):
            self._indices = torch.arange(0, self.n_steps - self.sample_span + 1,
                                         self.stride)

    def __delattr__(self, item):
        if item in _WINDOWING_KEYS:
            raise AttributeError(f"Cannot delete attribute '{item}'.")
        elif item == 'mask':
            self.set_mask(None)
        elif item == 'trend':
            self.set_trend(None)
        elif item in self._covariates:
            del self._covariates[item]
            if item in self.scalers:
                del self.scalers[item]
        else:
            super(SpatioTemporalDataset, self).__delattr__(item)

    # Dataset properties ######################################################

    @property
    def n_steps(self) -> int:
        """Total number of time steps in the dataset."""
        return self.target.size(0)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the dataset."""
        return self.target.size(1)

    @property
    def n_channels(self) -> int:
        """Number of channels in dataset's target."""
        return self.target.size(-1)

    @property
    def n_edges(self) -> Optional[int]:
        """Number of edges in the dataset, if a connectivity is set."""
        if isinstance(self.edge_index, Tensor):
            return self.edge_index.size(1)
        elif isinstance(self.edge_index, SparseTensor):
            return self.edge_index.numel()
        return None

    @property
    def shape(self) -> tuple:
        return tuple(self.target.size())

    @property
    def patterns(self) -> dict:
        """Shows the dimension of dataset's tensors in a more informative way.

        The pattern mapping can be useful to glimpse on how data are arranged.
        The convention we use is the following:

          * 't' stands for “number of time steps”
          * 'n' stands for “number of nodes”
          * 'f' stands for “number of features” (per node)
          * 'e' stands for “number edges”
        """
        patterns = {'target': 't n f'}
        # add mask pattern
        if self.mask is not None:
            patterns['mask'] = 't n f'
        # add connectivity patterns
        if self.edge_index is not None:
            patterns['edge_index'] = '2 e' if isinstance(self.edge_index,
                                                         Tensor) else 'n n'
            if self.edge_weight is not None:
                patterns['edge_weight'] = 'e f'
        # add covariates patterns
        patterns.update({name: attr['pattern']
                         for name, attr in self._covariates.items()})
        return patterns

    @property
    def batch_patterns(self) -> dict:
        """Shows the dimension of dataset's tensors in a more informative way.

        The pattern mapping can be useful to glimpse on how data are arranged.
        The convention we use is the following:
            + 't' stands for “number of time steps”
            + 'n' stands for “number of nodes”
            + 'f' stands for “number of features” (per node)
            + 'e' stands for “number edges”
        """
        # add input map patterns
        patterns = {name: attr.pattern
                    for name, attr in self.input_map.items()}
        # add mask pattern
        if self.mask is not None:
            patterns['mask'] = 't n f'
        # add connectivity patterns
        if self.edge_index is not None:
            patterns['edge_index'] = '2 e' if isinstance(self.edge_index,
                                                         Tensor) else 'n n'
            if self.edge_weight is not None:
                patterns['edge_weight'] = 'e f'
        # add target map patterns
        patterns.update({name: attr.pattern
                         for name, attr in self.target_map.items()})
        return patterns

    @property
    def keys(self) -> list:
        """Keys in dataset."""
        return list(self.patterns.keys())

    @property
    def batch_keys(self) -> list:
        """Keys in dataset item."""
        return list(self.batch_patterns.keys())

    # Indexing

    @property
    def horizon_offset(self) -> int:
        """Lag of starting step of horizon."""
        return self.window + self.delay

    @property
    def sample_span(self) -> int:
        """Total number of steps of an item, including window and horizon."""
        return max(self.horizon_offset + self.horizon, self.window)

    @property
    def samples_offset(self) -> int:
        """Difference (in number of steps) between two items."""
        return int(np.ceil(self.window / self.stride))

    @property
    def indices(self) -> Tensor:
        """Indices of the dataset. The :obj:`i`-th item is mapped to
        :obj:`indices[i]`"""
        return self._indices

    # Covariates properties

    @property
    def covariates(self) -> dict:
        """Mapping of dataset's covariates."""
        return {name: attr['value'] for name, attr in self._covariates.items()}

    @property
    def exogenous(self) -> dict:
        """Time-varying covariates of the dataset's target."""
        return {name: attr['value'] for name, attr in self._covariates.items()
                if 't' in attr['pattern']}

    @property
    def attributes(self) -> dict:
        """Static features related to the dataset."""
        return {name: attr['value'] for name, attr in self._covariates.items()
                if 't' not in attr['pattern']}

    @property
    def n_covariates(self) -> int:
        """Number of covariates in the dataset."""
        return len(self._covariates)

    # flags

    @property
    def has_connectivity(self) -> bool:
        return self.edge_index is not None

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_covariates(self) -> bool:
        return self.n_covariates > 0

    # Map Dataset to item #####################################################

    @property
    def targets(self) -> BatchMap:
        return self.target_map

    def reset_input_map(self):
        self._clear_batch_map('input')
        self.input_map['x'] = BatchMapItem('target', SynchMode.WINDOW,
                                           preprocess=True, cat_dim=None,
                                           pattern='t n f',
                                           shape=self.shape)
        for name, attr in self._covariates.items():
            self.input_map[name] = BatchMapItem(name, SynchMode.WINDOW,
                                                preprocess=True, cat_dim=None,
                                                pattern=attr['pattern'],
                                                shape=attr.size())

    def reset_target_map(self):
        self._clear_batch_map('target')
        self.target_map['y'] = BatchMapItem('target', SynchMode.HORIZON,
                                            preprocess=False, cat_dim=None,
                                            pattern='t n f',
                                            shape=self.shape)

    def set_input_map(self, input_map=None, **kwargs):
        self._clear_batch_map('input')
        self.update_input_map(input_map, **kwargs)

    def set_target_map(self, target_map=None, **kwargs):
        self._clear_batch_map('target')
        self.update_target_map(target_map, **kwargs)

    def update_input_map(self, input_map=None, **kwargs):
        self._update_batch_map('input', input_map, **kwargs)

    def update_target_map(self, target_map=None, **kwargs):
        self._update_batch_map('target', target_map, **kwargs)

    def _clear_batch_map(self, endpoint):
        batch_map = getattr(self, f"{endpoint}_map")
        for key in batch_map:
            if key in self._batch_scalers:
                del self._batch_scalers[key]
        setattr(self, f"{endpoint}_map", BatchMap())

    def _update_batch_map(self, endpoint, batch_map=None, **kwargs):
        # check batch map
        if batch_map is None:
            batch_map = BatchMap()
        elif not isinstance(batch_map, Mapping):
            raise TypeError(f"Type {type(batch_map)} is not valid for "
                            f"`{endpoint}_map` (must be a mapping).")
        # update from kwargs
        batch_map.update(**kwargs)
        # update endpoint_map
        endpoint_map = getattr(self, f"{endpoint}_map")
        endpoint_map.update(**batch_map)
        # update 'pattern' and 'shape' to added/updated keys
        for name in batch_map:
            item: BatchMapItem = endpoint_map[name]
            # keys sanity check and compute pattern and shape
            keys = item.keys
            if len(keys) > 1:
                tensor, scaler, pattern = self.collate_keys(keys,
                                                            cat_dim=item.cat_dim,
                                                            return_pattern=True)
                item.shape, item.pattern = tuple(tensor.size()), pattern
                if scaler is not None:
                    self._batch_scalers[name] = scaler
            else:
                item.shape = tuple(getattr(self, keys[0]).size())
                item.pattern = self.patterns[keys[0]]

    # Getters #################################################################

    def get(self, item):

        ndim = item.ndim if isinstance(item, Tensor) else 0
        pattern = {name: ('b ' * ndim + pattern) if 't' in pattern else pattern
                   for name, pattern in self.batch_patterns.items()}

        sample = Data(pattern=pattern)

        # get input synchronized with window
        if self.window > 0:
            wdw_idxs = self.get_window_indices(item)
            self._add_to_sample(sample, WINDOW, 'input', time_index=wdw_idxs)
            self._add_to_sample(sample, WINDOW, 'target', time_index=wdw_idxs)

        # get input synchronized with horizon
        hrz_idxs = self.get_horizon_indices(item)
        self._add_to_sample(sample, HORIZON, 'input', time_index=hrz_idxs)
        self._add_to_sample(sample, HORIZON, 'target', time_index=hrz_idxs)

        # get static data
        self._add_to_sample(sample, STATIC, 'input')
        self._add_to_sample(sample, STATIC, 'target')

        # get connectivity
        if self.edge_index is not None:
            sample.input['edge_index'] = self.edge_index
            if self.edge_weight is not None:
                sample.input['edge_weight'] = self.edge_weight

        # get mask (if any)
        if self.mask is not None:
            sample.mask = self.mask[hrz_idxs]

        return sample

    def expand_scaler(self, key: str, pattern: Optional[str] = None,
                      time_index: Union[List, Tensor] = None,
                      node_index: Union[List, Tensor] = None) \
            -> Optional[ScalerModule]:
        # check if there is a scaler
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        elif key not in self.scalers:
            return None
        # convert indices
        time_index = self._get_time_index(time_index, layout='index')
        node_index = self._get_time_index(node_index, layout='index')
        # get params
        if pattern is None:
            return self.scalers[key]
        # if there is an out-pattern, create new scaler
        scaler = ScalerModule(self.scalers[key], pattern=pattern)
        pattern = self.patterns[key] + ' -> ' + pattern
        scaler.bias = broadcast(scaler.bias, pattern,
                                time_index=time_index,
                                node_index=node_index)
        scaler.scale = broadcast(scaler.scale, pattern,
                                 time_index=time_index,
                                 node_index=node_index)
        return scaler

    def expand_tensor(self, key: str, pattern: str,
                      time_index: Union[List, Tensor] = None,
                      node_index: Union[List, Tensor] = None):
        x = getattr(self, key)
        pattern = self.patterns[key] + ' -> ' + pattern
        x = broadcast(x, pattern, t=self.n_steps, n=self.n_nodes,
                      time_index=time_index, node_index=node_index)
        return x

    def get_tensor(self, key: str, preprocess: bool = False,
                   time_index: Union[List, Tensor] = None,
                   node_index: Union[List, Tensor] = None) \
            -> Tuple[Tensor, Optional[ScalerModule]]:
        # get dataset item
        if key not in self.keys:
            raise KeyError(f"{key} not in dataset {self.name}.")

        # convert indices
        time_index = self._get_time_index(time_index, layout='index')
        node_index = self._get_time_index(node_index, layout='index')
        x = take(getattr(self, key), self.patterns[key],
                 time_index=time_index, node_index=node_index)

        # get scaler (if any)
        scaler = None
        if key in self.scalers is not None:
            scaler = self.scalers[key].slice(time_index=time_index,
                                             node_index=node_index)
            if preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_item_elem(self, key: str,
                          time_index: Union[List, Tensor] = None,
                          node_index: Union[List, Tensor] = None) \
            -> Tuple[Tensor, Optional[ScalerModule]]:
        # get batch item
        if key in self.input_map:
            itm = self.input_map[key]
        elif key in self.target_map:
            itm = self.target_map[key]
        else:
            raise KeyError(f"{key} not in any batch map of {self.name}.")

        # expand and concatenate tensors
        x = torch.cat([self.expand_tensor(k, itm.pattern,
                                          time_index, node_index)
                       for k in itm.keys], dim=itm.cat_dim)

        # get scaler (if any)
        scaler = None
        if key in self._batch_scalers is not None:
            scaler = self._batch_scalers[key].slice(time_index=time_index,
                                                    node_index=node_index)
            if itm.preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_keys(self, keys: Iterable, preprocess: bool = False,
                     time_index: Union[List, Tensor] = None,
                     node_index: Union[List, Tensor] = None,
                     cat_dim: Optional[int] = None,
                     return_pattern: bool = False):
        if any([key not in self.keys for key in keys]):
            unmatch = set(keys).difference(self.keys)
            raise KeyError(f"{unmatch} not in {self.name}.")
        pattern = outer_pattern([self.patterns[key] for key in keys])
        tensors, scalers = list(), list()
        for key in keys:
            tensor = self.expand_tensor(key, pattern, time_index, node_index)
            scaler = self.expand_scaler(key, pattern, time_index, node_index)
            if preprocess and scaler is not None:
                tensor = scaler(tensor)
            tensors.append(tensor)
            scalers.append(scaler)
        if len(tensors) == 1:
            if return_pattern:
                return tensors[0], scalers[0], pattern
            return tensors[0], scalers[0]
        if cat_dim is not None:
            scalers = ScalerModule.cat(scalers, dim=cat_dim,
                                       sizes=[t.size() for t in tensors])
            tensors = torch.cat(tensors, dim=cat_dim)
        if return_pattern:
            return tensors, scalers, pattern
        return tensors, scalers

    # Getters helpers

    def _get_time_index(self, time_index=None, layout='index'):
        if time_index is None:
            return slice(None) if layout == 'slice' else None
        if isinstance(time_index, slice):
            if layout == 'slice':
                return time_index
            return torch.arange(max(time_index.start or 0, 0),
                                min(time_index.stop or len(self), len(self)),
                                time_index.step or 1)
        if not isinstance(time_index, Tensor):
            time_index = torch.as_tensor(time_index, dtype=torch.long)
        if layout == 'mask':
            mask = torch.zeros(self.n_steps, dtype=torch.bool)
            mask[time_index] = True
            return mask
        return time_index

    def _get_node_index(self, node_index=None, layout='index'):
        if node_index is None:
            return slice(None) if layout == 'slice' else None
        if isinstance(node_index, slice):
            if layout == 'slice':
                return node_index
            return torch.arange(max(node_index.start or 0, 0),
                                min(node_index.stop or self.n_nodes,
                                    self.n_nodes),
                                node_index.step or 1)
        if not isinstance(node_index, Tensor):
            node_index = torch.as_tensor(node_index, dtype=torch.long)
        if layout == 'mask':
            mask = torch.zeros(self.n_nodes, dtype=torch.bool)
            mask[node_index] = True
            return mask
        return node_index

    def _add_to_sample(self, out, synch_mode, endpoint='input',
                       time_index=None, node_index=None):
        batch_map = getattr(self, f"{endpoint}_map")
        for key, item in batch_map.by_synch_mode(synch_mode).items():
            if len(item.keys) > 1:
                tensor, scaler = self.collate_item_elem(key,
                                                        time_index=time_index,
                                                        node_index=node_index)
            else:
                tensor, scaler = self.get_tensor(item.keys[0],
                                                 preprocess=item.preprocess,
                                                 time_index=time_index,
                                                 node_index=node_index)
            getattr(out, endpoint)[key] = tensor
            if scaler is not None:
                out.transform[key] = scaler

    # Setters #################################################################

    def set_data(self, data: DataArray):
        r"""Set sequence of primary channels at :obj:`self.data`."""
        self.target = self._parse_target(data)

    def set_mask(self, mask: Optional[DataArray]):
        r"""Set mask of target channels, i.e., a bool for each (node, time
        step, channel) triplet denoting if corresponding value in target is
        observed (obj:`True`) or not (obj:`False`)."""
        if mask is not None:
            mask = self._parse_target(mask)
            if mask.dtype not in [torch.bool, torch.uint8]:
                raise RuntimeError("Mask is not boolean, accepted types are "
                                   f"{[torch.bool, torch.uint8]}.")
            # check mask length is equal to target's length
            self._check_same_dim(mask.size(0), 'n_steps', 'mask')
            # check mask nodes/features are broadcastable to
            # target's nodes/features
            self._check_same_dim(mask.size(1), 'n_nodes', 'mask',
                                 allow_broadcasting=True)
            self._check_same_dim(mask.size(-1), 'n_channels', 'mask',
                                 allow_broadcasting=True)
        self.mask = mask

    def set_connectivity(self,
                         connectivity: Union[SparseTensArray, Tuple[DataArray]],
                         target_layout: Optional[str] = None):
        r"""Set dataset connectivity.

        The input can be either a
        dense/sparse matrix :math:`\mathbf{A} \in \mathbb{R}^{N \times N}`
        or an (:obj:`edge_index` :math:`\in \mathbb{N}^{2 \times E}`,
        :obj:`edge_weight` :math:`\in \mathbb{R}^{E})` tuple. If
        :obj:`target_layout` is :obj:`None`, the input layout will be
        preserved (e.g., a sparse matrix will be stored as a
        :class:`torch_sparse.SparseTensor`), otherwise the connectivity is
        converted to the specified layout. In any case, the connectivity
        will be stored in the attribute :obj:`edge_index`, and the weights
        will be eventually stored as :obj:`edge_weight`.

        Args:
            connectivity (SparseTensArray, tuple, optional): The connectivity
            target_layout (str, optional): If specified, the input connectivity
                is converted to this layout. Possible options are [dense,
                sparse, edge_index]. If :obj:`None`, the target layout is
                inferred from the input.
                (default: :obj:`None`)
        """
        self.edge_index, self.edge_weight = self._parse_adj(connectivity,
                                                            target_layout)

    # Setter for covariates

    def add_covariate(self, name: str, value: DataArray,
                      pattern: Optional[str] = None,
                      add_to_input_map: bool = True,
                      synch_mode: Optional[SynchMode] = None,
                      preprocess: bool = True):
        r"""Add covariate to the dataset. Examples of covariate are
        exogenous signals (in the form of dynamic multidimensional data) or
        static attributes (e.g., graph/node metadata). Parameter :obj:`pattern`
        specifies what each axis refers to:

        - 't': temporal dimension;
        - 'n': node dimension;
        - 'c'/'f': channels/features dimension.

        For instance, the pattern of a node-level covariate is 't n f', while a
        pairwise metric between nodes has pattern 'n n'.

        Args:
            name (str): the name of the object. You can then access the added
                object as :obj:`dataset.{name}`.
            value (DataArray): the object to be added. Can be a
                :class:`~pandas.DataFrame`, a :class:`~numpy.ndarray` or a
                :class:`~torch.Tensor`.
            pattern (str, optional): the pattern of the object. A pattern
                specifies what each axis refers to:

                - 't': temporal dimension;
                - 'n': node dimension;
                - 'c'/'f': channels/features dimension.

                If :obj:`None`, the pattern is inferred from the shape.
                (default :obj:`None`)
            add_to_input_map (bool): Whether to map the covariate to dataset
                item when calling :obj:`get` methods.
                (default: :obj:`True`)
            synch_mode (SynchMode): How to synchronize the exogenous variable
                inside dataset item, i.e., with the window slice
                (:obj:`SynchMode.WINDOW`) or horizon (:obj:`SynchMode.HORIZON`).
                It applies only for time-varying covariates.
                (default: :obj:`SynchMode.WINDOW`)
            preprocess (bool): If :obj:`True` and the dataset has a scaler with
                same key, then data are scaled when calling :obj:`get` methods.
                (default: :obj:`True`)
        """
        # validate name. name cannot be an attribute of self, but allow override
        self._check_name(name)
        value, pattern = self._parse_covariate(value, pattern, name=name)
        self._covariates[name] = dict(value=value, pattern=pattern)
        if add_to_input_map:
            self.input_map[name] = BatchMapItem(name, synch_mode, preprocess,
                                                cat_dim=None, pattern=pattern,
                                                shape=value.size())

    def add_exogenous(self, name: str, value: DataArray,
                      node_level: bool = True,
                      add_to_input_map: bool = True,
                      synch_mode: SynchMode = WINDOW,
                      preprocess: bool = True):
        r"""Shortcut method to add a time-varying covariate.

        Exogenous variables are time-varying covariates of the dataset's target.
        They can either be graph-level (i.e., with same temporal
        length as :obj:`target` but with no node dimension) or node-level (i.e.,
        with same temporal and node size as :obj:`target`).

        Args:
            name (str): The name of the exogenous variable. If the name starts
                with "global_", the variable is assumed to be graph-level
                (overriding parameter :obj:`node_level`), and the "global_"
                prefix is removed from the name.
            value (DataArray): The data sequence. Can be a
                :class:`~pandas.DataFrame`, a :class:`~numpy.ndarray` or a
                :class:`~torch.Tensor`.
            node_level (bool): Whether the input variable is node- or
                graph-level. If a 2-dimensional array is given and node-level is
                :obj:`True`, it is assumed that the input has one channel.
                (default: :obj:`True`)
            add_to_input_map (bool): Whether to map the exogenous variable to
                dataset item when calling :obj:`get` methods.
                (default: :obj:`True`)
            synch_mode (SynchMode): How to synchronize the exogenous variable
                inside dataset item, i.e., with the window slice
                (:obj:`SynchMode.WINDOW`) or horizon (:obj:`SynchMode.HORIZON`).
                (default: :obj:`SynchMode.WINDOW`)
            preprocess (bool): If :obj:`True` and the dataset has a scaler with
                same key, then data are scaled when calling :obj:`get` methods.
                (default: :obj:`True`)

        Returns:
            SpatioTemporalDataset: the dataset with added exogenous.
        """
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        pattern = 't n f' if node_level else 't f'
        self.add_covariate(name, value, pattern, synch_mode=synch_mode,
                           add_to_input_map=add_to_input_map,
                           preprocess=preprocess)

    # Setters for preprocessing

    def set_trend(self, trend: Optional[DataArray]):
        r"""Set trend of dataset's target data."""
        if trend is not None:
            trend = self._parse_target(trend)
            # check trend length is equal to target's length
            self._check_same_dim(trend.size(0), 'n_steps', 'mask')
            # check trend nodes/features are broadcastable to
            # target's nodes/features
            self._check_same_dim(trend.size(1), 'n_nodes', 'mask',
                                 allow_broadcasting=True)
            self._check_same_dim(trend.size(-1), 'n_channels', 'mask',
                                 allow_broadcasting=True)
            if 'target' in self.scalers:
                if self.__target_bias is None:
                    self.__target_bias = self.scalers['target'].bias
                self.scalers['target'].bias = trend
        else:
            if 'target' in self.scalers:
                self.scalers['target'].bias = self.__target_bias
                # refresh maps
                self.update_input_map(self.input_map)
                self.update_target_map(self.target_map)
        self.trend = trend

    def add_scaler(self, key: str, scaler: Union[Scaler, ScalerModule]):
        r"""Add a :class:`tsl.data.preprocessing.Scaler` for the object indexed
        by :obj:`key` in the dataset.

        Args:
            key (str): The name of the variable associated to the scaler. It
                must be a temporal variable, i.e., :obj:`data` or an exogenous.
            scaler (Scaler): The :class:`~tsl.data.preprocessing.Scaler`.
        """
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        # copy to ScalerModule
        scaler = ScalerModule(scaler)
        pattern = self.patterns[key]
        self._check_pattern(scaler.bias, pattern,
                            name=f"{key} (scaler)", allow_broadcasting=True)
        self._check_pattern(scaler.scale, pattern,
                            name=f"{key} (scaler)", allow_broadcasting=True)
        if key == 'target' and self.trend is not None:
            self.__target_bias = scaler.bias
            scaler.bias = scaler.bias + self.trend
        scaler.pattern = pattern
        self.scalers[key] = scaler

    # Dataset trimming ########################################################

    def reduce(self, time_index: Optional[IndexSlice] = None,
               node_index: Optional[IndexSlice] = None):
        """Reduce the dataset in terms of number of steps and nodes. Returns a
        copy of the reduced dataset.

        If dataset has a connectivity, edges ending to or starting from removed
        nodes will be removed as well.

        Args:
            time_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
            node_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
        """
        return deepcopy(self).reduce_(time_index, node_index)

    def reduce_(self, time_index: Optional[IndexSlice] = None,
                node_index: Optional[IndexSlice] = None):
        """Reduce the dataset in terms of number of steps and nodes. This is an
        inplace operation.

        If dataset has a connectivity, edges ending to or starting from removed
        nodes will be removed as well.

        Args:
            time_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
            node_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
        """
        # use slice to reduce known tensor
        time_slice = self._get_time_index(time_index, layout='slice')
        node_slice = self._get_node_index(node_index, layout='slice')
        # use index to reduce using index-fed functions
        time_index = self._get_time_index(time_index, layout='index')
        node_index = self._get_node_index(node_index, layout='index')
        try:
            if self.edge_index is not None and node_index is not None:
                if isinstance(self.edge_index, SparseTensor):
                    self.edge_index = self.edge_index[node_index, node_index]
                else:
                    node_subgraph = subgraph(node_slice, self.edge_index,
                                             self.edge_weight,
                                             num_nodes=self.n_nodes,
                                             relabel_nodes=True)
                    self.edge_index, self.edge_weight = node_subgraph
            self.target = self.target[time_slice, node_slice]
            if self.index is not None:
                self.index = self.index[time_index.numpy()]
            if self.mask is not None:
                self.mask = self.mask[time_slice, node_slice]
            if self.trend is not None:
                self.trend = self.trend[time_slice, node_slice]
            for name, attr in self._covariates.items():
                x, scaler = self.get_tensor(name, time_index=time_index,
                                            node_index=node_index)
                attr['value'] = x
                if scaler is not None:
                    self.scalers[name] = scaler
        except Exception as e:
            raise e
        return self

    @contextmanager
    def change_windowing(self, **kwargs):
        default, indices = dict(), self._indices
        try:
            assert all([k in _WINDOWING_KEYS[1:] for k in kwargs])
            for k, v in kwargs.items():
                default[k] = getattr(self, k)
                setattr(self, k, v)
            yield self
        finally:
            for k, v in default.items():
                setattr(self, k, v)
            self._indices = indices

    # Indexing ################################################################

    def get_window_indices(self, item):
        idx = self._indices[item]
        window_range = range(0, self.window, self.window_lag)
        return torch.stack([idx + t for t in window_range]).T

    def get_horizon_indices(self, item):
        idx = self._indices[item]
        horizon_range = range(self.horizon_offset, self.horizon_offset +
                              self.horizon, self.horizon_lag)
        return torch.stack([idx + t for t in horizon_range]).T

    def set_indices(self, indices: TensArray):
        indices = torch.as_tensor(indices, dtype=torch.long)
        max_index = self.n_steps - self.sample_span
        assert all((indices >= 0) & (indices <= max_index)), \
            f"indices must be in the range [0, {max_index}] for {self.name}."
        self._indices = indices

    def expand_indices(self, indices=None, unique=False, merge=False):
        indices = torch.arange(len(self)) if indices is None else indices

        ds_indices = dict()
        ds_indices['window'] = self.get_window_indices(indices)
        ds_indices['horizon'] = self.get_horizon_indices(indices)

        if merge:
            return torch.unique(torch.cat([ds_indices['window'].view(-1),
                                           ds_indices['horizon'].view(-1)]))

        if unique:
            ds_indices['window'] = torch.unique(ds_indices['window'])
            ds_indices['horizon'] = torch.unique(ds_indices['horizon'])

        return ds_indices

    def overlapping_indices(self, idxs1, idxs2,
                            synch_mode: Union[SynchMode, str] = WINDOW,
                            as_mask=False):
        if isinstance(synch_mode, SynchMode):
            synch_mode = synch_mode.name
        idxs1, idxs2 = np.asarray(idxs1), np.asarray(idxs2)
        ts1 = self.expand_indices(idxs1)[synch_mode.lower()].numpy()
        ts2 = self.expand_indices(idxs2)[synch_mode.lower()].numpy()
        common_ts = np.intersect1d(ts1, ts2)
        is_overlapping = lambda sample: np.any(np.in1d(sample, common_ts))
        m1 = np.apply_along_axis(is_overlapping, 1, ts1)
        m2 = np.apply_along_axis(is_overlapping, 1, ts2)
        if as_mask:
            return m1, m2
        return idxs1[m1], idxs2[m2]

    def data_timestamps(self, indices=None, unique=False) -> Optional[dict]:
        if self.index is None:
            return None
        ds_indices = self.expand_indices(indices, unique=unique)
        index = self.index if unique else self.index.to_numpy()
        ds_timestamps = {k: index[v.numpy()] for k, v in ds_indices.items()}
        return ds_timestamps

    # Representation ##########################################################

    def numpy(self):
        return np.asarray(self.target)

    def dataframe(self):
        columns = pd.MultiIndex.from_product([range(self.n_nodes),
                                              range(self.n_channels)],
                                             names=['nodes', 'channels'])
        data = self.numpy().reshape((-1, self.n_nodes * self.n_channels))
        return pd.DataFrame(data=data, index=self.index, columns=columns)

    # Utilities ###############################################################

    def save(self, filename: str) -> None:
        """Save :obj:`SpatioTemporalDataset` to disk.

        Args:
            filename (str): path to filename for storage.
        """
        torch.save(self, filename)

    @classmethod
    def load(cls, filename: str) -> "SpatioTemporalDataset":
        """Load instance of :obj:`SpatioTemporalDataset` from disk.

        Args:
            filename (str): path of :obj:`SpatioTemporalDataset`.
        """
        obj = torch.load(filename)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded file is not of class {cls}.")
        return obj

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--window', type=int, default=24)
        parser.add_argument('--horizon', type=int, default=24)
        parser.add_argument('--delay', type=int, default=0)
        parser.add_argument('--stride', type=int, default=1)
        parser.add_argument('--window-lag', type=int, default=1)
        parser.add_argument('--horizon-lag', type=int, default=1)
        return parser
