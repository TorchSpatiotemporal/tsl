from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Mapping, List, Union, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.utils import subgraph

from tsl.typing import TensArray, TemporalIndex, IndexSlice
from .batch import Batch
from .data import Data
from .batch_map import BatchMap, BatchMapItem
from .mixin import DataParsingMixin
from .preprocessing.scalers import Scaler, ScalerModule
from .utils import SynchMode, WINDOW, HORIZON, broadcast, outer_pattern

_WINDOWING_KEYS = ['data', 'window', 'delay', 'horizon', 'stride']


class SpatioTemporalDataset(Dataset, DataParsingMixin):
    r"""Base class for structures that are bridges between Datasets and Models.

    A :class:`SpatioTemporalDataset` takes as input a
    :class:`~tsl.datasets.Dataset` and
    build a proper structure to feed deep models.

    Args:
        data (TensArray): Data relative to the primary channels.
        index (TemporalIndex, optional): Temporal indices for the data.
            (default: :obj:`None`)
        mask (TensArray, optional): Boolean mask denoting if signal in data is
            valid (1) or not (0).
            (default: :obj:`None`)
        connectivity (TensArray, tuple, optional): The adjacency matrix defining nodes'
            relational information. It can be either a dense matrix
            :math:`\mathbf{A} \in \mathbb{R}^{N \times N}` or in COO format as a
            tuple (:obj:`edge_index` :math:`\in \mathbb{N}^{2 \times E}`,
            :obj:`edge_weight` :math:`\in \mathbb{R}^{E})`.
            (default: :obj:`None`)
        exogenous (dict, optional): Dictionary of exogenous channels with label.
            An :obj:`exogenous` element is a temporal array with node- or graph-
            level channels which are covariate to the main signal. The temporal
            dimension must be equal to the temporal dimension of data, as well
            as the number of nodes if the exogenous is node-level.
            (default: :obj:`None`)
        attributes (dict, optional):  Dictionary of static features with label.
            An :obj:`attributes` element is an array with node- or graph-level
            static features. In case of node-level attribute, the node dimension
            must be equal to the node dimension of data.
            (default: :obj:`None`)
        input_map (BatchMap or dict, optional): Defines how data, exogenous and
            attributes are mapped to the input of dataset samples. Keys in the
            mapping are keys in :obj:`item.input`, while values are
            :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        trend (TensArray, optional): Trend paired with main signal. Must be of
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

    def __init__(self, data: TensArray,
                 index: Optional[TemporalIndex] = None,
                 mask: Optional[TensArray] = None,
                 connectivity: Optional[
                     Union[TensArray, Tuple[TensArray]]] = None,
                 exogenous: Optional[Mapping[str, TensArray]] = None,
                 attributes: Optional[Mapping[str, TensArray]] = None,
                 input_map: Optional[Union[Mapping, BatchMap]] = None,
                 trend: Optional[TensArray] = None,
                 scalers: Optional[Mapping[str, Scaler]] = None,
                 window: int = 24,
                 horizon: int = 24,
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
        # Initialize private data holders
        self._exogenous = dict()
        self._attributes = dict()
        self.input_map = BatchMap()
        # Store data
        self.data: Tensor = self._parse_data(data)
        # Store time information
        self.index: Optional[TemporalIndex] = index
        # Store mask
        self.mask: Optional[Tensor] = self._parse_mask(mask)
        # Store adj
        self.edge_index, self.edge_weight = self._parse_adj(connectivity)
        # Store offset information
        self.window = window
        self.delay = delay
        self.horizon = horizon
        self.stride = stride
        self.window_lag = window_lag
        self.horizon_lag = horizon_lag
        # Updated input map (i.e., how to map data, exogenous and attribute
        # inside item)
        if input_map is None:
            input_map = self.default_input_map()
        self.set_input_map(input_map)
        # Store exogenous and attributes
        if exogenous is not None:
            for name, value in exogenous.items():
                self.add_exogenous(name, **self._exog_value_to_kwargs(value))
        if attributes is not None:
            for name, value in attributes.items():
                self.add_attribute(name, **self._attr_value_to_kwargs(value))
        # Store preprocessing options
        self.trend = None
        if trend is not None:
            self.set_trend(trend)
        self.scalers: dict = dict()
        if scalers is not None:
            for k, v in scalers.items():
                self.set_scaler(k, v)

    def __repr__(self):
        return "{}(n_samples={}, n_nodes={}, n_channels={})" \
            .format(self.name, len(self), self.n_nodes, self.n_channels)

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        keys = {'data'}. \
            union(self.exogenous.keys()). \
            union(self.attributes.keys())
        if self.mask is not None:
            keys.add('mask')
        return item in keys

    def __len__(self):
        return len(self._indices)

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
            return setattr(self, item, None)
        super(SpatioTemporalDataset, self).__delattr__(item)
        if item in self._exogenous:
            del self._exogenous[item]
        elif item in self._attributes:
            del self._attributes[item]

    # Map Dataset to item #####################################################

    @property
    def targets(self) -> BatchMap:
        return BatchMap(y=BatchMapItem('data', SynchMode.HORIZON,
                                       cat_dim=None, preprocess=False,
                                       n_channels=self.n_channels))

    def default_input_map(self) -> BatchMap:
        im = BatchMap(x=BatchMapItem('data', SynchMode.WINDOW, cat_dim=None,
                                     preprocess=True,
                                     n_channels=self.n_channels))
        for key, exo in self.exogenous.items():
            im[key] = BatchMapItem(key, SynchMode.WINDOW,
                                   cat_dim=None, preprocess=True,
                                   n_channels=exo.shape[-1])
        return im

    def set_input_map(self, input_map=None, **kwargs):
        if input_map is None:
            self.input_map = BatchMap()
        elif isinstance(input_map, BatchMap):
            self.input_map = input_map
        elif isinstance(input_map, Mapping):
            self.input_map = BatchMap(**input_map)
        else:
            raise TypeError(f"Type {type(input_map)} is not valid "
                            f"for `input_map`")
        self.update_input_map(**kwargs)

    def update_input_map(self, input_map=None, **kwargs):
        keys = []
        if input_map is not None:
            self.input_map.update(**input_map)
            keys += list(input_map.keys())
        self.input_map.update(**kwargs)
        keys += list(kwargs.keys())
        for key, item in self.input_map.items():
            if key in keys:
                item.n_channels = sum([getattr(self, k).shape[-1]
                                       for k in item.keys])

    @property
    def keys(self) -> list:
        keys = list(self.input_map.keys())
        keys += list(self.get_static_attributes().keys())
        if self.mask is not None:
            keys += ['mask']
        keys += list(self.targets.keys())
        return keys

    def _populate_input_frame(self, index, synch_mode, out):
        for key, value in self.input_map.by_synch_mode(synch_mode).items():
            tens, trans, pattern = self.get_tensors(value.keys,
                                                    cat_dim=value.cat_dim,
                                                    preprocess=value.preprocess,
                                                    step_index=index,
                                                    return_pattern=True)
            out.input[key] = tens
            if trans is not None:
                out.transform[key] = trans
            out.pattern[key] = pattern

    def _populate_target_frame(self, index, out):
        for key, value in self.targets.items():
            tens, trans, pattern = self.get_tensors(value.keys,
                                                    cat_dim=value.cat_dim,
                                                    preprocess=value.preprocess,
                                                    step_index=index,
                                                    return_pattern=True)
            out.target[key] = tens
            if trans is not None:
                out.transform[key] = trans
            out.pattern[key] = pattern

    def get(self, item):
        sample = Data()
        # get input synchronized with window
        if self.window > 0:
            wdw_idxs = self.get_window_indices(item)
            self._populate_input_frame(wdw_idxs, WINDOW, sample)
        # get input synchronized with horizon
        hrz_idxs = self.get_horizon_indices(item)
        self._populate_input_frame(hrz_idxs, HORIZON, sample)

        # get static attributes
        for key, value in self.get_static_attributes().items():
            sample.input[key] = value
            pattern = self.patterns.get(key)
            if pattern is not None:
                sample.pattern[key] = pattern

        # get mask (if any)
        if self.mask is not None:
            sample.mask = self.mask[hrz_idxs]
            sample.pattern['mask'] = 's n c'

        # get target
        self._populate_target_frame(hrz_idxs, sample)

        return sample

    # Setters #################################################################

    def set_data(self, data: TensArray):
        self.data = self._parse_data(data)

    def set_mask(self, mask: TensArray):
        self.mask = self._parse_mask(mask)

    def set_connectivity(self, connectivity: Union[TensArray,
                                                   Tuple[TensArray]]):
        self.edge_index, self.edge_weight = self._parse_adj(connectivity)

    def set_trend(self, trend: TensArray):
        self.trend: Tensor = self._parse_node_level_exogenous(trend, 'trend')

    def set_scaler(self, key: str, value: Scaler):
        self.scalers[key] = value

    # Setter for secondary data

    def add_exogenous(self, name: str, value: TensArray,
                      node_level: bool = True,
                      add_to_input_map: bool = True,
                      synch_mode: SynchMode = WINDOW,
                      preprocess: bool = True) -> "SpatioTemporalDataset":
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        # validate name
        self._check_name(name)
        # check if node-level or graph-level
        if node_level:
            value = self._parse_node_level_exogenous(value, name)
        else:
            value = self._parse_graph_level_exogenous(value, name)
        # add exogenous and set as attribute
        self._exogenous[name] = dict(value=value,
                                     node_level=node_level)
        setattr(self, name, value)
        if add_to_input_map:
            im = {name: BatchMapItem(name, synch_mode, preprocess,
                                     cat_dim=None)}
            self.update_input_map(im)
        return self

    def update_exogenous(self, name: str,
                         value: Optional[TensArray] = None,
                         node_level: Optional[bool] = None):
        if name not in self._exogenous:
            raise AttributeError(f"No exogenous named '{name}'.")
        # defaults to current value if None
        if value is None:
            value = self._exogenous[name]['value']
        # if node_level is provided, parse value according to node_level
        if node_level is True:
            value = self._parse_node_level_exogenous(value, name)
            self._exogenous[name]['node_level'] = node_level
        elif node_level is False:
            value = self._parse_graph_level_exogenous(value, name)
            self._exogenous[name]['node_level'] = node_level
        # update value (eventually)
        self._exogenous[name]['value'] = value
        setattr(self, name, value)

    def add_attribute(self, name: str, value: TensArray,
                      node_level: bool = True,
                      add_to_batch: bool = True) -> "SpatioTemporalDataset":
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        # validate name
        self._check_name(name)
        if node_level:
            value = self._parse_node_level_attribute(value, name)
        else:
            value = self._parse_graph_level_attribute(value)
        self._attributes[name] = dict(value=value, node_level=node_level,
                                      add_to_batch=add_to_batch)
        setattr(self, name, value)
        return self

    def update_attribute(self, name: str,
                         value: Optional[TensArray] = None,
                         node_level: Optional[bool] = None,
                         add_to_batch: Optional[bool] = None):
        if name not in self._attributes:
            raise AttributeError(f"No attribute named '{name}'.")
        # defaults to current value if None
        if value is None:
            value = self._attributes[name]['value']
        # if node_level is provided, parse value according to node_level
        if node_level is True:
            value = self._parse_node_level_attribute(value, name)
            self._attributes[name]['node_level'] = node_level
        elif node_level is False:
            value = self._parse_graph_level_attribute(value)
            self._attributes[name]['node_level'] = node_level
        if add_to_batch is not None:
            self._attributes[name]['add_to_batch'] = add_to_batch
        # update value (eventually)
        self._attributes[name]['value'] = value
        setattr(self, name, value)

    # Dataset properties ######################################################

    # Indexing

    @property
    def horizon_offset(self) -> int:
        return self.window + self.delay

    @property
    def sample_span(self) -> int:
        return max(self.horizon_offset + self.horizon, self.window)

    @property
    def samples_offset(self) -> int:
        return int(np.ceil(self.window / self.stride))

    @property
    def indices(self) -> Tensor:
        return self._indices

    # Shape

    @property
    def n_steps(self) -> int:
        return self.data.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.data.shape[1]

    @property
    def n_channels(self) -> int:
        return self.data.shape[-1]

    @property
    def n_edges(self) -> int:
        return self.edge_index.size(1) if self.edge_index is not None else None

    @property
    def patterns(self):
        patterns = dict(data='s n c')
        patterns.update({key: 's n c' if value['node_level'] else 's c'
                         for key, value in self._exogenous.items()})
        patterns.update({key: 'n c' if value['node_level'] else 'c'
                         for key, value in self._attributes.items()})
        for k, v in self.__dict__.items():
            if k.startswith('edge_') and v is not None:
                if k == 'edge_index':
                    patterns[k] = '2 e'
                else:
                    patterns[k] = 'e' + ' c' * (v.ndim - 1)
        return patterns

    # Secondary data

    @property
    def exogenous(self) -> dict:
        return {k: v['value'] for k, v in self._exogenous.items()}

    @property
    def attributes(self) -> dict:
        return {k: v['value'] for k, v in self._attributes.items()}

    # Methods for accessing tensor ############################################

    def get_transform_params(self, key: str, pattern: Optional[str] = None,
                             step_index: Union[List, Tensor] = None,
                             node_index: Union[List, Tensor] = None):
        if step_index is None:
            step_index = slice(None)
        params = dict()
        if key == 'data' and self.trend is not None:
            params['trend'] = self.trend[step_index]
            if node_index is not None:
                params['trend'] = params['trend'].index_select(1, node_index)
        if key in self.scalers:
            s_params = self.scalers[key].params()
            if pattern is not None:
                pattern = self.patterns[key] + ' -> ' + pattern
                s_params = {k: broadcast(p, pattern, n=1,
                                         node_index=node_index)
                            for k, p in s_params.items()}
            params.update(**s_params)
        return ScalerModule(**params) if len(params) else None

    def expand_tensor(self, key: str, pattern: str,
                      step_index: Union[List, Tensor] = None,
                      node_index: Union[List, Tensor] = None):
        x = getattr(self, key)
        pattern = self.patterns[key] + ' -> ' + pattern
        x = broadcast(x, pattern, s=self.n_steps, n=self.n_nodes,
                      step_index=step_index, node_index=node_index)
        return x

    def get_tensors(self, keys: Iterable, preprocess: bool = False,
                    step_index: Union[List, Tensor] = None,
                    node_index: Union[List, Tensor] = None,
                    cat_dim: Optional[int] = None,
                    return_pattern: bool = False):
        assert all([key in self for key in keys])
        pattern = outer_pattern([self.patterns[key] for key in keys])
        tensors, transforms = list(), list()
        for key in keys:
            tensor = self.expand_tensor(key, pattern, step_index, node_index)
            transform = self.get_transform_params(key, pattern,
                                                  step_index, node_index)
            if preprocess and transform is not None:
                tensor = transform(tensor)
            tensors.append(tensor)
            transforms.append(transform)
        if len(tensors) == 1:
            if return_pattern:
                return tensors[0], transforms[0], pattern
            return tensors[0], transforms[0]
        if cat_dim is not None:
            transforms = ScalerModule.cat(transforms, dim=cat_dim,
                                          sizes=[t.size() for t in tensors])
            tensors = torch.cat(tensors, dim=cat_dim)
        if return_pattern:
            return tensors, transforms, pattern
        return tensors, transforms

    def get_static_attributes(self) -> dict:
        static_attrs = {k: v['value'] for k, v in self._attributes.items()
                        if v['add_to_batch']}
        # add edge attrs
        static_attrs.update({k: v for k, v in self.__dict__.items() if
                             k.startswith('edge_') and v is not None})
        return static_attrs

    def data_timestamps(self, indices=None, unique=False) -> Optional[dict]:
        if self.index is None:
            return None
        ds_indices = self.expand_indices(indices, unique=unique)
        index = self.index.to_numpy()
        ds_timestamps = {k: index[v] for k, v in ds_indices.items()}
        return ds_timestamps

    # Dataset trimming ########################################################

    def reduce(self, step_index: Optional[IndexSlice] = None,
               node_index: Optional[IndexSlice] = None):
        return deepcopy(self).reduce_(step_index, node_index)

    def reduce_(self, step_index: Optional[IndexSlice] = None,
                node_index: Optional[IndexSlice] = None):
        if step_index is None:
            step_index = slice(None)
        if node_index is None:
            node_index = slice(None)
        try:
            if self.edge_index is not None:
                node_index = torch.arange(self.n_nodes)[node_index]
                node_subgraph = subgraph(node_index, self.edge_index,
                                         self.edge_weight,
                                         num_nodes=self.n_nodes,
                                         relabel_nodes=True)
                self.edge_index, self.edge_weight = node_subgraph
            self.data = self.data[step_index, node_index]
            if self.index is not None:
                self.index = self.index[step_index]
            if self.mask is not None:
                self.mask = self.mask[step_index, node_index]
            if self.trend is not None:
                self.trend = self.trend[step_index, node_index]
            for k, exo in self._exogenous.items():
                value = exo['value'][step_index]
                if exo['node_level']:
                    value = value[:, node_index]
                self.update_exogenous(k, value)
            for k, attr in self._attributes.items():
                if attr['node_level']:
                    self.update_attribute(k, attr['value'][node_index])
        except Exception as e:
            raise e
        return self

    @contextmanager
    def change_windowing(self, **kwargs):
        default = dict()
        try:
            assert all([k in _WINDOWING_KEYS[1:] for k in kwargs])
            for k, v in kwargs.items():
                default[k] = getattr(self, k)
                setattr(self, k, v)
            yield self
        finally:
            for k, v in default.items():
                setattr(self, k, v)

    # Indexing ################################################################

    def get_window_indices(self, item):
        idx = self._indices[item]
        return torch.arange(idx, idx + self.window, self.window_lag)

    def get_horizon_indices(self, item):
        idx = self._indices[item]
        return torch.arange(idx + self.horizon_offset,
                            idx + self.horizon_offset + self.horizon,
                            self.horizon_lag)

    def get_indices(self, item, synch_mode):
        if synch_mode is WINDOW:
            return self.get_window_indices(item)
        if synch_mode is HORIZON:
            return self.get_horizon_indices(item)
        return self.get_window_indices(item), self.get_horizon_indices(item)

    def expand_indices(self, indices=None, unique=False, merge=False):
        indices = np.arange(len(self._indices)) if indices is None else indices
        hrz_end = self.horizon_offset + self.horizon

        def expand_indices_range(rng_start, rng_end):
            allowable_offset = rng_end - rng_start - self.stride + 1
            contiguous = all(np.diff(indices) <= allowable_offset)
            if unique and contiguous:
                return np.arange(self._indices[indices[0]] + rng_start,
                                 self._indices[indices[-1]] + rng_end)
            ind_mtrx = [self._indices[indices].numpy() + inc for inc in
                        range(rng_start, rng_end)]
            idx = np.swapaxes(ind_mtrx, 0, 1)
            return np.unique(idx) if unique else idx

        if merge:
            unique = True
            return expand_indices_range(0, hrz_end)

        ds_indices = dict()
        if self.window > 0:
            ds_indices[WINDOW] = expand_indices_range(0, self.window)
        if self.horizon > 0:
            ds_indices[HORIZON] = expand_indices_range(self.horizon_offset,
                                                       hrz_end)
        return ds_indices

    def overlapping_indices(self, idxs1, idxs2,
                            synch_mode: SynchMode = WINDOW,
                            as_mask=False):
        idxs1, idxs2 = np.asarray(idxs1), np.asarray(idxs2)
        ts1 = self.expand_indices(idxs1)[synch_mode]
        ts2 = self.expand_indices(idxs2)[synch_mode]
        common_ts = np.intersect1d(ts1, ts2)
        is_overlapping = lambda sample: np.any(np.in1d(sample, common_ts))
        m1 = np.apply_along_axis(is_overlapping, 1, ts1)
        m2 = np.apply_along_axis(is_overlapping, 1, ts2)
        if as_mask:
            return m1, m2
        return idxs1[m1], idxs2[m2]

    # Representation ##########################################################

    def snapshot(self, indices=None):
        if indices is None:
            indices = range(len(self))
        return Batch.from_data_list([self.get(idx) for idx in indices])

    def numpy(self):
        return np.as_array(self.data)

    def dataframe(self):
        columns = pd.MultiIndex.from_product([np.arange(self.n_nodes),
                                              np.arange(self.n_channels)],
                                             names=['nodes', 'channels'])
        data = self.numpy().reshape((-1, self.n_nodes * self.n_channels))
        return pd.DataFrame(data=data,
                            index=self.index,
                            columns=columns)

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
