from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Mapping, Union, Dict, Tuple, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Index

from tsl import logger
from tsl.ops.framearray import aggregate, framearray_to_numpy, reduce, fill_nan
from tsl.typing import FrameArray, OptFrameArray, FillOptions, Scalar, \
    TemporalIndex
from tsl.utils.python_utils import ensure_list
from . import casting
from .dataset import Dataset
from .mixin import TabularParsingMixin
from ...ops.pattern import outer_pattern, broadcast


class TabularDataset(Dataset, TabularParsingMixin):
    r"""Base :class:`~tsl.datasets.prototypes.Dataset` class for tabular data.

    Tabular data are assumed to be 3-dimensional arrays where the dimensions
    represent time, nodes and features, respectively. They can be either
    :class:`~pandas.DataFrame` or :class:`~numpy.ndarray`.

    Args:
        target (FrameArray): :class:`~pandas.DataFrame` or
            :class:`numpy.ndarray` containing the data related to the target
            signals. The first dimension (or the DataFrame index) is considered
            as the temporal dimension. The second dimension represents nodes,
            the last one denotes the number of channels. If the input array is
            bi-dimensional (or the DataFrame's columns are not
            a :class:`~pandas.MultiIndex`), the sequence is assumed to be
            univariate (number of channels = 1). If DataFrame's columns are a
            :class:`~pandas.MultiIndex` with two levels, we assume nodes are at
            first level, channels at second.

        covariates (dict, optional): named mapping of :class:`~pandas.DataFrame`
            or :class:`numpy.ndarray` representing covariates. Examples of
            covariates are exogenous signals (in the form of dynamic,
            multidimensional data) or static attributes (e.g., graph/node
            metadata). You can specify what each axis refers to by providing a
            :obj:`pattern` for each item in the mapping. Every item can be:

            + a :class:`~pandas.DataFrame` or :class:`~numpy.ndarray`: in this
              case the pattern is inferred from the shape (if possible).
            + a :class:`dict` with keys 'value' and 'pattern' indexing the
              covariate object and the relative pattern, respectively.

            (default: :obj:`None`)
        mask (FrameArray, optional): Boolean mask denoting if values in target
            are valid (:obj:`True`) or not (:obj:`False`).
            (default: :obj:`None`)
        similarity_score (str): Default method to compute the similarity matrix
            with :obj:`compute_similarity`. It must be inside dataset's
            :obj:`similarity_options`.
            (default: :obj:`None`)
        temporal_aggregation (str): Default temporal aggregation method after
            resampling.
            (default: :obj:`sum`)
        spatial_aggregation (str): Default spatial aggregation method for
            :obj:`aggregate`, i.e., how to aggregate multiple nodes together.
            (default: :obj:`sum`)
        default_splitting_method (str, optional): Default splitting method for
            the dataset, i.e., how to split the dataset into train/val/test.
            (default: :obj:`temporal`)
        force_synchronization (bool): Synchronize all time-varying covariates
            with target.
            (default: :obj:`True`)
        name (str, optional): Optional name of the dataset.
            (default: :obj:`class_name`)
        precision (int or str, optional): numerical precision for data: 16 (or
            "half"), 32 (or "full") or 64 (or "double").
            (default: :obj:`32`)
    """

    def __init__(self, target: FrameArray,
                 mask: OptFrameArray = None,
                 covariates: Optional[Mapping[str, FrameArray]] = None,
                 similarity_score: Optional[str] = None,
                 temporal_aggregation: str = 'sum',
                 spatial_aggregation: str = 'sum',
                 default_splitting_method: Optional[str] = 'temporal',
                 force_synchronization: bool = True,
                 name: str = None,
                 precision: Union[int, str] = 32):
        super().__init__(name=name,
                         similarity_score=similarity_score,
                         temporal_aggregation=temporal_aggregation,
                         spatial_aggregation=spatial_aggregation,
                         default_splitting_method=default_splitting_method)
        # Set data precision before parsing objects
        self.precision = precision
        self.force_synchronization = force_synchronization

        # Set dataset's main signal
        self.target = self._parse_target(target)

        from .datetime_dataset import DatetimeDataset
        if not isinstance(self, DatetimeDataset) \
                and casting.is_datetime_like_index(self.index):
            logger.warn("It seems you have timestamped data. You may "
                        "consider to use tsl.datasets.DatetimeDataset instead.")

        self.mask: Optional[np.ndarray] = None
        self.set_mask(mask)

        # Store covariates (e.g., exogenous and attributes)
        self._covariates = dict()
        if covariates is not None:
            for name, value in covariates.items():
                self.add_covariate(name, **self._value_to_kwargs(value))

    def __getattr__(self, item):
        if '_covariates' in self.__dict__ and item in self._covariates:
            return self._covariates[item]['value']
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(self.__class__.__name__, item))

    def __delattr__(self, item):
        if item == 'mask':
            self.set_mask(None)
        elif item in self._covariates:
            del self._covariates[item]
        else:
            super(TabularDataset, self).__delattr__(item)

    # Dataset properties ######################################################

    @property
    def length(self) -> int:
        """Number of time steps in the dataset."""
        return self.target.shape[0]

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the dataset."""
        if self.is_target_dataframe:
            return len(self.nodes)
        return self.target.shape[1]

    @property
    def n_channels(self) -> int:
        """Number of channels in dataset's target."""
        if self.is_target_dataframe:
            return len(self.channels)
        return self.target.shape[2]

    @property
    def shape(self) -> tuple:
        return self.length, self.n_nodes, self.n_channels

    @property
    def index(self) -> Union[pd.Index, TemporalIndex, np.ndarray]:
        if self.is_target_dataframe:
            return self.target.index
        return np.arange(self.length)

    @property
    def nodes(self) -> Union[pd.Index, np.ndarray]:
        if self.is_target_dataframe:
            return self.target.columns.unique(0)
        return np.arange(self.n_nodes)

    @property
    def channels(self) -> Union[pd.Index, np.ndarray]:
        if self.is_target_dataframe:
            return self.target.columns.unique(1)
        return np.arange(self.n_channels)

    @property
    def patterns(self) -> dict:
        """Shows the dimension of the data in the dataset in a more informative
        way.

        The pattern mapping can be useful to glimpse on how data are arranged.
        The convention we use is the following:

          * 't' stands for “number of time steps”
          * 'n' stands for “number of nodes”
          * 'f' stands for “number of features” (per node)
        """
        patterns = {'target': 't n f'}
        if self.mask is not None:
            patterns['mask'] = 't n f'
        patterns.update({name: attr['pattern']
                         for name, attr in self._covariates.items()})
        return patterns

    # Covariates properties

    @property
    def covariates(self) -> dict:
        return {name: attr['value'] for name, attr in self._covariates.items()}

    @property
    def exogenous(self):
        """Time-varying covariates of the dataset's target."""
        return {name: attr['value'] for name, attr in self._covariates.items()
                if 't' in attr['pattern']}

    @property
    def attributes(self):
        """Static features related to the dataset."""
        return {name: attr['value'] for name, attr in self._covariates.items()
                if 't' not in attr['pattern']}

    @property
    def n_covariates(self) -> int:
        """Number of covariates in the dataset."""
        return len(self._covariates)

    # flags

    @property
    def is_target_dataframe(self) -> bool:
        return isinstance(self.target, pd.DataFrame)

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_covariates(self) -> bool:
        return self.n_covariates > 0

    # Setters #################################################################

    def set_target(self, value: FrameArray):
        r"""Set sequence of target channels at :obj:`self.target`."""
        self.target = self._parse_target(value)

    def set_mask(self, mask: OptFrameArray):
        r"""Set mask of target channels, i.e., a bool for each (node, time
        step, feature) triplet denoting if corresponding value in target is
        observed (obj:`True`) or not (obj:`False`)."""
        if mask is not None:
            mask = self._parse_target(mask).astype('bool')
            with self.synchronize(True):
                mask, _ = self._parse_covariate(mask, 't n f')
            mask = framearray_to_numpy(mask)
            # check mask features are broadcastable to target's features
            if mask.shape[-1] not in [1, self.n_channels]:
                raise RuntimeError(f"Mask features ({mask.shape[-1]}) cannot "
                                   "be broadcasted to target's number of "
                                   f"features {self.n_channels}.")
        self.mask = mask

    # Setters for covariates

    def add_covariate(self, name: str, value: FrameArray,
                      pattern: Optional[str] = None):
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
            value (FrameArray): the object to be added.
            pattern (str, optional): the pattern of the object. A pattern
                specifies what each axis refers to:

                - 't': temporal dimension;
                - 'n': node dimension;
                - 'c'/'f': channels/features dimension.

                If :obj:`None`, the pattern is inferred from the shape.
                (default :obj:`None`)
        """
        # name cannot be an attribute of self, but allow override
        invalid_names = set(dir(self))
        if name in invalid_names:
            raise ValueError(f"Cannot add object with name '{name}', "
                             f"{self.__class__.__name__} contains already an "
                             f"attribute named '{name}'.")
        value, pattern = self._parse_covariate(value, pattern)
        self._covariates[name] = dict(value=value, pattern=pattern)

    def add_exogenous(self, name: str, value: FrameArray,
                      node_level: bool = True):
        """Shortcut method to add a time-varying covariate."""
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        pattern = 't n f' if node_level else 't f'
        self.add_covariate(name, value, pattern)

    # Getters #################################################################

    def get_mask(self, dtype: Union[type, str, np.dtype] = None,
                 as_dataframe: bool = False) -> FrameArray:
        mask = self.mask if self.has_mask else ~np.isnan(self.numpy())
        if dtype is not None:
            assert dtype in ['bool', 'uint8', bool, np.bool, np.uint8]
            mask = mask.astype(dtype)
        if as_dataframe:
            assert self.is_target_dataframe
            data = mask.reshape(self.length, -1)
            mask = pd.DataFrame(data, index=self.index,
                                columns=self._columns_multiindex())
        return mask

    def expand_frame(self, key: str, pattern: str,
                     time_index: Union[List, np.ndarray] = None,
                     node_index: Union[List, np.ndarray] = None,
                     channel_index: Union[List, np.ndarray] = None) \
            -> np.ndarray:
        obj = getattr(self, key)
        x = framearray_to_numpy(obj)
        in_pattern = self.patterns[key]
        if channel_index is not None:
            assert in_pattern.count('f') == 1, \
                "Can select channels only in frames with just one " \
                "channel dimension."
            dim = in_pattern.strip().split(' ').index('f')
            if isinstance(obj, pd.DataFrame):
                axis = 'columns' if dim > 0 else 'index'
                level = dim - 1 if dim > 0 else 0
                channels = getattr(obj, axis).unique(level)
                channel_indexer = channels.get_indexer(channel_index)
                if any(channel_indexer < 0):
                    unmatch = channel_index[channel_indexer < 0]
                    raise KeyError(f"Channels {unmatch} not in {key}.")
                channel_index = channel_indexer
            x = x.take(channel_index, dim)
        pattern = in_pattern + ' -> ' + pattern
        x = broadcast(x, pattern, t=self.length, n=self.n_nodes,
                      time_index=time_index, node_index=node_index)
        return x

    def get_frame(self, channels: Union[str, List,
                                        Dict[str, Union[
                                            str, int, List, None]]] = None,
                  node_index: Union[List, np.ndarray] = None,
                  time_index: Union[List, np.ndarray] = None,
                  cat_dim: Optional[int] = -1,
                  return_pattern: bool = True,
                  as_numpy: bool = True):

        # parse channels
        if channels is None:
            # defaults to all data
            channels = list(self.patterns.keys())
        elif isinstance(channels, str):
            channels = [channels]

        # build a channel index for each queried key
        if not isinstance(channels, dict):
            channels = {key: None for key in channels}
        else:
            channels = {key: ensure_list(chn) for key, chn in channels.items()}

        time_index = self._get_time_index(time_index)
        node_index = self._get_node_index(node_index)

        pattern = outer_pattern([self.patterns[key] for key in channels])
        frames = [self.expand_frame(key, pattern, time_index, node_index,
                                    channel_index=channel_index)
                  for key, channel_index in channels.items()]

        if cat_dim is not None:
            frames = np.concatenate(frames, axis=cat_dim)

        if not as_numpy:
            time_index = self._get_time_index(time_index, layout="slice")
            node_index = self._get_node_index(node_index, layout="slice")
            assert self.is_target_dataframe
            idxs, names = [], []
            for dim in pattern.replace(' ', ''):
                if dim == 't':
                    idxs.append(self.index[time_index])
                    names.append('index')
                elif dim == 'n':
                    idxs.append(self.nodes[node_index])
                    names.append('nodes')
                else:  # dim = 'f'
                    channel_index = []
                    for key, chn in channels.items():
                        if chn is None:
                            obj = getattr(self, key)
                            dim = self.patterns[key].split(' ').index('f')
                            if isinstance(obj, pd.DataFrame):
                                axis = 'columns' if dim > 0 else 'index'
                                level = dim - 1 if dim > 0 else 0
                                chn = getattr(obj, axis).unique(level)
                            else:
                                chn = np.arange(obj.shape[dim])
                        channel_index.extend([f"{key}/{c}" for c in chn])
                    idxs.append(channel_index)
                    names.append('nodes')
            index = pd.Index(idxs.pop(0), name=names.pop(0))
            columns = pd.MultiIndex.from_product(idxs, names=names)
            frames = pd.DataFrame(frames.reshape(frames.shape[0], -1),
                                  index=index, columns=columns)

        if return_pattern:
            return frames, pattern
        return frames

    # Private getters #########################################################

    def _get_time_index(self, time_index=None, layout='index'):
        if time_index is None:
            return slice(None) if layout == 'slice' else None
        if isinstance(time_index, slice):
            if layout == 'slice':
                return time_index
            time_index = np.arange(max(time_index.start or 0, 0),
                                   min(time_index.stop or len(self), len(self)),
                                   time_index.step or 1)
        elif isinstance(time_index, pd.Index):
            assert self.is_target_dataframe
            time_indexer = self.index.get_indexer(time_index)
            if any(time_indexer < 0):
                unmatch = time_index[time_indexer < 0]
                raise KeyError(f"Indices {unmatch} not in index.")
            time_index = time_indexer
        time_index = np.asarray(time_index)
        if layout == 'mask':
            mask = np.zeros_like(self.index, dtype=bool)
            mask[time_index] = True
            return mask
        return time_index

    def _get_node_index(self, node_index=None, layout='index'):
        if node_index is None:
            return slice(None) if layout == 'slice' else None
        if isinstance(node_index, slice):
            if layout == 'slice':
                return node_index
            node_index = np.arange(max(node_index.start or 0, 0),
                                   min(node_index.stop or self.n_nodes,
                                       self.n_nodes),
                                   node_index.step or 1)
        elif isinstance(node_index, pd.Index):
            assert self.is_target_dataframe
            node_indexer = self.nodes.get_indexer(node_index)
            if any(node_indexer < 0):
                unmatch = node_index[node_indexer < 0]
                raise KeyError(f"Indices {unmatch} not in nodes.")
            node_index = node_indexer
        node_index = np.asarray(node_index)
        if layout == 'mask':
            mask = np.zeros_like(self.nodes, dtype=bool)
            mask[node_index] = True
            return mask
        return node_index

    # Aggregation methods #####################################################

    def aggregate_(self, node_index: Optional[Union[Index, Mapping]] = None,
                   aggr: str = None, mask_tolerance: float = 0.):

        # get aggregation function among numpy functions
        aggr = aggr if aggr is not None else self.spatial_aggregation
        aggr_fn = getattr(np, aggr)

        # node_index parsing: eventually must be an n_nodes-sized array where
        # value at position i is the cluster id of i-th node
        if node_index is None:
            # if not provided, aggregate all nodes together, with cluster id 0
            node_index = np.zeros(self.n_nodes)
        # otherwise, node_index can be a mapping {cluster_id: [nodes]}
        # the set of all nodes in mapping values must be equal to dataset nodes
        elif isinstance(node_index, Mapping):
            ids, groups = [], []
            for group_id, group in node_index.items():
                ids += [group_id] * len(group)
                groups += list(group)
            assert set(groups) == set(self.nodes)
            # reorder node_index according to nodes order in dataset
            ids, groups = np.array(ids), np.array(groups)
            _, order = np.where(self.nodes[:, None] == groups)
            node_index = ids[order]
        else:
            node_index = np.asarray(node_index)

        assert len(node_index) == self.n_nodes

        # aggregate main dataframe
        self.target = aggregate(self.target, node_index, aggr_fn)

        # aggregate mask (if node-wise) and threshold aggregated value
        if self.has_mask:
            mask = aggregate(self.mask, node_index, np.mean)
            mask = mask >= (1. - mask_tolerance)
            self.set_mask(mask)

        # aggregate all node-level exogenous
        for name, attr in self._covariates.items():
            value, pattern = attr['value'], attr['pattern']
            dims = pattern.strip().split(' ')
            if dims[0] == 'n':
                value = aggregate(value, node_index, aggr_fn, axis=0)
            for lvl, dim in enumerate(dims[1:]):
                if dim == 'n':
                    value = aggregate(value, node_index, aggr_fn,
                                      axis=1, level=lvl)
            self._covariates[name]['value'] = value

    def aggregate(self, node_index: Optional[Union[Index, Mapping]] = None,
                  aggr: str = None, mask_tolerance: float = 0.):
        ds = deepcopy(self)
        ds.aggregate_(node_index, aggr, mask_tolerance)
        return ds

    def reduce_(self, time_index=None, node_index=None):
        time_index = self._get_time_index(time_index, layout='mask')
        node_index = self._get_node_index(node_index, layout='mask')
        try:
            self.target = reduce(self.target, time_index, axis=0)
            self.target = reduce(self.target, node_index, axis=1, level=0)
            if self.has_mask:
                self.mask = reduce(self.mask, time_index, axis=0)
                self.mask = reduce(self.mask, node_index, axis=1, level=0)

            for name, attr in self._covariates.items():
                value, pattern = attr['value'], attr['pattern']
                dims = pattern.strip().split(' ')
                if dims[0] == 't':
                    value = reduce(value, time_index, axis=0)
                elif dims[0] == 'n':
                    value = reduce(value, node_index, axis=0)
                for lvl, dim in enumerate(dims[1:]):
                    if dim == 't':
                        value = reduce(value, time_index, axis=1, level=lvl)
                    elif dim == 'n':
                        value = reduce(value, node_index, axis=1, level=lvl)
                self._covariates[name]['value'] = value
        except Exception as e:
            raise e
        return self

    def reduce(self, time_index=None, node_index=None):
        return deepcopy(self).reduce_(time_index, node_index)

    @contextmanager
    def synchronize(self, flag=True):
        try:
            is_synch = self.force_synchronization
            self.force_synchronization = flag
            yield self
        finally:
            self.force_synchronization = is_synch

    # Preprocessing

    def fill_nan_(self, value: Optional[Union[Scalar, FrameArray]] = None,
                  method: FillOptions = None, axis: int = 0):
        self.target = fill_nan(self.target, value, method, axis)

    # Representations

    def dataframe(self) -> pd.DataFrame:
        if self.is_target_dataframe:
            return self.target.copy()
        data = self.target.reshape(self.length, -1)
        df = pd.DataFrame(data, columns=self._columns_multiindex())
        return df

    def numpy(self, return_idx=False) -> Union[ndarray, Tuple[ndarray, Index]]:
        if return_idx:
            return self.numpy(return_idx=False), self.index
        if self.is_target_dataframe:
            return self.dataframe().values.reshape(self.shape)
        return self.target

    def copy(self) -> 'TabularDataset':
        return deepcopy(self)
