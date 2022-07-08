from copy import deepcopy
from typing import Optional, Mapping, Union, Sequence, Dict, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import Index
from torch import Tensor

from tsl import logger
from . import checks
from .dataset import Dataset
from .mixin import TabularParsingMixin
from ...ops.dataframe import aggregate, framearray_to_numpy, reduce
from ...typing import FrameArray, OptFrameArray
from ...utils.python_utils import ensure_list


class TabularDataset(Dataset, TabularParsingMixin):
    r"""Base :class:`~tsl.datasets.Dataset` class for tabular data.

    Tabular data are assumed to be 3-dimensional arrays where the dimensions
    represent time, nodes and features, respectively. They can be either
    :class:`~pandas.DataFrame` or :class:`~numpy.ndarray`.

    Args:
        primary (pandas.Dataframe): DataFrame containing the data related to
            the main signals. The index is considered as the temporal dimension.
            The columns are identified as:

            + *nodes*: if there is only one level (we assume the number of
              channels to be 1).

            + *(nodes, channels)*: if there are two levels (i.e., if columns is
              a :class:`~pandas.MultiIndex`). We assume nodes are at first
              level, channels at second.

        secondary (dict, optional): named mapping of DataFrame (or numpy arrays)
            with secondary data. Examples of secondary data are exogenous
            variables (in the form of multidimensional covariates) or static
            attributes (e.g., metadata). You can specify what each axis refers
            to by providing a :obj:`pattern` for each item in the mapping.
            Every item can be:

            + a :class:`~pandas.DataFrame` or :class:`~numpy.ndarray`: in this
              case the pattern is inferred from the shape (if possible).

            TODO
            (default: :obj:`None`)
        mask (pandas.Dataframe or numpy.ndarray, optional): Boolean mask
            denoting if values in data are valid (:obj:`True`) or not
            (:obj:`False`).
            (default: :obj:`None`)
        freq (str, optional): Force a sampling rate, eventually by resampling.
            (default: :obj:`None`)
        similarity_score (str): Default method to compute the similarity matrix
            with :obj:`compute_similarity`. It must be inside dataset's
            :obj:`similarity_options`.
            (default: :obj:`None`)
        temporal_aggregation (str): Default temporal aggregation method after
            resampling. This method is used during instantiation to resample the
            dataset. It must be inside dataset's
            :obj:`temporal_aggregation_options`.
            (default: :obj:`sum`)
        spatial_aggregation (str): Default spatial aggregation method for
            :obj:`aggregate`, i.e., how to aggregate multiple nodes together.
            It must be inside dataset's :obj:`spatial_aggregation_options`.
            (default: :obj:`sum`)
        default_splitting_method (str, optional): Default splitting method for
            the dataset, i.e., how to split the dataset into train/val/test.
            (default: :obj:`temporal`)
        sort_index (bool): whether to sort the dataset chronologically at
            initialization.
            (default: :obj:`True`)
        name (str, optional): Optional name of the dataset.
            (default: :obj:`class_name`)
        precision (int or str, optional): numerical precision for data: 16 (or
            "half"), 32 (or "full") or 64 (or "double").
            (default: :obj:`32`)
    """

    def __init__(self, primary: FrameArray,
                 secondary: Optional[Mapping[str, FrameArray]] = None,
                 mask: OptFrameArray = None,
                 similarity_score: Optional[str] = None,
                 temporal_aggregation: str = 'sum',
                 spatial_aggregation: str = 'sum',
                 default_splitting_method: Optional[str] = 'temporal',
                 name: str = None,
                 precision: Union[int, str] = 32):
        super().__init__(name=name,
                         similarity_score=similarity_score,
                         temporal_aggregation=temporal_aggregation,
                         spatial_aggregation=spatial_aggregation,
                         default_splitting_method=default_splitting_method)
        # Set data precision before parsing objects
        self.precision = precision

        # Set dataset's main signal
        self.primary = self._parse_primary(primary)

        from .pd_dataset import PandasDataset
        if not isinstance(self, PandasDataset) \
                and checks.is_datetime_like_index(self.index):
            logger.warn("It seems you have timestamped data. You may "
                        "consider to use DateTimeDataset instead.")

        self.mask: Optional[np.ndarray] = None
        self.set_mask(mask)

        # Store exogenous and attributes
        self._secondary = dict()
        if secondary is not None:
            for name, value in secondary.items():
                self.add_secondary(name, **self._value_to_kwargs(value))

    def __getattr__(self, item):
        if '_secondary' in self.__dict__ and item in self._secondary:
            return self._secondary[item]['value']
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(self.__class__.__name__, item))

    def __delattr__(self, item):
        if item == 'mask':
            self.set_mask(None)
        elif item in self._secondary:
            del self._secondary[item]
        else:
            super(TabularDataset, self).__delattr__(item)

    # Dataset properties

    @property
    def is_primary_dataframe(self) -> bool:
        return isinstance(self.primary, pd.DataFrame)

    @property
    def length(self) -> int:
        return self.primary.shape[0]

    @property
    def n_nodes(self) -> int:
        if self.is_primary_dataframe:
            return len(self.nodes)
        return self.shape[1]

    @property
    def n_channels(self) -> int:
        if self.is_primary_dataframe:
            return len(self.channels)
        return self.shape[2]

    @property
    def index(self) -> Union[pd.Index, np.ndarray]:
        if self.is_primary_dataframe:
            return self.primary.index
        return np.arange(self.length)

    @property
    def nodes(self) -> Union[pd.Index, np.ndarray]:
        if self.is_primary_dataframe:
            return self.primary.columns.unique(0)
        return np.arange(self.n_nodes)

    @property
    def channels(self) -> Union[pd.Index, np.ndarray]:
        if self.is_primary_dataframe:
            return self.primary.columns.unique(1)
        return np.arange(self.n_channels)

    @property
    def shape(self) -> tuple:
        return self.length, self.n_nodes, self.n_channels

    # Secondary properties

    @property
    def exogenous(self):
        return {name: attr['value'] for name, attr in self._secondary.items()
                if 't' in attr['pattern']}

    @property
    def attributes(self):
        return {name: attr['value'] for name, attr in self._secondary.items()
                if 't' not in attr['pattern']}

    # flags

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_exogenous(self) -> bool:
        return len(self.exogenous) > 0

    @property
    def has_attributes(self) -> bool:
        return len(self.attributes) > 0

    # Setters #################################################################

    def set_primary(self, value: FrameArray):
        r"""Set sequence of primary channels at :obj:`self.primary`."""
        self.primary = self._parse_primary(value)

    def set_mask(self, mask: OptFrameArray):
        r"""Set mask of primary channels, i.e., a bool for each (node, time
        step, channel) triplet denoting if corresponding value in primary
        DataFrame is observed (1) or not (0)."""
        if mask is not None:
            mask = self._parse_primary(mask).astype('bool')
            mask, _ = self._parse_secondary(mask, 't n f')
            mask = framearray_to_numpy(mask)
        self.mask = mask

    # Setter for secondary data

    def add_secondary(self, name: str, value: FrameArray,
                      pattern: Optional[str] = None):
        r"""Add secondary data to the dataset. Examples of secondary data are
        exogenous variables (in the form of multidimensional covariates) or
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
        value, pattern = self._parse_secondary(value, pattern)
        self._secondary[name] = dict(value=value, pattern=pattern)

    def add_exogenous(self, name: str, value: FrameArray,
                      node_level: bool = True):
        """Shortcut method to add dynamic secondary data."""
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        pattern = 't n f' if node_level else 't f'
        self.add_secondary(name, value, pattern)

    # Getters

    def get_mask(self, dtype: Union[type, str, np.dtype] = None,
                 as_dataframe: bool = False) -> FrameArray:
        mask = self.mask if self.has_mask else ~np.isnan(self.numpy())
        if dtype is not None:
            assert dtype in ['bool', 'uint8', bool, np.bool, np.uint8]
            mask = mask.astype(dtype)
        if as_dataframe:
            assert self.is_primary_dataframe
            data = mask.reshape(self.length, -1)
            mask = pd.DataFrame(data, index=self.index,
                                columns=self._columns_multiindex())
        return mask

    def get_exogenous(self, channels: Union[Sequence, Dict] = None,
                      nodes: Sequence = None,
                      index: Sequence = None,
                      as_numpy: bool = True):
        if index is None:
            index = self.index

        if nodes is None:
            nodes = self.nodes

        # parse channels
        if channels is None:
            # defaults to all channels
            channels = self.exogenous.keys()
        elif isinstance(channels, str):
            assert channels in self.exogenous, \
                f"{channels} is not an exogenous group."
            channels = [channels]
        # expand exogenous
        if not isinstance(channels, dict):
            channels = {label: self.exogenous[label].columns.unique('channels')
                        for label in channels}
        else:
            # analyze channels dict
            for exo, chnls in channels.items():
                exo_channels = self.exogenous[exo].columns.unique('channels')
                # if value is None, default to all channels
                if chnls is None:
                    channels[exo] = exo_channels
                else:
                    chnls = ensure_list(chnls)
                    # check that all passed channels are in exo
                    wrong_channels = set(chnls).difference(exo_channels)
                    if len(wrong_channels):
                        raise KeyError(wrong_channels)

        dfs = [self.exogenous[exo].loc[index, (nodes, chnls)]
               for exo, chnls in channels.items()]

        # todo remove 'exogenous' level
        df = pd.concat(dfs, axis=1, keys=channels.keys(),
                       names=['exogenous', 'nodes', 'channels'])
        df = df.swaplevel(i='exogenous', j='nodes', axis=1)
        # sort only nodes, keep other order as in the input variables
        df = df.loc[:, nodes]

        if as_numpy:
            return df.values.reshape((len(index), len(nodes), -1))
        return df

    # Aggregation methods

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

        assert len(node_index) == self.n_nodes

        # aggregate main dataframe
        self.primary = aggregate(self.primary, node_index, aggr_fn)

        # aggregate mask (if node-wise) and threshold aggregated value
        if self.has_mask:
            mask = aggregate(self.mask, node_index, np.mean)
            mask = mask >= (1. - mask_tolerance)
            self.set_mask(mask)

        # aggregate all node-level exogenous
        for name, attr in self._secondary.items():
            value, pattern = attr['value'], attr['pattern']
            dims = pattern.strip().split(' ')
            if dims[0] == 'n':
                value = aggregate(value, node_index, aggr_fn, axis=0)
            for lvl, dim in enumerate(dims[1:]):
                if dim == 'n':
                    value = aggregate(value, node_index, aggr_fn,
                                      axis=1, level=lvl)
            self._secondary[name]['value'] = value

    def aggregate(self, node_index: Optional[Union[Index, Mapping]] = None,
                  aggr: str = None, mask_tolerance: float = 0.):
        ds = deepcopy(self)
        ds.aggregate_(node_index, aggr, mask_tolerance)
        return ds

    def reduce_(self, time_index=None, node_index=None):

        def index_to_mask(index, support):
            if index is None:
                return None
            index: np.ndarray = np.asarray(index)
            if index.dtype != np.bool:
                index = np.in1d(support, index)
            assert index.any()
            return index

        time_index = index_to_mask(time_index, self.index)
        node_index = index_to_mask(node_index, self.nodes)
        try:
            self.primary = reduce(self.primary, time_index, axis=0)
            self.primary = reduce(self.primary, node_index, axis=1, level=0)
            if self.has_mask:
                self.mask = reduce(self.mask, time_index, axis=0)
                self.mask = reduce(self.mask, node_index, axis=1, level=0)

            for name, attr in self._secondary.items():
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
                self._secondary[name]['value'] = value
        except Exception as e:
            raise e
        return self

    def reduce(self, time_index=None, node_index=None):
        return deepcopy(self).reduce_(time_index, node_index)

    def cluster_(self,
                 clustering_algo,
                 clustering_kwarks,
                 sim_type='correntropy',
                 trainlen=None,
                 kn=20,
                 scale=1.):
        sim = self.get_similarity(method=sim_type, k=kn, trainlen=trainlen)
        algo = clustering_algo(**clustering_kwarks, affinity='precomputed')
        idx = algo.fit_predict(sim)
        _, counts = np.unique(idx, return_counts=True)
        logger.info(('{} ' * len(counts)).format(*counts))
        self.aggregate_(idx)
        self.primary /= scale
        return self

    # Preprocessing

    def fill_missing_(self, method):
        # todo
        raise NotImplementedError()

    # Representations

    def dataframe(self) -> pd.DataFrame:
        if self.is_primary_dataframe:
            return self.primary.reindex(index=self.index,
                                        columns=self._columns_multiindex(),
                                        copy=True)
        data = self.primary.reshape(self.length, -1)
        df = pd.DataFrame(data, columns=self._columns_multiindex())
        return df

    def numpy(self, return_idx=False) -> Union[ndarray, Tuple[ndarray, Index]]:
        if return_idx:
            return self.numpy(return_idx=False), self.index
        if self.is_primary_dataframe:
            return self.dataframe().values.reshape(self.shape)
        return self.primary

    def pytorch(self) -> Tensor:
        data = self.numpy()
        return torch.tensor(data)

    def copy(self) -> 'TabularDataset':
        return deepcopy(self)
