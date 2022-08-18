from copy import deepcopy
from typing import Optional, Mapping, Union, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import Index
from torch import Tensor

from tsl import logger
from tsl.ops.framearray import aggregate, framearray_to_numpy, reduce
from tsl.typing import FrameArray, OptFrameArray
from tsl.utils.python_utils import ensure_list
from . import casting
from .dataset import Dataset
from .mixin import TabularParsingMixin


class TabularDataset(Dataset, TabularParsingMixin):
    r"""Base :class:`~tsl.datasets.Dataset` class for tabular data.

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
        name (str, optional): Optional name of the dataset.
            (default: :obj:`class_name`)
        precision (int or str, optional): numerical precision for data: 16 (or
            "half"), 32 (or "full") or 64 (or "double").
            (default: :obj:`32`)
    """

    def __init__(self, target: FrameArray,
                 covariates: Optional[Mapping[str, FrameArray]] = None,
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
        self.target = self._parse_target(target)

        from .pd_dataset import PandasDataset
        if not isinstance(self, PandasDataset) \
                and checks.is_datetime_like_index(self.index):
            logger.warn("It seems you have timestamped data. You may "
                        "consider to use DateTimeDataset instead.")

        self.mask: Optional[np.ndarray] = None
        self.set_mask(mask)

        # Store exogenous and attributes
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

    # Dataset properties

    @property
    def length(self) -> int:
        return self.target.shape[0]

    @property
    def n_nodes(self) -> int:
        if self.is_target_dataframe:
            return len(self.nodes)
        return self.shape[1]

    @property
    def n_channels(self) -> int:
        if self.is_target_dataframe:
            return len(self.channels)
        return self.shape[2]

    @property
    def index(self) -> Union[pd.Index, np.ndarray]:
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
    def shape(self) -> tuple:
        return self.length, self.n_nodes, self.n_channels

    # Covariates properties

    @property
    def covariates(self):
        return self._covariates

    @property
    def exogenous(self):
        return {name: attr['value'] for name, attr in self._covariates.items()
                if 't' in attr['pattern']}

    @property
    def attributes(self):
        return {name: attr['value'] for name, attr in self._covariates.items()
                if 't' not in attr['pattern']}

    @property
    def n_covariates(self):
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
        step, channel) triplet denoting if corresponding value in target is
        observed (obj:`True`) or not (obj:`False`)."""
        if mask is not None:
            mask = self._parse_target(mask).astype('bool')
            mask, _ = self._parse_covariate(mask, 't n f')
            mask = framearray_to_numpy(mask)
        self.mask = mask

    # Setter for covariates

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
        """Shortcut method to add dynamic covariate data."""
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        pattern = 't n f' if node_level else 't f'
        self.add_covariate(name, value, pattern)

    # Getters

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

    def get_frame(self, channels: Union[Sequence, Dict] = None,
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
        self.target /= scale
        return self

    # Preprocessing

    def fill_missing_(self, method):
        # todo
        raise NotImplementedError()

    # Representations

    def dataframe(self) -> pd.DataFrame:
        if self.is_target_dataframe:
            return self.target.reindex(index=self.index,
                                       columns=self._columns_multiindex(),
                                       copy=True)
        data = self.target.reshape(self.length, -1)
        df = pd.DataFrame(data, columns=self._columns_multiindex())
        return df

    def numpy(self, return_idx=False) -> Union[ndarray, Tuple[ndarray, Index]]:
        if return_idx:
            return self.numpy(return_idx=False), self.index
        if self.is_target_dataframe:
            return self.dataframe().values.reshape(self.shape)
        return self.target

    def pytorch(self) -> Tensor:
        data = self.numpy()
        return torch.tensor(data)

    def copy(self) -> 'TabularDataset':
        return deepcopy(self)
