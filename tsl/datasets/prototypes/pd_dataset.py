from copy import deepcopy
from typing import Optional, Mapping, Union, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import Index
from torch import Tensor

from tsl import logger
from . import checks
from .dataset import Dataset
from .mixin import TemporalFeaturesMixin, PandasParsingMixin
from ...ops.dataframe import aggregate
from ...typing import FrameArray, OptFrameArray
from ...utils.python_utils import ensure_list


class PandasDataset(Dataset, PandasParsingMixin, TemporalFeaturesMixin):
    r"""Create a tsl dataset from a :class:`pandas.DataFrame`.

    Args:
        dataframe (pandas.Dataframe): DataFrame containing the data related to
            the main signals. The index is considered as the temporal dimension.
            The columns are identified as:
              - *nodes*: if there is only one level (we assume the number of
                channels to be 1).
              - *(nodes, channels)*: if there are two levels (i.e., if columns
                is a :class:`~pandas.MultiIndex`). We assume nodes are at
                first level, channels at second.
        exogenous (dict, optional): named mapping of DataFrame (or numpy arrays)
            with exogenous variables. An exogenous variable is a
            multi-dimensional covariate to the main channels. They can be either
            *node-wise*, with a column for each *(node, channel)*, or *global*,
            with a column for each *channel*. If the input is bi-dimensional, it
            is assumed to be *(steps, nodes)*, thus it is broadcasted to
            *(steps, nodes, 1 channel)*. To specify a global exogenous variable,
            prefix the name with :obj:`global_`, i.e., bi-dimensional input
            :obj:`global_covariate` will be mapped to :obj:`dataset.covariate`
            global exogenous.
            (default: :obj:`None`)
        attributes: (dict, optional): named mapping of static :obj:`objects`. An
            :obj:`attribute` is a data structure that does not vary with time,
            e.g., metadata.
            (default: :obj:`None`)
        mask (pandas.Dataframe or numpy.ndarray, optional): Boolean mask
            denoting if values in data are valid (1) or not (0).
            (default: :obj:`None`)
        freq (str, optional): Force a frequency (possibly by resampling).
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
        sort_index (bool): whether to keep the dataset chronologically sorted.
            (default: :obj:`True`)
        name (str, optional): Optional name of the dataset.
            (default: :obj:`class_name`)
        precision (int or str, optional): numerical precision for data: 16 (or
            "half"), 32 (or "full") or 64 (or "double").
            (default: :obj:`32`)
    """
    similarity_options = {'correntropy'}
    temporal_aggregation_options = {'sum', 'mean', 'min', 'max', 'nearest'}
    spatial_aggregation_options = {'sum', 'mean', 'min', 'max'}

    def __init__(self, dataframe: pd.DataFrame,
                 exogenous: Optional[Mapping[str, FrameArray]] = None,
                 attributes: Optional[Mapping[str, FrameArray]] = None,
                 mask: OptFrameArray = None,
                 freq: Optional[str] = None,
                 similarity_score: Optional[str] = None,
                 temporal_aggregation: str = 'sum',
                 spatial_aggregation: str = 'sum',
                 default_splitting_method: Optional[str] = 'temporal',
                 sort_index: bool = True,
                 name: str = None,
                 precision: Union[int, str] = 32):
        super().__init__(name=name,
                         similarity_score=similarity_score,
                         temporal_aggregation=temporal_aggregation,
                         spatial_aggregation=spatial_aggregation,
                         default_splitting_method=default_splitting_method)
        # Private data buffers
        self._exogenous = dict()
        self._attributes = dict()

        self.sort_index = sort_index
        self.precision = precision

        # set dataset's dataframe
        self.df = self._parse_dataframe(dataframe)
        if self.sort_index:
            self.df.sort_index(inplace=True)

        if mask is not None:
            mask = self._to_primary_df_schema(mask).values.astype('uint8')
        self.mask = mask

        # Store exogenous and attributes
        if exogenous is not None:
            for name, value in exogenous.items():
                self.add_exogenous(value, name)
        if attributes is not None:
            for name, value in attributes.items():
                self.add_attribute(value, name)

        # Set dataset frequency
        if freq is not None:
            self.freq = checks.to_pandas_freq(freq)
            # resample all dataframes to new frequency
            self.resample_(freq=self.freq, aggr=self.temporal_aggregation)
        else:
            try:
                freq = self.df.index.freq or self.df.index.inferred_freq
            except AttributeError:
                pass
            self.freq = None if freq is None else checks.to_pandas_freq(freq)
            self.index.freq = self.freq

    def __getattr__(self, item):
        if '_exogenous' in self.__dict__ and item in self._exogenous:
            return self._exogenous[item]
        if '_attributes' in self.__dict__ and item in self._attributes:
            return self._attributes[item]
        raise AttributeError("'{}' object has no attribute '{}'"
                             .format(self.__class__.__name__, item))

    def __delattr__(self, item):
        if item in self._exogenous:
            del self._exogenous[item]
        elif item in self._attributes:
            del self._attributes[item]
        else:
            super(PandasDataset, self).__delattr__(item)

    # Dataset properties

    @property
    def length(self):
        return self.df.values.shape[0]

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def index(self):
        return self.df.index

    @property
    def nodes(self):
        return self.df.columns.unique(0).values

    @property
    def channels(self):
        return self.df.columns.unique(1).values

    @property
    def shape(self):
        return self.length, self.n_nodes, self.n_channels

    # Secondary properties

    @property
    def mask(self):
        if self.has_mask:
            return self._mask.reshape(self.shape)
        return (~np.isnan(self.numpy())).astype('uint8')

    @mask.setter
    def mask(self, value):
        self._mask = value

    @mask.deleter
    def mask(self):
        self._mask = None

    @property
    def exogenous(self):
        return {name: df for name, df in self._exogenous.items()
                if df.columns.nlevels == 2}

    @property
    def global_exogenous(self):
        return {name: df for name, df in self._exogenous.items()
                if df.columns.nlevels == 1}

    @property
    def attributes(self):
        return self._attributes

    # flags

    @property
    def has_mask(self) -> bool:
        return self._mask is not None

    @property
    def has_exogenous(self) -> bool:
        return len(self._exogenous) > 0

    @property
    def has_attributes(self) -> bool:
        return len(self._attributes) > 0

    # get columns multiindex

    def columns(self, nodes=None, channels=None):
        nodes = nodes if nodes is not None else self.nodes
        channels = channels if channels is not None else self.channels
        return pd.MultiIndex.from_product([nodes, channels],
                                          names=['nodes', 'channels'])

    # Setter for secondary data

    def add_exogenous(self, obj: FrameArray, name: str,
                      node_level: bool = True):
        if name.startswith('global_'):
            name = name[7:]
            node_level = False
        # name cannot be an attribute of self, but allow override of exogenous
        self._check_name(name, 'exogenous')
        # add exogenous
        if not isinstance(obj, pd.DataFrame):
            obj = np.asarray(obj)
            if node_level:
                obj = self._to_primary_df_schema(obj)
            else:
                obj = self._to_indexed_df(obj)
        df = self._parse_dataframe(obj, node_level)
        df = self._synch_with_primary(df)
        self._exogenous[name] = df
        return self

    def add_attribute(self, obj: FrameArray, name: str):
        assert isinstance(obj, (pd.DataFrame, np.ndarray))
        # name cannot be an attribute of self, but allow override of attribute
        self._check_name(name, 'attribute')
        # add attribute
        self._attributes[name] = obj
        return self

    # Getter for covariates

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

        df = pd.concat(dfs, axis=1, keys=channels.keys(),
                       names=['exogenous', 'nodes', 'channels'])
        df = df.swaplevel(i='exogenous', j='nodes', axis=1)
        # sort only nodes, keep other order as in the input variables
        df = df.loc[:, nodes]

        if as_numpy:
            return df.values.reshape((len(index), len(nodes), -1))
        return df

    # Aggregation methods

    def resample_(self, freq=None, aggr: str = None, keep: bool = 'first',
                  mask_tolerance: float = 0.):
        freq = checks.to_pandas_freq(freq) if freq is not None else self.freq
        aggr = aggr if aggr is not None else self.temporal_aggregation

        # remove duplicated steps from index
        valid_steps = ~self.index.duplicated(keep=keep)

        # aggregate mask by considering valid if average validity is higher than
        # mask_tolerance
        if self.has_mask:
            mask = pd.DataFrame(self._mask, index=self.index)[valid_steps] \
                .resample(freq)
            mask = mask.mean() >= (1. - mask_tolerance)
            self._mask = mask.to_numpy(dtype='uint8')

        self.df = self.df[valid_steps].resample(freq).apply(aggr)

        for name, value in self._exogenous.items():
            df = value[valid_steps].resample(freq).apply(aggr)
            self._exogenous[name] = df

        self.freq = freq

    def resample(self, freq=None, aggr: str = None, keep: bool = 'first',
                 mask_tolerance: float = 0.):
        out = self.copy()
        out.resample_(freq, aggr, keep, mask_tolerance)
        return out

    def aggregate_(self, node_index: Optional[Union[Index, Mapping]] = None,
                   mask_tolerance: float = 0.):

        # get aggregation function among numpy functions
        aggr_fn = getattr(np, self.spatial_aggregation)

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

        # aggregate mask (if node-wise) and threshold aggregated value
        if self.has_mask and self.mask.ndim == 3:
            columns = self.columns(channels=pd.RangeIndex(self.mask.shape[-1]))
            new_mask = pd.DataFrame(self._mask, self.index, columns)
            new_mask = aggregate(new_mask, node_index, np.mean)
            mask = new_mask >= (1. - mask_tolerance)
            self._mask = mask.to_numpy(dtype='uint8')

        # aggregate main dataframe
        self.df = aggregate(self.df, node_index, aggr_fn)

        # aggregate all node-level exogenous
        for name, value in self.exogenous.items():
            df = aggregate(value, node_index, aggr_fn)
            self.add_exogenous(df, name, node_level=True)

    def aggregate(self, node_index: Optional[Union[Index, Mapping]] = None,
                  mask_tolerance: float = 0.):
        out = self.copy()
        out.aggregate_(node_index, mask_tolerance)
        return out

    # Representations

    def dataframe(self) -> pd.DataFrame:
        df = self.df.reindex(index=self.index,
                             columns=self.columns(),
                             copy=True)
        return df

    def numpy(self, return_idx=False) -> Union[ndarray, Tuple[ndarray, Index]]:
        if return_idx:
            return self.numpy(), self.index
        return self.dataframe().values.reshape(self.shape)

    def pytorch(self) -> Tensor:
        data = self.numpy()
        return torch.tensor(data)

    def copy(self) -> 'PandasDataset':
        return deepcopy(self)

    # Old methods, still to be checked

    def cluster(self,
                clustering_algo,
                clustering_kwarks,
                sim_type='correntropy',
                trainlen=None,
                kn=20,
                inplace=True,
                scale=1.):
        sim = self.get_similarity(method=sim_type, k=kn, trainlen=trainlen)
        algo = clustering_algo(**clustering_kwarks, affinity='precomputed')
        idx = algo.fit_predict(sim)
        _, counts = np.unique(idx, return_counts=True)
        logger.info(('{} ' * len(counts)).format(*counts))
        df = self.aggregate(idx) / scale
        if inplace:
            self.df = df
            self._mask = None
            self.name = self.name + '_clustered'
        return df

    def get_detrended_data(self, train_len=None):
        """
        Perform detrending on a time series by subtracting from each value of the dataset
        the average value computed over the training dataset for each hour/weekday


        :param train_len: test length,
        :return:
            - the detrended datasets
            - the trend values that has to be added back after computing the prediction
        """
        df = self.dataframe()
        df[train_len:] = np.nan
        means = df.groupby(
            [df.index.weekday, df.index.hour, df.index.minute]).transform(
            np.nanmean)
        return self.dataframe() - means, means
