from copy import deepcopy
from typing import Optional, Mapping, Union, Literal

from tsl.typing import FrameArray, OptFrameArray
from .casting import to_pandas_freq
from .mixin import TemporalFeaturesMixin
from .tabular_dataset import TabularDataset


class DatetimeDataset(TabularDataset, TemporalFeaturesMixin):
    r"""Create a tsl dataset from a :class:`pandas.DataFrame`.

    Args:
        target (pandas.Dataframe): DataFrame containing the data related to
            the main signals. The index is considered as the temporal dimension.
            The columns are identified as:

            + *nodes*: if there is only one level (we assume the number of
              channels to be 1).

            + *(nodes, channels)*: if there are two levels (i.e., if columns is
              a :class:`~pandas.MultiIndex`). We assume nodes are at first
              level, channels at second.

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
        force_synchronization (bool): Synchronize all time-varying covariates
            with target.
            (default: :obj:`True`)
        name (str, optional): Optional name of the dataset.
            (default: :obj:`class_name`)
        precision (int or str, optional): numerical precision for data: 16 (or
            "half"), 32 (or "full") or 64 (or "double").
            (default: :obj:`32`)
    """
    similarity_options = {'correntropy'}

    def __init__(self, target: FrameArray,
                 mask: OptFrameArray = None,
                 covariates: Optional[Mapping[str, FrameArray]] = None,
                 freq: Optional[str] = None,
                 similarity_score: Optional[str] = None,
                 temporal_aggregation: str = 'sum',
                 spatial_aggregation: str = 'sum',
                 default_splitting_method: Optional[str] = 'temporal',
                 sort_index: bool = True,
                 force_synchronization: bool = True,
                 name: str = None,
                 precision: Union[int, str] = 32):
        super().__init__(target=target,
                         mask=mask,
                         covariates=covariates,
                         similarity_score=similarity_score,
                         temporal_aggregation=temporal_aggregation,
                         spatial_aggregation=spatial_aggregation,
                         default_splitting_method=default_splitting_method,
                         force_synchronization=force_synchronization,
                         name=name,
                         precision=precision)

        if sort_index:
            self.sort()

        # Set dataset frequency
        if freq is not None:
            self.freq = to_pandas_freq(freq)
            # resample all dataframes to new frequency
            self.resample_(freq=self.freq, aggr=self.temporal_aggregation)
        else:
            try:
                freq = self.target.index.freq or self.target.index.inferred_freq
            except AttributeError:
                pass
            self.freq = None if freq is None else to_pandas_freq(freq)
            self.index.freq = self.freq

    # Aggregation methods

    def sort(self):
        """"""
        self.target.sort_index(inplace=True)
        if self.force_synchronization:
            for name, attr in self._covariates.items():
                if 't' in attr['pattern']:
                    attr['value'] = attr['value'].reindex(self.index)

    def resample_(self, freq=None, aggr: str = None,
                  keep: Literal["first", "last", False] = 'first',
                  mask_tolerance: float = 0.):
        """"""
        freq = to_pandas_freq(freq) if freq is not None else self.freq
        aggr = aggr if aggr is not None else self.temporal_aggregation

        # remove duplicated steps from index
        valid_steps = ~self.index.duplicated(keep=keep)

        # get mask as DataFrame before resampling
        mask = self.get_mask(as_dataframe=True) if self.has_mask else None

        _target = self.target[valid_steps].resample(freq).apply(aggr)
        self.set_target(_target)

        # aggregate mask by considering valid if average validity is higher than
        # mask_tolerance
        if mask is not None:
            mask = mask[valid_steps].resample(freq)
            mask = mask.mean() >= (1. - mask_tolerance)
            self.set_mask(mask)

        for name, attr in self._covariates.items():
            value, pattern = attr['value'], attr['pattern']
            dims = pattern.strip().split(' ')
            if dims[0] == 't':
                value = value[valid_steps].resample(freq).apply(aggr)
            for lvl, dim in enumerate(dims[1:]):
                if dim == 't':
                    value = value[valid_steps] \
                        .resample(freq, axis=1, level=lvl).apply(aggr)
            self._covariates[name]['value'] = value

        self.freq = freq

    def resample(self, freq=None, aggr: str = None,
                 keep: Literal["first", "last", False] = 'first',
                 mask_tolerance: float = 0.):
        """"""
        return deepcopy(self).resample_(freq, aggr, keep, mask_tolerance)

    # Preprocessing

    def detrend(self, method):
        raise NotImplementedError()
