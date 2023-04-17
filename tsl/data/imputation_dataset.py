from typing import Callable, Mapping, Optional, Tuple, Union

import torch

from tsl.typing import DataArray, SparseTensArray, TemporalIndex

from .batch_map import BatchMap, BatchMapItem
from .preprocessing import Scaler
from .spatiotemporal_dataset import SpatioTemporalDataset
from .synch_mode import HORIZON, WINDOW


class ImputationDataset(SpatioTemporalDataset):
    r"""A dataset for imputation tasks. It is a subclass of
    :class:`~tsl.data.SpatioTemporalDataset` and most of its attributes. The
    main difference is the addition of a :obj:`eval_mask` attribute which is a
    boolean mask denoting if values to evaluate imputations.

    Args:
        target (DataArray): Data relative to the primary channels.
        eval_mask (DataArray): Boolean mask denoting values that can be used for
            evaluating imputations. The mask is :obj:`True` if the corresponding
            value must be used for evaluation and :obj:`False` otherwise.
        index (TemporalIndex, optional): Temporal indices for the data.
            (default: :obj:`None`)
        mask (DataArray, optional): Boolean mask denoting if signal in data is
            valid (:obj:`True`) or not (:obj:`False`).
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
        covariates (dict, optional): Dictionary of exogenous channels with
            label. An :obj:`exogenous` element is a temporal array with node- or
            graph-level channels which are covariates to the main signal. The
            temporal dimension must be equal to the temporal dimension of data,
            as well as the number of nodes if the exogenous is node-level.
            (default: :obj:`None`)
        input_map (BatchMap or dict, optional): Defines how data (i.e., the
            target and the covariates) are mapped to dataset sample input. Keys
            in the mapping are keys in both :obj:`item` and :obj:`item.input`,
            while values are :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        target_map (BatchMap or dict, optional): Defines how data (i.e., the
            target and the covariates) are mapped to dataset sample target. Keys
            in the mapping are keys in both :obj:`item` and :obj:`item.target`,
            while values are :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        auxiliary_map (BatchMap or dict, optional): Defines how data (i.e., the
            target and the covariates) are added as additional attributes to the
            dataset sample. Keys in the mapping are keys only in :obj:`item`,
            while values are :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        scalers (Mapping or None): Dictionary of scalers that must be used for
            data preprocessing.
            (default: :obj:`None`)
        trend (DataArray, optional): Trend paired with main signal. Must be of
            the same shape of `data`.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`tsl.data.Data` object and returns a transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        window (int): Length (in number of steps) of the lookback window.
            (default: 12)
        stride (int): Offset (in number of steps) between a sample and the next
            one.
            (default: 1)
        window_lag (int): Sampling frequency (in number of steps) in lookback
            window.
            (default: 1)
        precision (int or str, optional): The float precision to store the data.
            Can be expressed as number (16, 32, or 64) or string ("half",
            "full", "double").
            (default: 32)
        name (str, optional): The (optional) name of the dataset.
    """

    def __init__(self,
                 target: DataArray,
                 eval_mask: DataArray,
                 index: Optional[TemporalIndex] = None,
                 mask: Optional[DataArray] = None,
                 connectivity: Optional[Union[SparseTensArray,
                                              Tuple[DataArray]]] = None,
                 covariates: Optional[Mapping[str, DataArray]] = None,
                 input_map: Optional[Union[Mapping, BatchMap]] = None,
                 target_map: Optional[Union[Mapping, BatchMap]] = None,
                 auxiliary_map: Optional[Union[Mapping, BatchMap]] = None,
                 scalers: Optional[Mapping[str, Scaler]] = None,
                 trend: Optional[DataArray] = None,
                 transform: Optional[Callable] = None,
                 window: int = 12,
                 stride: int = 1,
                 window_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):
        horizon = window
        delay = -window
        horizon_lag = window_lag

        super(ImputationDataset, self).__init__(target,
                                                index=index,
                                                mask=None,
                                                connectivity=connectivity,
                                                covariates=covariates,
                                                input_map=input_map,
                                                target_map=target_map,
                                                auxiliary_map=auxiliary_map,
                                                trend=trend,
                                                transform=transform,
                                                scalers=scalers,
                                                window=window,
                                                horizon=horizon,
                                                delay=delay,
                                                stride=stride,
                                                window_lag=window_lag,
                                                horizon_lag=horizon_lag,
                                                precision=precision,
                                                name=name)
        # add eval_mask as covariate
        self.add_covariate(
            name='eval_mask',
            value=eval_mask,
            pattern='t n f',
            add_to_input_map=False,  # NB
            synch_mode=HORIZON,
            preprocess=False)
        # add eval_mask to auxiliary map
        self.auxiliary_map['eval_mask'] = BatchMapItem('eval_mask',
                                                       synch_mode=HORIZON,
                                                       pattern='t n f',
                                                       preprocess=False)

        # ensure evaluation datapoints are removed from input
        if mask is None:
            mask = ~torch.isnan(self.target)
        mask = torch.logical_not(self.eval_mask) & mask

        # set mask and add to input map
        self.set_mask(mask, add_to_input_map=True)

    def reset_auxiliary_map(self):
        self._clear_batch_map('auxiliary')
        self.auxiliary_map['eval_mask'] = BatchMapItem('eval_mask',
                                                       synch_mode=HORIZON,
                                                       pattern='t n f',
                                                       preprocess=False)

    def reset_input_map(self):
        super().reset_input_map()
        if self.mask is not None:
            self.input_map['mask'] = BatchMapItem('mask',
                                                  synch_mode=WINDOW,
                                                  pattern='t n f',
                                                  preprocess=False)

    def set_mask(self,
                 mask: Optional[DataArray],
                 add_to_input_map: bool = True):
        super().set_mask(mask, add_to_auxiliary_map=False)
        if mask is not None and add_to_input_map:
            self.input_map['mask'] = BatchMapItem('mask',
                                                  synch_mode=WINDOW,
                                                  pattern='t n f',
                                                  preprocess=False,
                                                  shape=self.mask.shape)
