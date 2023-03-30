from typing import Callable, Mapping, Optional, Tuple, Union

import torch

from tsl.typing import DataArray, SparseTensArray, TemporalIndex

from .batch_map import BatchMap, BatchMapItem
from .preprocessing import Scaler
from .spatiotemporal_dataset import SpatioTemporalDataset
from .synch_mode import HORIZON


class ImputationDataset(SpatioTemporalDataset):
    r"""A dataset for imputation tasks. It is a subclass of
    :class:`SpatioTemporalDataset` and most of its attributes. The main
    difference is the addition of a :obj:`eval_mask` attribute which is a
    boolean mask denoting if values to evaluate imputations.

    Args:
        target (DataArray): Data relative to the primary channels.
        eval_mask (DataArray): Boolean mask denoting values that can be used for
            evaluating imputations. The mask is (1) if the corresponding value
            can be used for evaluation and (0) otherwise.
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
        horizon_lag (int): Sampling frequency (in number of steps) in prediction
            horizon.
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

        if covariates is None:
            covariates = dict()
        # add to covariate
        covariates['eval_mask'] = dict(value=eval_mask,
                                       pattern='t n f',
                                       add_to_input_map=True,
                                       synch_mode=HORIZON,
                                       preprocess=False)
        # add to input map
        if input_map is not None:
            input_map['eval_mask'] = BatchMapItem('eval_mask',
                                                  synch_mode=HORIZON,
                                                  pattern='t n f',
                                                  preprocess=False)

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

        # ensure evaluation datapoints are removed from input
        if mask is not None:
            mask = torch.logical_not(self.eval_mask) & mask
        else:
            mask = torch.logical_not(
                self.eval_mask) & ~torch.isnan(self.target)
        self.set_mask(mask)
