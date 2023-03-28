from typing import Callable, Mapping, Optional, Tuple, Union

import torch

from tsl.typing import DataArray, SparseTensArray, TemporalIndex

from .batch_map import BatchMap, BatchMapItem
from .preprocessing import Scaler
from .spatiotemporal_dataset import SpatioTemporalDataset
from .synch_mode import WINDOW


class ImputationDataset(SpatioTemporalDataset):
    """Extension of :class:`~tsl.data.SpatioTemporalDataset` for imputation."""

    def __init__(self,
                 target: DataArray,
                 eval_mask: DataArray,
                 index: Optional[TemporalIndex] = None,
                 input_mask: Optional[DataArray] = None,
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
                 horizon_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):

        if input_mask is not None:
            if covariates is None:
                covariates = dict()
            # add to covariate
            covariates['input_mask'] = dict(value=input_mask,
                                            pattern='t n f',
                                            add_to_input_map=True,
                                            synch_mode=WINDOW,
                                            preprocess=False)
            # add to input map
            if input_map is not None:
                input_map['input_mask'] = BatchMapItem('input_mask',
                                                       synch_mode=WINDOW,
                                                       pattern='t n f',
                                                       preprocess=False)

        horizon = window
        delay = -window

        super(ImputationDataset, self).__init__(target,
                                                index=index,
                                                mask=eval_mask,
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
        if 'input_mask' in self:
            input_mask = self.input_mask & torch.logical_not(self.mask)
            self.update_covariate('input_mask', value=input_mask)
        else:
            input_mask = ~torch.isnan(self.target) & \
                         torch.logical_not(self.mask)
            self.add_covariate('input_mask',
                               value=input_mask,
                               pattern='t n f',
                               add_to_input_map=True,
                               synch_mode=WINDOW,
                               preprocess=False)

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--window', type=int, default=12)
        parser.add_argument('--stride', type=int, default=1)
        parser.add_argument('--window-lag', type=int, default=1)
        parser.add_argument('--horizon-lag', type=int, default=1)
        return parser
