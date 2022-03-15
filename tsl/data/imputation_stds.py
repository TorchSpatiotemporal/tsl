from typing import Union, Optional, Mapping, Tuple

import numpy as np

from tsl.data import SpatioTemporalDataset, BatchMap, BatchMapItem
from tsl.data.preprocessing import Scaler
from tsl.typing import (TensArray, TemporalIndex)


class ImputationDataset(SpatioTemporalDataset):

    def __init__(self, data: TensArray,
                 index: Optional[TemporalIndex] = None,
                 training_mask: Optional[TensArray] = None,
                 eval_mask: Optional[TensArray] = None,
                 connectivity: Optional[
                     Union[TensArray, Tuple[TensArray]]] = None,
                 exogenous: Optional[Mapping[str, TensArray]] = None,
                 attributes: Optional[Mapping[str, TensArray]] = None,
                 input_map: Optional[Union[Mapping, BatchMap]] = None,
                 trend: Optional[TensArray] = None,
                 scalers: Optional[Mapping[str, Scaler]] = None,
                 window: int = 24,
                 stride: int = 1,
                 window_lag: int = 1,
                 horizon_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):
        if training_mask is None:
            training_mask = np.isnan(data)
        if exogenous is None:
            exogenous = dict()
        if eval_mask is not None:
            exogenous['eval_mask'] = eval_mask
        if input_map is not None:
            input_map['eval_mask'] = BatchMapItem('eval_mask', preprocess=False)
        super(ImputationDataset, self).__init__(data,
                                                index=index,
                                                mask=training_mask,
                                                connectivity=connectivity,
                                                exogenous=exogenous,
                                                attributes=attributes,
                                                input_map=input_map,
                                                trend=trend,
                                                scalers=scalers,
                                                window=window,
                                                horizon=window,
                                                delay=-window,
                                                stride=stride,
                                                window_lag=window_lag,
                                                horizon_lag=horizon_lag,
                                                precision=precision,
                                                name=name)

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--window', type=int, default=24)
        parser.add_argument('--stride', type=int, default=1)
        parser.add_argument('--window-lag', type=int, default=1)
        parser.add_argument('--horizon-lag', type=int, default=1)
        return parser
