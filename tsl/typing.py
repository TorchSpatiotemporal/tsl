from typing import Union, Tuple, List, Optional

from numpy import ndarray
from pandas import DatetimeIndex, PeriodIndex, TimedeltaIndex, DataFrame
from torch import Tensor

TensArray = Union[Tensor, ndarray]
OptTensArray = Optional[TensArray]

FrameArray = Union[DataFrame, ndarray]
OptFrameArray = Optional[FrameArray]

TemporalIndex = Union[DatetimeIndex, PeriodIndex, TimedeltaIndex]

Index = Union[List, Tuple, TensArray]
IndexSlice = Union[slice, Index]
