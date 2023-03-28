from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from numpy import ndarray
from pandas import DataFrame, DatetimeIndex, PeriodIndex, TimedeltaIndex
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from torch import Tensor
from torch_sparse import SparseTensor

Scalar = Union[int, float]

TensArray = Union[Tensor, ndarray]
OptTensArray = Optional[TensArray]

ScipySparseMatrix = Union[coo_matrix, csr_matrix, csc_matrix]
SparseTensArray = Union[Tensor, SparseTensor, ndarray, ScipySparseMatrix]
OptSparseTensArray = Optional[SparseTensArray]

TorchConnectivity = Union[Tensor, Tuple[Tensor, Optional[Tensor]], SparseTensor]

FrameArray = Union[DataFrame, ndarray]
OptFrameArray = Optional[FrameArray]

DataArray = Union[DataFrame, ndarray, Tensor]
OptDataArray = Optional[DataArray]

TemporalIndex = Union[DatetimeIndex, PeriodIndex, TimedeltaIndex]

Index = Union[List, Tuple, TensArray]
IndexSlice = Union[slice, Index]

FillOptions = Optional[Literal["backfill", "bfill", "ffill", "pad", "mean", "linear"]]

ModelReturnOptions = Type[Union[Tensor, Dict, List, Tuple]]
