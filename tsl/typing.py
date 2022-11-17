from typing import Union, Tuple, List, Optional, Literal, Type

from numpy import ndarray
from pandas import DatetimeIndex, PeriodIndex, TimedeltaIndex, DataFrame
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from torch import Tensor
from torch_sparse import SparseTensor

Scalar = Union[int, float]

TensArray = Union[Tensor, ndarray]
OptTensArray = Optional[TensArray]

ScipySparseMatrix = Union[coo_matrix, csr_matrix, csc_matrix]
SparseTensArray = Union[Tensor, SparseTensor, ndarray, ScipySparseMatrix]
OptSparseTensArray = Optional[SparseTensArray]

FrameArray = Union[DataFrame, ndarray]
OptFrameArray = Optional[FrameArray]

DataArray = Union[DataFrame, ndarray, Tensor]
OptDataArray = Optional[DataArray]

TemporalIndex = Union[DatetimeIndex, PeriodIndex, TimedeltaIndex]

Index = Union[List, Tuple, TensArray]
IndexSlice = Union[slice, Index]

FillOptions = Optional[Literal["backfill", "bfill", "ffill", "pad",
                               "mean", "linear"]]

ModelReturnOptions = Union[Type[Tensor], Type[dict], Type[list]]
