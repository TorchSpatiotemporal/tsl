from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import (
        DataFrame,
        DatetimeIndex,
        PeriodIndex,
        Series,
        TimedeltaIndex,
    )
    from pandas import (
        Index as PandasIndex,
    )
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
    from torch import LongTensor, Tensor
    from torch_geometric.typing import Adj, OptPairTensor, OptTensor, PairTensor
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
    FillOptions = Optional[
        Literal["backfill", "bfill", "ffill", "pad", "mean", "linear"]
    ]
    ModelReturnOptions = Type[Union[Tensor, Dict, List, Tuple]]
else:
    ndarray = Any
    DataFrame = Any
    DatetimeIndex = Any
    PandasIndex = Any
    PeriodIndex = Any
    Series = Any
    TimedeltaIndex = Any
    coo_matrix = Any
    csc_matrix = Any
    csr_matrix = Any
    Tensor = Any
    LongTensor = Any
    Adj = Any
    OptPairTensor = Any
    OptTensor = Any
    PairTensor = Any
    Scalar = Any
    TensArray = Any
    OptTensArray = Any
    ScipySparseMatrix = Any
    SparseTensor = Any
    SparseTensArray = Any
    OptSparseTensArray = Any
    TorchConnectivity = Any
    FrameArray = Any
    OptFrameArray = Any
    DataArray = Any
    OptDataArray = Any
    TemporalIndex = Any
    Index = Any
    IndexSlice = Any
    FillOptions = Any
    ModelReturnOptions = Any
