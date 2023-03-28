from typing import Any, Literal, Union

import torch
from einops import rearrange
from torch import Tensor
from torch_geometric.data.storage import recursive_apply
from torch_sparse import SparseTensor

from tsl.ops.framearray import framearray_to_numpy
from tsl.typing import IndexSlice
from tsl.utils.python_utils import precision_stoi


def to_time_nodes_channels(obj):
    if obj.ndim == 2:  # [time, nodes] -> [time, nodes, 1 feature]
        obj = rearrange(obj, 't (n f) -> t n f', f=1)
    elif obj.ndim == 1:  # [time] -> [time, 1 node, 1 feature]
        obj = rearrange(obj, '(t n f) -> t n f', n=1, f=1)
    elif obj.ndim != 3:
        raise ValueError(f'Invalid data dimensions {obj.shape}')
    return obj


def copy_to_tensor(obj) -> Tensor:
    if isinstance(obj, torch.Tensor):
        obj = obj.clone().detach()
    else:
        obj = torch.as_tensor(framearray_to_numpy(obj))
    obj = torch.atleast_1d(obj)
    return obj


def convert_precision_tensor(tensor: Union[Tensor, SparseTensor],
                             precision: Union[int, str] = None) \
        -> Union[Tensor, SparseTensor]:
    if precision is None:
        return tensor
    precision = precision_stoi(precision)
    dtype = tensor.dtype() if isinstance(tensor,
                                         SparseTensor) else tensor.dtype
    # float to float{precision}
    if dtype in [torch.float16, torch.float32, torch.float64]:
        new_dtype = getattr(torch, f'float{precision}')
        return tensor.to(new_dtype)
    # int to int{precision}
    elif dtype in [torch.int16, torch.int32, torch.int64]:
        new_dtype = getattr(torch, f'int{precision}')
        return tensor.to(new_dtype)
    return tensor


def torch_to_numpy(tensors: Any) -> Any:
    """Cast tensors to numpy arrays.

    Args:
        tensors: A tensor or a list or dictionary containing tensors.

    Returns:
        Tensors casted to numpy arrays.
    """
    return recursive_apply(tensors, lambda t: t.detach().cpu().numpy())


def parse_index(index: IndexSlice = None,
                length: int = None,
                layout: Literal['index', 'slice', 'mask'] = 'index'):
    if index is None:
        return slice(None) if layout == 'slice' else None
    if isinstance(index, slice):
        if layout == 'slice':
            return index
        assert length is not None, "'length' cannot be None with 'slice' layout"
        index = torch.tensor(range(length)[index])
    if not isinstance(index, Tensor):
        index = torch.as_tensor(index, dtype=torch.long)
    if layout == 'mask':
        assert length is not None, "'length' cannot be None with 'mask' layout"
        mask = torch.zeros(length, dtype=torch.bool)
        mask[index] = True
        return mask
    return index
