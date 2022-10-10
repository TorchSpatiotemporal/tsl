from typing import Union

import torch
from einops import rearrange
from torch import Tensor
from torch_sparse import SparseTensor

from tsl.ops.framearray import framearray_to_numpy
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
    dtype = tensor.dtype() if isinstance(tensor, SparseTensor) else tensor.dtype
    # float to float{precision}
    if dtype in [torch.float16, torch.float32, torch.float64]:
        new_dtype = getattr(torch, f'float{precision}')
        return tensor.to(new_dtype)
    # int to int{precision}
    elif dtype in [torch.int16, torch.int32, torch.int64]:
        new_dtype = getattr(torch, f'int{precision}')
        return tensor.to(new_dtype)
    return tensor
