from enum import Enum
from typing import Iterable, Optional, Union, List

import torch
from einops import rearrange
from torch import Tensor

import tsl


class SynchMode(Enum):
    WINDOW = 'window'
    HORIZON = 'horizon'


# Aliases
WINDOW = SynchMode.WINDOW
HORIZON = SynchMode.HORIZON


def to_steps_nodes_channels(obj):
    if obj.ndim == 3:
        tsl.logger.info('Inferred input data-format: [steps, nodes, channels]')
    elif obj.ndim == 2:  # [steps, nodes] -> [steps, nodes, 1 channel]
        tsl.logger.info('Inferred input data-format: [steps, nodes]')
        obj = rearrange(obj, 's (n c) -> s n c', c=1)
    elif obj.ndim == 1:  # [steps] -> [steps, 1 node, 1 channel]
        tsl.logger.info('Inferred input data-format: [steps]')
        obj = rearrange(obj, '(s n c) -> s n c', n=1, c=1)
    else:
        raise ValueError(f'Invalid data dimensions {obj.shape}')
    return obj


def to_steps_channels(obj):
    if obj.ndim == 2:
        tsl.logger.info('Inferred input data-format: [steps, channels]')
    elif obj.ndim == 1:  # [steps] -> [steps, 1 channel]
        tsl.logger.info('Inferred input data-format: [steps]')
        obj = rearrange(obj, '(s c) -> s c', c=1)
    else:
        raise ValueError(f'Invalid data dimensions {obj.shape}')
    return obj


def to_nodes_channels(obj):
    if obj.ndim == 2:
        tsl.logger.info('Inferred input data-format: [nodes, channels]')
    elif obj.ndim == 1:  # [nodes] -> [nodes, 1 channel]
        tsl.logger.info('Inferred input data-format: [nodes]')
        obj = rearrange(obj, '(n c) -> n c', c=1)
    else:
        raise ValueError(f'Invalid data dimensions {obj.shape}')
    return obj


def copy_to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.clone().detach()
    else:
        obj = torch.as_tensor(obj)
    obj = torch.atleast_1d(obj)
    return obj


def cast_tensor(obj: torch.Tensor, precision: Union[int, str] = 32):
    if isinstance(precision, str):
        precision = dict(half=16, full=32, double=64).get(precision)
    assert precision in [16, 32, 64], \
        "precision must be one of 16 (or 'half'), 32 (or 'full') or 64 " \
        f"(or 'double'). Default is 32, invalid input '{precision}'."
    if obj.dtype in [torch.float16, torch.float32, torch.float64]:
        dtype = getattr(torch, f'float{precision}')
        return obj.to(dtype)
    elif obj.dtype in [torch.int16, torch.int32, torch.int64]:
        dtype = getattr(torch, f'int{precision}')
        return obj.to(dtype)
    elif obj.dtype is torch.bool:
        return obj.byte()
    return obj


def parse_pattern(pattern: str):
    pattern = pattern.strip()
    dims = pattern.split(' ')
    try:
        # check elements are only 's' 'n' 'c'
        assert set(dims).issubset('snc')
        # check elements are not repeated
        assert len(set(dims)) == len(dims)
        # allowed shapes only 's n c', 's c', 'n c', 'c'
        assert dims == sorted(dims, reverse=True)
    except AssertionError:
        raise ValueError(f'Pattern "{pattern}" not allowed.')
    return dims


def outer_pattern(patterns: Iterable[str]):
    dims = {dim for p in patterns for dim in parse_pattern(p)}
    dims = sorted(dims, reverse=True)
    return ' '.join(dims)


def broadcast(x, pattern: str,
              s: Optional[int] = None, n: Optional[int] = None,
              step_index: Union[List, Tensor] = None,
              node_index: Union[List, Tensor] = None):
    # check patterns
    left, rght = pattern.split('->')
    left_dims = parse_pattern(left)
    rght_dims = parse_pattern(rght)
    if not set(left_dims).issubset(rght_dims):
        raise RuntimeError(f"Shape {left_dims} cannot be "
                           f"broadcasted to {rght.strip()}.")

    dim_map = dict(s=s, n=n)
    if step_index is not None:
        step_index = torch.as_tensor(step_index, dtype=torch.long)
        dim_map['s'] = step_index.size(0)
    if node_index is not None:
        node_index = torch.as_tensor(node_index, dtype=torch.long)
        dim_map['n'] = node_index.size(0)
    if 's' in rght_dims and 's' not in left_dims and s is None:
        raise RuntimeError("Cannot infer dimension for s")
    if 'n' in rght_dims and 'n' not in left_dims and n is None:
        raise RuntimeError("Cannot infer dimension for n")

    for pos, rght_dim in enumerate(rght_dims):
        left_dim = left_dims[pos]
        if left_dim != rght_dim:
            x = x.unsqueeze(pos)
            shape = [dim_map[rght_dim] if i == pos else -1
                     for i in range(x.ndim)]
            x = x.expand(shape)
            left_dims.insert(pos, rght_dim)
        elif rght_dim == 's' and step_index is not None:
            x = x.index_select(pos, step_index)
        elif rght_dim == 'n' and node_index is not None:
            x = x.index_select(pos, node_index)
    return x
