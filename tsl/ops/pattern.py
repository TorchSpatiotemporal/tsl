import re
from collections import Counter
from typing import Iterable, Union, List, Optional

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

_PATTERNS = {
    'tnef': re.compile('^[1-2]?t?(n{0,2}|e?)f*$'),
    'btnef': re.compile('^b?t?(n{0,2}|e?)f*$'),
}


def check_pattern(pattern: str, split: bool = False, ndim: int = None,
                  include_batch: bool = False) -> Union[str, list]:
    r"""Check that :attr:`pattern` is allowed. A pattern is a string of tokens
    interleaved with blank spaces, where each token specifies what an axis in a
    tensor refers to. The supported tokens are:

    * 't', for the time dimension
    * 'n', for the node dimension
    * 'e', for the edge dimension
    * 'f' or 'c', for the feature/channel dimension ('c' token is automatically
      converted to 'f')

    In order to be valid, a pattern must have:

    1. at most one 't' dimension, as the first token;
    2. at most two (consecutive) 'n' dimensions, right after the 't' token or
       at the beginning of the pattern;
    3. at most one 'e' dimension, either as the first token or after a 't';
    3. either 'n' or 'e' dimensions, but not both together;
    4. all further tokens must be 'c' or 'f'.

    Args:
        pattern (str): The input pattern, specifying with a token what an axis
            in a tensor refers to. The supported tokens are:

            * 't', for the time dimension
            * 'n', for the node dimension
            * 'e', for the edge dimension
            * 'f' or 'c', for the feature/channel dimension ('c' token is
              automatically converted to 'f')

        split (bool): If :obj:`True`, then return an ordered list of the tokens
            in the sanitized pattern.
            (default: :obj:`False`)
        ndim (int, optional): If it is not :obj:`None`, then check that
            :attr:`pattern` has :attr:`ndim` tokens.
            (default: :obj:`None`)
        include_batch (bool): If :obj:`True`, then allows the token :obj:`b`.
            (default: :obj:`False`)

    Returns:
        str or list: The sanitized pattern as a string, or a list of the tokens
            in the pattern.
    """
    pattern_squeezed = pattern.replace(' ', '').replace('c', 'f')
    # check 'c'/'f' follows 'n', 'n' follows 't'
    # allow for duplicate 'n' dims (e.g., 'n n', 't n n f')
    # allow for limitless 'c'/'f' dims (e.g., 't n f f')
    #  if include_batch, then allow for batch dimension
    match_with = _PATTERNS['btnef' if include_batch else 'tnef']
    if not match_with.match(pattern_squeezed):
        raise RuntimeError(f'Pattern "{pattern}" not allowed.')
    elif ndim is not None and len(pattern_squeezed) != ndim:
        raise RuntimeError(f'Pattern "{pattern}" has not {ndim} dimensions.')
    if split:
        return list(pattern_squeezed)
    return ' '.join(pattern_squeezed)


def infer_pattern(shape: tuple, t: Optional[int] = None,
                  n: Optional[int] = None, e: Optional[int] = None) -> str:
    out = []
    for dim in shape:
        if t is not None and dim == t:
            out.append('t')
        elif n is not None and dim == n:
            out.append('n')
        elif e is not None and dim == e:
            out.append('e')
        else:
            out.append('f')
    pattern = ' '.join(out)
    try:
        pattern = check_pattern(pattern)
    except RuntimeError:
        raise RuntimeError(f"Cannot infer pattern from shape: {shape}.")
    return pattern


def outer_pattern(patterns: Iterable[str]):
    dims = dict(t=0, n=0, f=0)
    for pattern in patterns:
        dim_count = Counter(check_pattern(pattern, split=True))
        for dim, count in dim_count.items():
            dims[dim] = max(dims[dim], count)
    dims = [d for dim, count in dims.items() for d in [dim] * count]
    return ' '.join(dims)


def take(x: Union[np.ndarray, torch.Tensor], pattern: str,
         time_index: Union[List, ndarray, Tensor] = None,
         node_index: Union[List, ndarray, Tensor] = None):
    dims = check_pattern(pattern, split=True)

    # select backend
    if isinstance(x, np.ndarray):
        def select(obj, index, dim):
            return obj.take(index, dim)
    elif isinstance(x, torch.Tensor):
        def select(obj, index, dim):
            return obj.index_select(dim, index)
        if time_index is not None and not isinstance(time_index, Tensor):
            time_index = torch.as_tensor(time_index, dtype=torch.long)
        if node_index is not None and not isinstance(node_index, Tensor):
            node_index = torch.as_tensor(node_index, dtype=torch.long)
    else:
        raise RuntimeError("Can slice only object of type "
                           "'np.ndarray' and 'torch.Tensor'.")

    # assume that 't' can only be first dimension, then allow multidimensional
    # temporal indexing
    pad_dim = 0
    if dims[0] == 't':
        pad_dim = 1
        if time_index is not None:
            # time_index can be multidimensional
            pad_dim = time_index.ndim
            # pad pattern with 'batch' dimensions
            dims = ['b'] * (pad_dim - 1) + dims
            x = x[time_index]

    # broadcast array/tensor to pattern according to backend
    for pos, dim in list(enumerate(dims))[pad_dim:]:
        if dim == 'n' and node_index is not None:
            x = select(x, node_index, pos)

    return x


def broadcast(x: Union[np.ndarray, torch.Tensor], pattern: str,
              t: int = 1, n: int = 1, f: int = 1,
              time_index: Union[List, ndarray, Tensor] = None,
              node_index: Union[List, ndarray, Tensor] = None):
    # check patterns
    left, rght = pattern.split('->')
    left_dims = check_pattern(left, split=True)
    rght_dims = check_pattern(rght, split=True)
    if not set(left_dims).issubset(rght_dims):
        raise RuntimeError(f"Shape {left_dims} cannot be "
                           f"broadcasted to {rght.strip()}.")

    # select backend
    if isinstance(x, np.ndarray):
        def select(obj, index, dim):
            return obj.take(index, dim)

        def expand(obj, size, dim):
            obj = np.expand_dims(obj, dim)
            if size > 1:
                obj = obj.repeat(size, dim)
            return obj
    elif isinstance(x, torch.Tensor):
        def select(obj, index, dim):
            return obj.index_select(dim, index)

        def expand(obj, size, dim):
            obj = obj.unsqueeze(dim)
            shape = [size if i == dim else -1 for i in range(obj.ndim)]
            return obj.expand(shape)
    else:
        raise RuntimeError("Can broadcast to pattern only object of type "
                           "'np.ndarray' and 'torch.Tensor'.")

    # build indices and default values for broadcasting
    dim_map = dict(t=t, n=n, f=f)
    if time_index is not None:
        dim_map['t'] = len(time_index)
        if isinstance(x, torch.Tensor):
            time_index = torch.as_tensor(time_index, dtype=torch.long)
    if node_index is not None:
        dim_map['n'] = len(node_index)
        if isinstance(x, torch.Tensor):
            node_index = torch.as_tensor(node_index, dtype=torch.long)

    # assume that 't' can only be first dimension, then allow multidimensional
    # temporal indexing
    pad_dim = 1 if rght_dims[0] == 't' else 0
    if time_index is None:
        # n f -> t n f  ==> dim_map['t'] n f
        if rght_dims[0] == 't' and left_dims[0] != 't':
            x = expand(x, dim_map['t'], 0)
            left_dims = ['t'] + left_dims
    else:
        pad_dim = time_index.ndim
        # t n f -> t n f  ==> t[time_index] n f
        if left_dims[0] == 't':
            x = x[time_index]
        # n f -> t n f  ==> t[time_index] n f
        elif rght_dims[0] == 't' and left_dims[0] != 't':
            for p in range(pad_dim):
                x = expand(x, time_index.size(p), p)
            left_dims = ['t'] + left_dims
        left_dims = ['b'] * (pad_dim - 1) + left_dims
        rght_dims = ['b'] * (pad_dim - 1) + rght_dims

    # broadcast array/tensor to pattern according to backend
    for pos, rght_dim in list(enumerate(rght_dims))[pad_dim:]:
        left_dim = left_dims[pos] if pos < len(left_dims) else None
        if left_dim != rght_dim:
            x = expand(x, dim_map[rght_dim], pos)
            left_dims.insert(pos, rght_dim)
        elif rght_dim == 'n' and node_index is not None:
            x = select(x, node_index, pos)
    return x
