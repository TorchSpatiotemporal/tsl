import re
from collections import Counter
from types import ModuleType
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

_PATTERNS = {
    'tnef': re.compile('^[1-2]?t?(n{0,2}|e?)f*$'),
    'btnef': re.compile('^b?t?(n{0,2}|e?)f*$'),
}

#  PATTERN PARSING ############################################################


def check_pattern(pattern: str,
                  split: bool = False,
                  ndim: int = None,
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
    4. either 'n' or 'e' dimensions, but not both together;
    5. all further tokens must be 'c' or 'f'.


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


def infer_pattern(shape: tuple,
                  t: Optional[int] = None,
                  n: Optional[int] = None,
                  e: Optional[int] = None) -> str:
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
    dims = dict(t=0, n=0, e=0, f=0)
    for pattern in patterns:
        dim_count = Counter(check_pattern(pattern, split=True))
        for dim, count in dim_count.items():
            dims[dim] = max(dims[dim], count)
    dims = [d for dim, count in dims.items() for d in [dim] * count]
    if 'n' in dims and 'e' in dims:
        raise RuntimeError("Cannot join node-level and edge-level tensors.")
    return ' '.join(dims)


#  PATTERN-BASED OPERATIONS ###################################################


def _infer_backend(obj, backend: ModuleType = None):
    if backend is not None:
        return backend
    elif isinstance(obj, Tensor):
        return torch
    elif isinstance(obj, np.ndarray):
        return np
    raise RuntimeError(f"Cannot infer valid backed from {type(obj)}. "
                       "Expected backends are 'torch' and 'numpy'.")


def _parse_indices(backend,
                   time_index: Union[List, ndarray, Tensor] = None,
                   node_index: Union[List, ndarray, Tensor] = None,
                   edge_mask: Union[List, ndarray, Tensor] = None):
    indices = [time_index, node_index, edge_mask]
    if backend is torch:
        for i, index in enumerate(indices):
            if index is not None:
                if not isinstance(index, Tensor):
                    index = torch.as_tensor(index)
                if index.ndim == 1 and index.dtype is torch.bool:
                    index = index.nonzero(as_tuple=True)[0]
            indices[i] = index
    elif backend is np:
        for i, index in enumerate(indices):
            if index is not None:
                index = np.asarray(index)
                if index.ndim == 1 and index.dtype == bool:
                    index = index.nonzero()[0]
            indices[i] = index
    return indices


def _get_select_fn(backend):
    select = None
    if backend is np:

        def select(obj, index, dim):
            return obj.take(index, dim)
    elif backend is torch:

        def select(obj, index, dim):
            return obj.index_select(dim, index)

    return select


def _get_expand_fn(backend):
    expand = None
    if backend is np:

        def expand(obj, size, dim):
            obj = np.expand_dims(obj, dim)
            if size > 1:
                obj = obj.repeat(size, dim)
            return obj
    elif backend is torch:

        def expand(obj, size, dim):
            obj = obj.unsqueeze(dim)
            shape = [size if i == dim else -1 for i in range(obj.ndim)]
            return obj.expand(shape)

    return expand


def take(x: Union[np.ndarray, torch.Tensor],
         pattern: str,
         time_index: Union[List, ndarray, Tensor] = None,
         node_index: Union[List, ndarray, Tensor] = None,
         edge_mask: Union[List, ndarray, Tensor] = None,
         *,
         backend: ModuleType = None):
    backend = _infer_backend(x, backend)
    dims = check_pattern(pattern, split=True)

    select = _get_select_fn(backend)
    time_index, node_index, edge_index = _parse_indices(backend,
                                                        time_index=time_index,
                                                        node_index=node_index,
                                                        edge_mask=edge_mask)

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
        elif dim == 'e' and edge_index is not None:
            x = select(x, edge_index, pos)

    return x


def broadcast(x: Union[np.ndarray, torch.Tensor],
              pattern: str,
              time_index: Union[List, ndarray, Tensor] = None,
              node_index: Union[List, ndarray, Tensor] = None,
              edge_mask: Union[List, ndarray, Tensor] = None,
              *,
              t: int = 1,
              n: int = 1,
              e: int = 1,
              f: int = 1,
              backend: ModuleType = None):
    # check patterns
    left, rght = pattern.split('->')
    left_dims = check_pattern(left, split=True)
    rght_dims = check_pattern(rght, split=True)
    if not set(left_dims).issubset(rght_dims):
        raise RuntimeError(f"Shape {left_dims} cannot be "
                           f"broadcasted to {rght.strip()}.")

    select = _get_select_fn(backend)
    expand = _get_expand_fn(backend)
    time_index, node_index, edge_index = _parse_indices(backend,
                                                        time_index=time_index,
                                                        node_index=node_index,
                                                        edge_mask=edge_mask)

    # build indices and default values for broadcasting
    dim_map = dict(t=t if time_index is None else len(time_index),
                   n=n if node_index is None else len(node_index),
                   e=e if edge_index is None else len(edge_index),
                   f=f)

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
        elif rght_dim == 'e' and edge_index is not None:
            x = select(x, edge_index, pos)
    return x
