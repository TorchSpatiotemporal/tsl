import re
from collections import Counter
from typing import Iterable, Optional, Union, List

import torch
from torch import Tensor

PATTERN_MATCH = re.compile('^(t?){2}(n?){2}[f]*$')


def check_pattern(pattern: str, split: bool = False) -> Union[str, list]:
    pattern_squeezed = pattern.replace(' ', '').replace('c', 'f')
    # check 'c'/'f' follows 'n', 'n' follows 't'
    # allow for duplicate 'n' or 't' dims (e.g., 'n n', 't t n f')
    # allow for limitless 'c'/'f' dims (e.g., 't n f f')
    if not PATTERN_MATCH.match(pattern_squeezed):
        raise RuntimeError(f'Pattern "{pattern}" not allowed.')
    if split:
        return list(pattern_squeezed)
    return ' '.join(pattern_squeezed)


def outer_pattern(patterns: Iterable[str]):
    dims = dict(t=0, n=0, f=0)
    for pattern in patterns:
        dim_count = Counter(check_pattern(pattern))
        for dim, count in dim_count.items():
            dims[dim] = max(dims[dim], count)
    dims = [d for dim, count in dims.items() for d in [dim] * count]
    return ' '.join(dims)


def broadcast(x, pattern: str,
              t: Optional[int] = None, n: Optional[int] = None,
              time_index: Union[List, Tensor] = None,
              node_index: Union[List, Tensor] = None):
    # check patterns
    left, rght = pattern.split('->')
    left_dims = check_pattern(left, split=True)
    rght_dims = check_pattern(rght, split=True)
    if not set(left_dims).issubset(rght_dims):
        raise RuntimeError(f"Shape {left_dims} cannot be "
                           f"broadcasted to {rght.strip()}.")

    dim_map = dict(t=t, n=n)
    if time_index is not None:
        time_index = torch.as_tensor(time_index, dtype=torch.long)
        dim_map['t'] = time_index.size(0)
    if node_index is not None:
        node_index = torch.as_tensor(node_index, dtype=torch.long)
        dim_map['n'] = node_index.size(0)
    if 't' in rght_dims and 't' not in left_dims and t is None:
        raise RuntimeError("Cannot infer dimension for t")
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
        elif rght_dim == 't' and time_index is not None:
            x = x.index_select(pos, time_index)
        elif rght_dim == 'n' and node_index is not None:
            x = x.index_select(pos, node_index)
    return x
