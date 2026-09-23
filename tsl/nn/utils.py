from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from tsl.typing import Tensor

__all__ = [
    'get_functional_activation',
    'get_layer_activation',
    'expand_then_cat',
    'maybe_cat_exog',
    'broadcast',
]

_torch_activations_dict = {
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'selu': 'SELU',
    'celu': 'CELU',
    'gelu': 'GELU',
    'glu': 'GLU',
    'mish': 'Mish',
    'sigmoid': 'Sigmoid',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'silu': 'SiLU',
    'swish': 'SiLU',
    'linear': 'Identity',
}


def _identity(x):
    return x


def get_functional_activation(activation: Optional[str] = None):
    if activation is None:
        return _identity
    activation = activation.lower()
    if activation == 'linear':
        return _identity
    if activation in ['tanh', 'sigmoid']:
        return getattr(torch, activation)
    if activation in _torch_activations_dict:
        return getattr(F, activation)
    raise ValueError(f"Activation '{activation}' not valid.")


def get_layer_activation(activation: Optional[str] = None):
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    if activation in _torch_activations_dict:
        return getattr(nn, _torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")


def expand_then_cat(
    tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim: int = -1
) -> Tensor:
    """Match the dimensions of tensors in the input list and then concatenate.

    Args:
        tensors (list): Tensors to concatenate.
        dim (int): Dimension along which to concatenate.
            (default: :obj:`-1`)
    """
    shapes = [t.shape for t in tensors]
    expand_dims = torch.max(torch.tensor(shapes), 0).values
    expand_dims[dim] = -1
    tensors = [t.expand(*expand_dims) for t in tensors]
    return torch.cat(tensors, dim=dim)


def maybe_cat_exog(x, u, dim=-1):
    r"""
    Concatenate `x` and `u` if `u` is not `None`.

    We assume `x` to be a 4-dimensional tensor, if `u` has only 3 dimensions we
    assume it to be a global exog variable.

    Args:
        x: Input 4-d tensor.
        u: Optional exogenous variable.
        dim (int): Concatenation dimension.

    Returns:
        Concatenated `x` and `u`.
    """
    if u is not None:
        if u.dim() == 3:
            u = rearrange(u, 'b s f -> b s 1 f')
        x = expand_then_cat([x, u], dim)
    return x


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """Broadcast a tensor to the shape of another tensor.

    This function has been adapted from `torch_scatter` to avoid the dependency on the
    `torch_scatter` package.

    Args:
        src (Tensor): The tensor to broadcast.
        other (Tensor): The tensor to match the shape of.
        dim (int): The dimension along which to broadcast.
    Returns:
        Tensor: The broadcasted tensor.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src
