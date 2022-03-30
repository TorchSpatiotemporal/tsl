from typing import Optional

import torch
from torch.nn import functional as F
from torch import nn

from einops import rearrange

from ..ops import expand_then_cat


def _identity(x):
    return x


def get_functional_activation(activation: Optional[str] = None):
    if activation is None:
        return _identity
    activation = activation.lower()
    if activation == 'linear':
        return _identity
    # todo extend with all activations
    torch_activations = ['elu', 'leaky_relu', 'relu', 'sigmoid', 'softplus', 'tanh']
    if activation in torch_activations:
        return getattr(F, activation)
    raise ValueError(f"Activation '{activation}' not valid.")


def get_layer_activation(activation: Optional[str] = None):
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    # todo extend with all activations
    torch_activations_dict = {'elu': 'ELU',
                        'leaky_relu': 'LeakyReLU',
                        'relu': 'ReLU',
                        'sigmoid': 'Sigmoid',
                        'softplus': 'Softplus',
                        'tanh': 'Tanh',
                        'silu': 'SiLU',
                        'swish': 'SiLU',
                        'linear': 'Identity'
                        }
    if activation in torch_activations_dict:
        return getattr(nn, torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")


def maybe_cat_exog(x, u, dim=-1):
    r"""
    Concatenate `x` and `u` if `u` is not `None`.

    We assume `x` to be a 4-dimensional tensor, if `u` has only 3 dimensions we assume it to be a global exog variable.

    Args:
        x: Input 4-d tensor.
        u: Optional exogenous variable.

    Returns:
        Concatenated `x` and `u`.
    """
    if u is not None:
        if u.dim() == 3:
            u = rearrange(u, 'b s f -> b s 1 f')
        x = expand_then_cat([x, u], dim)
    return x
