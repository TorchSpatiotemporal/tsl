from typing import Optional

from torch.nn import functional as F
from torch import nn


def get_functional_activation(activation: Optional[str] = None):
    def identity(x):
        return x

    if activation is None:
        return identity
    activation = activation.lower()
    if activation == 'linear':
        return identity
    # todo extend with all activations
    torch_activations = ['elu', 'leaky_relu', 'relu', 'sigmoid', 'softplus', 'tanh']
    if activation in torch_activations:
        return getattr(F, activation)
    raise ValueError(f"Activation '{activation}' not valid.")


def get_layer_activation(activation: Optional[str] = None):
    if activation is None:
        return nn.Identity
    activation = activation.lower()
    if activation == 'linear':
        return nn.Identity
    # todo extend with all activations
    torch_activations_dict = {'elu': 'ELU',
                        'leaky_relu': 'LeakyReLU',
                        'relu': 'ReLU',
                        'sigmoid': 'Sigmoid',
                        'softplus': 'Softplus',
                        'tanh': 'Tanh'}
    if activation in torch_activations_dict:
        return getattr(nn, torch_activations_dict[activation])
    raise ValueError(f"Activation '{activation}' not valid.")
