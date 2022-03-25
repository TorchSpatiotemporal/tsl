import torch
from torch import nn

from torch.nn import functional as F

from ...utils.utils import maybe_cat_exog, get_layer_activation


class MLP(nn.Module):
    r"""
    Simple Multi-layer Perceptron encoder with optional linear readout.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        exog_size (int, optional): Size of the optional exogenous variables.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
        dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 exog_size=None,
                 n_layers=1,
                 activation='relu',
                 dropout=0.):
        super(MLP, self).__init__()

        if exog_size is not None:
            input_size += exog_size

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(get_layer_activation(activation)())
            if dropout > 0.:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x, u=None):
        """"""
        x = maybe_cat_exog(x, u)
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out

