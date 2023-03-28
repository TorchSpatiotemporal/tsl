from torch import nn

from tsl.nn.layers.base import Dense
from tsl.nn.utils import maybe_cat_exog


class MLP(nn.Module):
    """Simple Multi-layer Perceptron encoder with optional linear readout.

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
        layers = [
            Dense(input_size=input_size if i == 0 else hidden_size,
                  output_size=hidden_size,
                  activation=activation,
                  dropout=dropout) for i in range(n_layers)
        ]
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


class ResidualMLP(nn.Module):
    """Multi-layer Perceptron with residual connections.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        exog_size (int, optional): Size of the optional exogenous variables.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
        dropout (float, optional): Dropout probability. (default: 0.)
        parametrized_skip (bool, optional): Whether to use parametrized skip
            connections for the residuals.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 exog_size=None,
                 n_layers=1,
                 activation='relu',
                 dropout=0.,
                 parametrized_skip=False):
        super(ResidualMLP, self).__init__()

        if exog_size is not None:
            input_size += exog_size

        self.layers = nn.ModuleList([
            nn.Sequential(
                Dense(input_size=input_size if i == 0 else hidden_size,
                      output_size=hidden_size,
                      activation=activation,
                      dropout=dropout), nn.Linear(hidden_size, hidden_size))
            for i in range(n_layers)
        ])

        self.skip_connections = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and input_size != output_size:
                self.skip_connections.append(nn.Linear(input_size,
                                                       hidden_size))
            elif parametrized_skip:
                self.skip_connections.append(
                    nn.Linear(hidden_size, hidden_size))
            else:
                self.skip_connections.append(nn.Identity())

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x, u=None):
        """"""
        x = maybe_cat_exog(x, u)
        for layer, skip in zip(self.layers, self.skip_connections):
            x = layer(x) + skip(x)
        if self.readout is not None:
            return self.readout(x)
        return x
