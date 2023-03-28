from torch import nn

from tsl.nn.layers.multi import MultiDense, MultiLinear
from tsl.nn.utils import maybe_cat_exog


class MultiMLP(nn.Module):
    """A multi-layer perceptron (MLP) (with optional linear readout) with
    different weights for each element in the specified dimension.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int, optional): Size of the optional readout.
        exog_size (int, optional): Size of the optional exogenous variables.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        activation (str, optional): Activation function. (default: `relu`)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_instances: int,
                 *,
                 ndim: int = None,
                 pattern: str = None,
                 instance_dim: int = -2,
                 output_size: int = None,
                 exog_size: int = None,
                 n_layers: int = 1,
                 activation: str = 'relu',
                 dropout: float = 0.):
        super(MultiMLP, self).__init__()
        if exog_size is not None:
            input_size += exog_size
        layers = [
            MultiDense(input_size if i == 0 else hidden_size,
                       hidden_size,
                       n_instances,
                       ndim=ndim,
                       pattern=pattern,
                       instance_dim=instance_dim,
                       dropout=dropout,
                       activation=activation) for i in range(n_layers)
        ]
        if output_size is not None:
            layers += [
                MultiLinear(hidden_size,
                            output_size,
                            n_instances,
                            ndim=ndim,
                            pattern=pattern,
                            instance_dim=instance_dim)
            ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, u=None):
        """"""
        x = maybe_cat_exog(x, u)
        out = self.mlp(x)
        return out
