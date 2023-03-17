from torch import nn

from tsl.nn.utils import get_layer_activation


class Activation(nn.Module):
    r"""A utility layer for any activation function.

    Args:
        activation (str): Name of the activation function.
        **kwargs: Keyword arguments for the activation layer.
    """

    def __init__(self, activation: str, **kwargs):
        super(Activation, self).__init__()
        activation_class = get_layer_activation(activation)
        self.activation: nn.Module = activation_class(**kwargs)

    def forward(self, x):
        """Returns :obj:`self.activation(input)`."""
        return self.activation(x)
