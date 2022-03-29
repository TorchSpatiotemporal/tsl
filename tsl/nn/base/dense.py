from torch import nn

from tsl.nn.utils import utils


class Dense(nn.Module):
    r"""
    A simple fully-connected layer.

    Args:
        input_size (int): Size of the input.
        output_size (int): Size of the output.
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout rate.
        bias (bool, optional): Whether to use a bias.
    """
    def __init__(self, input_size, output_size, activation='linear', dropout=0., bias=True):
        super(Dense, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
            utils.get_layer_activation(activation)(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)
