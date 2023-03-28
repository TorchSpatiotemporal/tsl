from torch import nn

from tsl.nn import utils


class Dense(nn.Module):
    r"""A simple fully-connected layer implementing

    .. math::

        \mathbf{x}^{\prime} = \sigma\left(\boldsymbol{\Theta}\mathbf{x} +
        \mathbf{b}\right)

    where :math:`\mathbf{x} \in \mathbb{R}^{d_{in}}, \mathbf{x}^{\prime} \in
    \mathbb{R}^{d_{out}}` are the input and output features, respectively,
    :math:`\boldsymbol{\Theta} \in \mathbb{R}^{d_{out} \times d_{in}} \mathbf{b}
    \in \mathbb{R}^{d_{out}}` are trainable parameters, and :math:`\sigma` is
    an activation function.

    Args:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        activation (str, optional): Activation function to be used.
            (default: :obj:`'relu'`)
        dropout (float, optional): The dropout rate.
            (default: :obj:`0`)
        bias (bool, optional): If :obj:`True`, then the bias vector is used.
            (default: :obj:`True`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 bias: bool = True):
        super(Dense, self).__init__()
        self.affinity = nn.Linear(input_size, output_size, bias=bias)
        self.activation = utils.get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        """"""
        out = self.activation(self.affinity(x))
        return self.dropout(out)
