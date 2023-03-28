from torch import Tensor, nn

from tsl.nn.utils import get_layer_activation

from .linear import MultiLinear


class MultiDense(MultiLinear):
    r"""Applies linear transformations with different weights to the different
    instances in the input data with a final nonlinear activation.

    .. math::

        \mathbf{X}^{\prime} = \left[\sigma\left(\boldsymbol{\Theta}_i
        \mathbf{x}_i + \mathbf{b}_i \right)\right]_{i=0,\ldots,N}

    Args:
        in_channels (int): Size of instance input sample.
        out_channels (int): Size of instance output sample.
        n_instances (int): The number :math:`N` of parallel linear
            operations. Each operation has different weights and biases.
        activation (str, optional): Activation function to be used.
            (default: :obj:`'relu'`)
        dropout (float, optional): Dropout rate.
            (default: :obj:`0`)
        instance_dim (int or str): Dimension of the instances (must match
            :attr:`n_instances` at runtime).
            (default: :obj:`-2`)
        channel_dim (int or str): Dimension of the input channels.
            (default: :obj:`-1`)
        bias (bool): If :obj:`True`, then the layer will learn an additive
            bias for each instance.
            (default: :obj:`True`)
        device (optional): The device of the parameters.
            (default: :obj:`None`)
        dtype (optional): The data type of the parameters.
            (default: :obj:`None`)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_instances: int,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 *,
                 ndim: int = None,
                 pattern: str = None,
                 instance_dim: int = -2,
                 channel_dim: int = -1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super(MultiDense, self).__init__(in_channels,
                                         out_channels,
                                         n_instances=n_instances,
                                         ndim=ndim,
                                         pattern=pattern,
                                         instance_dim=instance_dim,
                                         channel_dim=channel_dim,
                                         bias=bias,
                                         device=device,
                                         dtype=dtype)
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        r"""Compute :math:`\mathbf{X}^{\prime} = \left[\sigma\left(\boldsymbol{
        \Theta}_i\mathbf{x}_i + \mathbf{b}_i \right)\right]_{i=0,\ldots,N}`."""
        out = self.activation(super().forward(input))
        return self.dropout(out)
