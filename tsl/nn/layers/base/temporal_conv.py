from typing import Tuple, Union

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from tsl.nn.functional import gated_tanh


class TemporalConv(nn.Module):
    """Learns a standard temporal convolutional filter.

    Args:
        input_channels (int): Input size.
        output_channels (int): Output size.
        kernel_size (int): Size of the convolution kernel.
        dilation (int, optional): Spacing between kernel elements.
        stride (int, optional):  Stride of the convolution.
        bias (bool, optional): Whether to add a learnable bias to the output of
            the convolution.
        padding (int or tuple, optional): Padding of the input. Used only of
            `causal_pad` is `False`.
        causal_pad (bool, optional): Whether to pad the input as to preserve
            causality.
        weight_norm (bool, optional): Wheter to apply weight normalization to
            the parameters of the filter.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 stride: int = 1,
                 bias: bool = True,
                 padding: Union[int, Tuple[int]] = 0,
                 causal_pad: bool = True,
                 weight_norm: bool = False,
                 channel_last: bool = False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.bias = bias
        self.padding = padding
        self.causal_pad = causal_pad
        self.weight_norm = weight_norm
        self.channel_last = channel_last

        if causal_pad:
            padding = ((kernel_size - 1) * dilation, 0, 0, 0)
        elif isinstance(padding, int):
            padding = (padding, padding, 0, 0)
        elif isinstance(padding, (list, tuple)):
            padding = (padding[0], padding[1], 0, 0)
        self.pad_layer = nn.ZeroPad2d(padding)

        # We use Conv2d here to accommodate multiple input sequences
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=(1, kernel_size),
                              stride=(1, stride),
                              padding=(0, 0),
                              dilation=(1, dilation),
                              bias=bias)
        if self.weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def __repr__(self):
        s = ('{cls}({input_channels}, {output_channels}, '
             'kernel_size={kernel_size}, stride={stride}')
        if self.causal_pad:
            s += ', causal_padding={padding}'
        elif self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if not self.bias:
            s += ', bias=False'
        return (s + ')').format(cls=self.__class__.__name__, **self.__dict__)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        if self.channel_last:
            x = rearrange(x, 'b t n f -> b f n t')
        # x: [batch, features, nodes, time]
        x = self.pad_layer(x)
        x = self.conv(x)
        if self.channel_last:
            x = rearrange(x, 'b f n t -> b t n f')
        return x


class GatedTemporalConv(TemporalConv):
    """Temporal convolutional filter with gated tanh connection."""

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 stride: int = 1,
                 bias: bool = True,
                 padding: Union[int, Tuple[int]] = 0,
                 causal_pad: bool = True,
                 weight_norm: bool = False,
                 channel_last: bool = False):
        super(GatedTemporalConv, self).__init__(
            input_channels=input_channels,
            output_channels=2 * output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            bias=bias,
            padding=padding,
            causal_pad=causal_pad,
            weight_norm=weight_norm,
            channel_last=channel_last,
        )

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = super(GatedTemporalConv, self).forward(x)
        dim = -1 if self.channel_last else 1
        return gated_tanh(x, dim=dim)
