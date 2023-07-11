import math
from typing import Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class MultiConv1d(nn.Module):
    """Applies convolutions with different weights to the different instances in
     the input data."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_instances: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[str, int] = 0,
                 dilation: int = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_instances = n_instances
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty((n_instances * out_channels, in_channels, kernel_size),
                        **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(
                torch.empty(n_instances, out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        """"""
        return f'{self.in_channels}, {self.out_channels}, ' \
               f'kernel_size={self.kernel_size}, n_instances={self.n_instances}'

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_channels * self.kernel_size)
        init.uniform_(self.weight.data, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias.data, -bound, bound)

    def forward(self, x):
        """"""
        x = rearrange(x, '... t n f -> ... (n f) t')

        out = F.conv1d(x,
                       weight=self.weight,
                       bias=None,
                       stride=self.stride,
                       dilation=self.dilation,
                       groups=self.n_instances,
                       padding=self.padding)

        out = rearrange(out, '... (n f) t -> ... t n f', f=self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out


class MultiTemporalConv(nn.Module):
    """
    Multi temporal convolutional filters.

    Inputs are expected to be of shape (batch, steps, n_instances, channels).


    Args:
        input_channels (int): Input size.
        output_channels (int): Output size.
        kernel_size (int): Size of the convolution kernel.
        dilation (int, optional): Spacing between kernel elements.
        stride (int, optional):  Stride of the convolution.
        bias (bool, optional): Whether to add a learnable bias to the output of
            the convolution.
        padding (int, optional): Padding of the input. Used only of
            `causal_padding` is `False`.
        causal_padding (bool, optional): Whether to pad the input as to preserve
            causality.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 n_instances: int,
                 dilation: int = 1,
                 stride: int = 1,
                 bias: bool = True,
                 padding: tuple = 0,
                 causal_padding: bool = True):
        super().__init__()
        self._causal_pad_sizes = (0, 0, 0, 0, (kernel_size - 1) * dilation, 0)
        if causal_padding:
            assert padding == 0
            self.causal_padding = True

        self.conv = MultiConv1d(in_channels=input_channels,
                                out_channels=output_channels,
                                n_instances=n_instances,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=0,
                                dilation=dilation,
                                bias=bias)

    def forward(self, x):
        """"""
        if self.causal_padding:
            x = F.pad(x, self._causal_pad_sizes, mode='constant', value=0.)
        x = self.conv(x)
        return x
