import torch.nn as nn
from einops import rearrange

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
                 input_channels,
                 output_channels,
                 kernel_size,
                 dilation=1,
                 stride=1,
                 bias=True,
                 padding=0,
                 causal_pad=True,
                 weight_norm=False,
                 channel_last=False):
        super().__init__()
        if causal_pad:
            self.padding = ((kernel_size - 1) * dilation, 0, 0, 0)
        else:
            self.padding = padding
        self.pad_layer = nn.ZeroPad2d(self.padding)
        # we use Conv2d here to accommodate multiple input sequences
        self.conv = nn.Conv2d(input_channels,
                              output_channels, (1, kernel_size),
                              stride=stride,
                              padding=(0, 0),
                              dilation=(1, dilation),
                              bias=bias)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        self.channel_last = channel_last

    def forward(self, x):
        """"""
        if self.channel_last:
            x = rearrange(x, 'b s n c -> b c n s')
        # batch, channels, nodes, steps
        x = self.pad_layer(x)
        x = self.conv(x)
        if self.channel_last:
            x = rearrange(x, 'b c n s -> b s n c')
        return x


class GatedTemporalConv(TemporalConv):
    """Temporal convolutional filter with gated tanh connection."""

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 dilation=1,
                 stride=1,
                 bias=True,
                 padding=0,
                 causal_pad=True,
                 weight_norm=False,
                 channel_last=False):
        super(GatedTemporalConv,
              self).__init__(input_channels=input_channels,
                             output_channels=2 * output_channels,
                             kernel_size=kernel_size,
                             dilation=dilation,
                             stride=stride,
                             bias=bias,
                             padding=padding,
                             causal_pad=causal_pad,
                             weight_norm=weight_norm,
                             channel_last=channel_last)

    def forward(self, x):
        """"""
        # batch, channels, nodes, steps
        x = super(GatedTemporalConv, self).forward(x)
        dim = -1 if self.channel_last else 1
        return gated_tanh(x, dim=dim)
