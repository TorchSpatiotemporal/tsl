import torch.nn as nn
from einops import rearrange

from tsl.nn.layers.base import GatedTemporalConv, TemporalConv
from tsl.nn.utils import get_functional_activation, maybe_cat_exog


class TemporalConvNet(nn.Module):
    r"""Simple TCN encoder with optional linear readout.

    Args:
        input_channels (int): Input size.
        hidden_channels (int): Channels in the hidden layers.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation coefficient of the convolutional kernel.
        stride (int, optional): Stride of the convolutional kernel.
        output_channels (int, optional): Channels of the optional exogenous
            variables.
        output_channels (int, optional): Channels in the output layer.
        n_layers (int, optional): Number of hidden layers. (default: 1)
        gated (bool, optional): Whether to used the GatedTanH activation
            function.
            (default: :obj:`False`)
        dropout (float, optional): Dropout probability.
        activation (str, optional): Activation function.
            (default: :obj:`'relu'`)
        exponential_dilation (bool, optional): Whether to increase
            exponentially the dilation factor at each layer.
        weight_norm (bool, optional): Whether to apply weight normalization to
            the temporal convolutional filters.
        causal_padding (bool, optional): Whether to pad the input sequence to
            preserve causality.
        bias (bool, optional): Whether to add a learnable bias to the output.
        channel_last (bool, optional): If :obj:`True` input must have layout
            (b s n c), (b c n s) otherwise.
    """

    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 dilation,
                 stride=1,
                 exog_channels=None,
                 output_channels=None,
                 n_layers=1,
                 gated=False,
                 dropout=0.,
                 activation='relu',
                 exponential_dilation=False,
                 weight_norm=False,
                 causal_padding=True,
                 bias=True,
                 channel_last=True):
        super(TemporalConvNet, self).__init__()
        self.channel_last = channel_last
        base_conv = TemporalConv if not gated else GatedTemporalConv

        if exog_channels is not None:
            input_channels += exog_channels

        layers = []
        d = dilation
        for i in range(n_layers):
            if exponential_dilation:
                d = dilation**i
            layers.append(
                base_conv(input_channels=input_channels
                          if i == 0 else hidden_channels,
                          output_channels=hidden_channels,
                          kernel_size=kernel_size,
                          dilation=d,
                          stride=stride,
                          causal_pad=causal_padding,
                          weight_norm=weight_norm,
                          bias=bias))

        self.convs = nn.ModuleList(layers)
        self.f = get_functional_activation(
            activation) if not gated else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if output_channels is not None:
            self.readout = TemporalConv(input_channels=hidden_channels,
                                        output_channels=output_channels,
                                        kernel_size=1)
        else:
            self.register_parameter('readout', None)

    def forward(self, x, u=None):
        """"""
        if self.channel_last:
            x = maybe_cat_exog(x, u, -1)
            x = rearrange(x, 'b s n c -> b c n s')
        else:
            x = maybe_cat_exog(x, u, 1)

        for conv in self.convs:
            x = self.dropout(self.f(conv(x)))
        if self.readout is not None:
            x = self.readout(x)
        if self.channel_last:
            x = rearrange(x, 'b c n s -> b s n c')
        return x
