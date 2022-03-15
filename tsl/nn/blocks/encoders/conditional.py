from torch import nn as nn
from torch.nn import Module
from torch.nn import functional as F

from tsl.nn.base import TemporalConv2d, GatedTemporalConv2d
from tsl.nn.utils.utils import get_layer_activation


class ConditionalBlock(Module):
    r"""Simple layer to condition the input on a set of exogenous variables.

    .. math::
        \text{CondBlock}(\mathbf{x}, \mathbf{u}) =
         \left(\text{MLP}_x(\mathbf{x})\right) +
         \left(\text{MLP}_u(\mathbf{u})\right)

    Args:
        input size (int): Input size.
        exog_size (int): Size of the covariates.
        output_size (int): Output size.
        dropout (float, optional): Dropout probability.
        skip_connection (bool, optional): Whether to add a parametrized residual
            connection.
            (default: `False`).
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size,
                 exog_size,
                 output_size,
                 dropout=0.,
                 skip_connection=False,
                 activation='relu'):
        super().__init__()
        self.d_in = input_size
        self.d_u = exog_size
        self.d_out = output_size
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout)

        # inputs module
        self.input_affinity = nn.Linear(self.d_in, self.d_out)
        self.condition_affinity = nn.Linear(self.d_u, self.d_out)

        self.out_inputs_affinity = nn.Linear(self.d_out, self.d_out)
        self.out_cond_affinity = nn.Linear(self.d_out, self.d_out, bias=False)
        if skip_connection:
            self.skip_conn = nn.Linear(self.d_in, self.d_out)
        else:
            self.register_parameter('skip_conn', None)

    def forward(self, x, u=None):
        if u is None:
            x, u = x
        # *, features

        # inputs block
        out = self.activation(self.input_affinity(x))
        # conditions block
        conditions = self.activation(self.condition_affinity(u))

        out = self.out_inputs_affinity(out) + self.out_cond_affinity(conditions)
        out = self.dropout(self.activation(out))
        if self.skip_conn is not None:
            out = self.skip_conn(x) + out
        return out


class ConditionalTCNBlock(nn.Module):
    r"""
    Mirrors the architecture of `ConditionalBlock` but using temporal convolutions instead of affine transformations.

    Args:
        input_size (int): Size of the input.
        exog_size (int): Size of the exogenous variables.
        output_size (int): Size of the output.
        kernel_size (int): Size of the convolution kernel.
        dilation (int, optional): Spacing between kernel elements.
        dropout (float, optional): Dropout probability.
        gated (bool, optional): Whether to use `gated tanh` activations.
        activation (str, optional): Activation function.
        weight_norm (bool, optional): Whether to apply weight normalization to the parameters of the filter.
        channel_last (bool, optional): If `True` input data must follow the `B S N C` layout, assumes `B C N S` otherwise.
        skip_connection (bool, optional): If `True` adds a parametrized skip connection from the input to the output.
    """
    def __init__(self,
                 input_size,
                 exog_size,
                 output_size,
                 kernel_size,
                 dilation=1,
                 dropout=0.,
                 gated=False,
                 activation='relu',
                 weight_norm=False,
                 channel_last=True,
                 skip_connection=False):
        super().__init__()

        if gated:
            # inputs module
            self.inputs_conv = nn.Sequential(
                GatedTemporalConv2d(input_channels=input_size,
                                    output_channels=output_size,
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    weight_norm=weight_norm,
                                    channel_last=channel_last),
                nn.Dropout(dropout)
            )
            self.conditions_conv = nn.Sequential(
                GatedTemporalConv2d(input_channels=exog_size,
                                    output_channels=output_size,
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    weight_norm=weight_norm,
                                    channel_last=channel_last),
                nn.Dropout(dropout)
            )
        else:
            # inputs module
            self.inputs_conv = nn.Sequential(
                TemporalConv2d(input_channels=input_size,
                               output_channels=output_size,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               weight_norm=weight_norm),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )
            self.conditions_conv = nn.Sequential(
                TemporalConv2d(input_channels=exog_size,
                               output_channels=output_size,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               weight_norm=weight_norm),
                get_layer_activation(activation)(),
                nn.Dropout(dropout)
            )
        self.out_input = nn.Linear(output_size, output_size)
        self.out_cond = nn.Linear(output_size, output_size, bias=False)
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout)

        if skip_connection:
            self.skip_conn = TemporalConv2d(input_size, output_size, 1, channel_last=channel_last)
        else:
            self.register_parameter('skip_conn', None)

    def forward(self, x, u=None):
        """"""
        if u is None:
            x, u = x
        # inputs block
        out = self.inputs_conv(x)
        # conditions block
        conditions = self.conditions_conv(u)

        out = self.out_input(out) + self.out_input(conditions)
        out = self.dropout(self.activation(out))
        if self.skip_conn is not None:
            out = self.skip_conn(out)
        return out
