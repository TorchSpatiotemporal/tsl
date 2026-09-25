from typing import Optional

from torch import Tensor
from torch import nn as nn
from torch.nn import Module
from torch.nn import functional as F

from tsl.nn.layers.base import GatedTemporalConv, TemporalConv
from tsl.nn.utils import get_layer_activation


class ConditionalBlock(Module):
    r"""Simple layer to condition the input on a set of exogenous variables.

    .. math::
        \text{CondBlock}(\boldsymbol{x}, \boldsymbol{u}) =
         \left(\text{MLP}_x(\boldsymbol{x})\right) +
         \left(\text{MLP}_u(\boldsymbol{u})\right)

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

    def __init__(
        self,
        input_size,
        exog_size,
        output_size,
        dropout=0.0,
        skip_connection=False,
        activation='relu',
    ):
        super().__init__()
        self.d_in = input_size
        self.d_u = exog_size
        self.d_out = output_size
        self.activation = get_layer_activation(activation)()
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
        """"""
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


class ConditionalEncoder(Module):
    r"""Condition an STGNN input on dynamic, static, and node features.

    Each available input is projected independently to ``output_size``. The
    projected features are broadcast over time and nodes as needed, summed, and
    transformed by a final activation. Unlike concatenation-based encoders, the
    number of input channels of one feature group does not affect the projections
    of the other groups and broadcasting is done after the projections, reducing
    the computational overhead.

    Args:
        input_size (int): Number of features in the input sequence ``x``.
        output_size (int): Number of hidden features in the output.
        exog_size (int, optional): Number of time-varying covariate features in ``u``.
            (default: :obj:`0`)
        static_size (int, optional): Number of static attribute features in ``v``.
            (default: :obj:`0`)
        emb_size (int, optional): Number of node-embedding features in ``emb``.
            (default: :obj:`0`)
        dropout (float, optional): Dropout probability after the output
            activation. (default: :obj:`0`)
        skip_connection (bool, optional): Whether to add a learned projection
            of ``x`` to the output. (default: :obj:`False`)
        activation (str, optional): Activation function. (default: :obj:`'relu'`)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        exog_size: int = 0,
        static_size: int = 0,
        emb_size: int = 0,
        dropout: float = 0.0,
        skip_connection: bool = False,
        activation: str = 'relu',
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.exog_size = exog_size
        self.static_size = static_size
        self.emb_size = emb_size
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout)

        self.input_affinity = nn.Linear(input_size, output_size)
        self.input_out = nn.Linear(output_size, output_size)
        self._add_condition_affinity('u', exog_size, output_size)
        self._add_condition_affinity('v', static_size, output_size)
        self._add_condition_affinity('emb', emb_size, output_size)

        if skip_connection:
            self.skip_conn = nn.Linear(input_size, output_size)
        else:
            self.register_module('skip_conn', None)

    def _add_condition_affinity(
        self, name: str, input_size: int, output_size: int
    ) -> None:
        """Register the projections associated with an optional feature group."""
        if input_size > 0:
            self.add_module(f'{name}_affinity', nn.Linear(input_size, output_size))
            self.add_module(
                f'{name}_out', nn.Linear(output_size, output_size, bias=False)
            )
        else:
            self.register_module(f'{name}_affinity', None)
            self.register_module(f'{name}_out', None)

    @staticmethod
    def _broadcast_feature(feature: Tensor, x: Tensor, name: str) -> Tensor:
        """Broadcast a feature group to the time and node axes of ``x``."""
        if name == 'u' and feature.dim() == 3:
            # [B, T, F] -> [B, T, 1, F]
            feature = feature.unsqueeze(2)
        elif name == 'v':
            if feature.dim() == 2:
                # [B, F] -> [B, 1, 1, F]
                feature = feature[:, None, None]
            elif feature.dim() == 3:
                # [B, N, F] -> [B, 1, N, F]
                feature = feature[:, None]
        elif name == 'emb':
            if feature.dim() == 2:
                # [N, F] -> [1, 1, N, F]
                feature = feature[None, None]
            elif feature.dim() == 3:
                # [B, N, F] -> [B, 1, N, F]
                feature = feature[:, None]
        if feature.dim() != 4:
            raise ValueError(
                f"'{name}' must be broadcastable to the [B, T, N, F] layout."
            )
        # [B|1, T|1, N|1, F] -> [B, T, N, F]
        return feature.expand(*x.shape[:-1], feature.size(-1))

    def _condition(self, x: Tensor, feature: Tensor, name: str) -> Tensor:
        """Project and align one optional feature group."""
        affinity = getattr(self, f'{name}_affinity')
        if affinity is None:
            raise ValueError(
                f"'{name}' was provided, but its configured feature size is zero."
            )
        out = self.activation(affinity(feature))
        out = getattr(self, f'{name}_out')(out)
        return self._broadcast_feature(out, x, name)

    def forward(
        self,
        x: Tensor,
        u: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode an STGNN input without concatenating feature groups.

        Args:
            x (Tensor): Input sequence.
            u (Tensor, optional): Time-varying covariates. (default: :obj:`None`)
            v (Tensor, optional): Static node attributes. (default: :obj:`None`)
            emb (Tensor, optional): Node embeddings. (default: :obj:`None`)

        Shapes:
            x: :math:`(B, T, N, F_x)`, where :math:`B` is the batch size,
                :math:`T` is the input-window length, :math:`N` is the number
                of nodes, and :math:`F_x` is ``input_size``.
            u: :math:`(B, T, N, F_u)` or :math:`(B, T, F_u)`.
            v: :math:`(B, N, F_v)` or :math:`(B, F_v)`.
            emb: :math:`(N, F_e)` or :math:`(B, N, F_e)`.
            return: :math:`(B, T, N, F_h)`, where :math:`F_h` is ``output_size``.

        Returns:
            Tensor: Encoded hidden representation.
        """
        out = self.input_out(self.activation(self.input_affinity(x)))
        for feature, name in ((u, 'u'), (v, 'v'), (emb, 'emb')):
            if feature is not None:
                out = out + self._condition(x, feature, name)
        out = self.dropout(self.activation(out))
        if self.skip_conn is not None:
            out = out + self.skip_conn(x)
        return out


class ConditionalTCNBlock(nn.Module):
    r"""Mirrors the architecture of
    :class:`tsl.nn.blocks.encoders.ConditionalBlock` but using temporal
    convolutions instead of affine transformations.

    Args:
        input_size (int): Size of the input.
        exog_size (int): Size of the exogenous variables.
        output_size (int): Size of the output.
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Spacing between kernel elements.
        dropout (float): Dropout probability.
        gated (bool): Whether to use `gated tanh` activations.
        activation (str, optional): Activation function.
        weight_norm (bool): Whether to apply weight normalization to the
            parameters of the filter.
        channel_last (bool): If :obj:`True` input data must follow the `b t n f`
            layout, assumes `b f n t` otherwise.
        skip_connection (bool): If :obj:`True` adds a parametrized skip
            connection from the input to the output.
    """

    def __init__(
        self,
        input_size,
        exog_size,
        output_size,
        kernel_size,
        dilation=1,
        dropout=0.0,
        gated=False,
        activation='relu',
        weight_norm=False,
        channel_last=True,
        skip_connection=False,
    ):
        super().__init__()

        if gated:
            # inputs module
            self.inputs_conv = nn.Sequential(
                GatedTemporalConv(
                    input_channels=input_size,
                    output_channels=output_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    weight_norm=weight_norm,
                    channel_last=channel_last,
                ),
                nn.Dropout(dropout),
            )
            self.conditions_conv = nn.Sequential(
                GatedTemporalConv(
                    input_channels=exog_size,
                    output_channels=output_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    weight_norm=weight_norm,
                    channel_last=channel_last,
                ),
                nn.Dropout(dropout),
            )
        else:
            # inputs module
            self.inputs_conv = nn.Sequential(
                TemporalConv(
                    input_channels=input_size,
                    output_channels=output_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    weight_norm=weight_norm,
                ),
                get_layer_activation(activation)(),
                nn.Dropout(dropout),
            )
            self.conditions_conv = nn.Sequential(
                TemporalConv(
                    input_channels=exog_size,
                    output_channels=output_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    weight_norm=weight_norm,
                ),
                get_layer_activation(activation)(),
                nn.Dropout(dropout),
            )
        self.out_input = nn.Linear(output_size, output_size)
        self.out_cond = nn.Linear(output_size, output_size, bias=False)
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout)

        if skip_connection:
            self.skip_conn = TemporalConv(
                input_size, output_size, 1, channel_last=channel_last
            )
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
