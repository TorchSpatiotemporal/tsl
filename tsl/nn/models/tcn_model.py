from typing import Optional

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.tcn import TemporalConvNet
from tsl.nn.layers.norm import Norm
from tsl.nn.models.base_model import BaseModel
from tsl.nn.ops.ops import Lambda
from tsl.nn.utils.utils import get_layer_activation


class TCNModel(BaseModel):
    r"""A simple Causal Dilated Temporal Convolutional Network for
    multistep forecasting. Learned temporal embeddings are pooled together
    using dynamics weights.

    Args:
        input_size (int): Input size.
        hidden_size (int): Channels in the hidden layers.
        ff_size (int): Number of units in the hidden layers of the decoder.
        output_size (int): Output channels.
        horizon (int): Forecasting horizon.
        kernel_size (int): Size of the convolutional kernel.
        n_layers (int): Number of TCN blocks.
        exog_size (int): Size of the exogenous variables.
        readout_kernel_size (int, optional): Width of the readout kernel size.
        resnet (bool, optional): Whether to use residual connections.
        dilation (int): Dilation coefficient of the convolutional kernel.
        activation (str, optional): Activation function. (default: `relu`)
        n_convs_layer (int, optional): Number of temporal convolutions in each
            layer. (default: 2)
        norm (str, optional): Normalization strategy.
        gated (bool, optional): Whether to used the GatedTanH activation
            function. (default: obj`False`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: Optional[int] = None,
                 hidden_size: int = 32,
                 ff_size: int = 32,
                 kernel_size: int = 2,
                 n_layers: int = 4,
                 n_convs_layer: int = 2,
                 readout_kernel_size: int = 1,
                 dilation: int = 2,
                 activation: str = 'relu',
                 dropout: float = 0.,
                 gated: bool = False,
                 resnet: bool = True,
                 norm: str = 'batch'):
        super(TCNModel, self).__init__()

        if exog_size > 0:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  dropout=dropout,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        layers = []
        self.receptive_field = 0
        for i in range(n_layers):
            layers.append(nn.Sequential(
                Norm(norm_type=norm, in_channels=hidden_size),
                TemporalConvNet(input_channels=hidden_size,
                                hidden_channels=hidden_size,
                                kernel_size=kernel_size,
                                dilation=dilation,
                                gated=gated,
                                activation=activation,
                                exponential_dilation=True,
                                n_layers=n_convs_layer,
                                causal_padding=True)
            )
            )
        self.convs = nn.ModuleList(layers)
        self.resnet = resnet
        activation_layer = get_layer_activation(activation=activation)

        self.readout = nn.Sequential(
            Lambda(lambda x: x[:, -readout_kernel_size:]),
            Rearrange('b t n f -> b n (f t)'),
            nn.Linear(hidden_size * readout_kernel_size, ff_size * horizon),
            activation_layer(),
            nn.Dropout(dropout),
            Rearrange('b n (f h) -> b h n f ', c=ff_size, h=horizon),
            nn.Linear(ff_size, output_size),
        )
        self.window = readout_kernel_size
        self.horizon = horizon

    def forward(self, x: Tensor, u: Optional[Tensor] = None) -> Tensor:
        # x: [b t n f]
        # u: [b t (n) f]
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b t f -> b t 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        for conv in self.convs:
            x = x + conv(x) if self.resnet else conv(x)
        return self.readout(x)
