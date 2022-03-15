from torch import nn

from tsl.nn.blocks.encoders.tcn import TemporalConvNet
from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.layers.norm import Norm

from tsl.utils.parser_utils import str_to_bool, ArgParser
from tsl.nn.utils.utils import get_layer_activation
from tsl.nn.ops.ops import Lambda

from einops import rearrange
from einops.layers.torch import Rearrange


class TCNModel(nn.Module):
    r"""
    A simple Causal Dilated Temporal Convolutional Network for multi-step forecasting.
    Learned temporal embeddings are pooled together using dynamics weights.

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
        n_convs_layer (int, optional): Number of temporal convolutions in each layer. (default: 2)
        norm (str, optional): Normalization strategy.
        gated (bool, optional): Whether to used the GatedTanH activation function. (default: `False`)
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 output_size,
                 horizon,
                 kernel_size,
                 n_layers,
                 exog_size,
                 readout_kernel_size=1,
                 resnet=True,
                 dilation=1,
                 activation='relu',
                 n_convs_layer=2,
                 dropout=0.,
                 norm="none",
                 gated=False):
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
            Rearrange('b s n c -> b n (c s)'),
            nn.Linear(hidden_size * readout_kernel_size, ff_size * horizon),
            activation_layer(),
            nn.Dropout(dropout),
            Rearrange('b n (c h) -> b h n c ', c=ff_size, h=horizon),
            nn.Linear(ff_size, output_size),
        )
        self.window = readout_kernel_size
        self.horizon = horizon

    def forward(self, x, u=None, **kwargs):
        """"""
        # x: [b s n c]
        # u: [b s (n) c]
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s f -> b s 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        for conv in self.convs:
            x = x + conv(x) if self.resnet else conv(x)
        return self.readout(x)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True, options=[32])
        parser.opt_list('--ff-size', type=int, default=32, tunable=True, options=[256])
        parser.opt_list('--kernel-size', type=int, default=2, tunable=True, options=[2, 3])
        parser.opt_list('--n-layers', type=int, default=4, tunable=True, options=[2, 4, 6])
        parser.opt_list('--n-convs-layer', type=int, default=2, tunable=True, options=[1, 2])
        parser.opt_list('--dilation', type=int, default=2, tunable=True, options=[1, 2])
        parser.opt_list('--dropout', type=float, default=0., tunable=True, options=[0., 0.2])
        parser.opt_list('--gated', type=str_to_bool, tunable=False, nargs='?', const=True, default=False, options=[True, False])
        parser.opt_list('--resnet', type=str_to_bool, tunable=False, nargs='?', const=True, default=True, options=[True, False])
        parser.opt_list('--norm', type=str, default="batch", options=["none", "batch", "instance", "layer"])
        return parser
