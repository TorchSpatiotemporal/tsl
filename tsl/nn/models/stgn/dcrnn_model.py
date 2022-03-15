from tsl.nn.blocks.encoders.dcrnn import DCRNN
from tsl.utils.parser_utils import ArgParser

from einops import rearrange
from torch import nn

from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder


class DCRNNModel(nn.Module):
    r"""
    Diffusion ConvolutionalRecurrent Neural Network with a nonlinear readout.

    From Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", ICLR 2018.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the DCRNN hidden layer.
        ff_size (int): Number of units in the nonlinear readout.
        output_size (int): Number of output channels.
        n_layers (int): Number DCRNN cells.
        exog_size (int): Number of channels in the exogenous variable.
        horizon (int): Number of steps to forecast.
        activation (str, optional): Activation function in the readout.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 output_size,
                 n_layers,
                 exog_size,
                 horizon,
                 activation='relu',
                 dropout=0.,
                 kernel_size=2):
        super(DCRNNModel, self).__init__()
        if exog_size:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        self.dcrnn = DCRNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers,
                           k=kernel_size)

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=ff_size,
                                  output_size=output_size,
                                  horizon=horizon,
                                  activation=activation,
                                  dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None, u=None, **kwargs):
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s c -> b s 1 c')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)

        h, _ = self.dcrnn(x, edge_index, edge_weight)
        return self.readout(h)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True, options=[16, 32, 64, 128])
        parser.opt_list('--ff-size', type=int, default=256, tunable=True, options=[64, 128, 256, 512])
        parser.opt_list('--n-layers', type=int, default=1, tunable=True, options=[1, 2])
        parser.opt_list('--dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.25, 0.5])
        parser.opt_list('--kernel-size', type=int, default=2, tunable=True, options=[1, 2])
        return parser
