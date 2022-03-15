from torch import nn
from einops import rearrange

from tsl.utils.parser_utils import ArgParser

from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.rnn import RNN

from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder


class RNNModel(nn.Module):
    r"""
    Simple RNN for multi-step forecasting.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the recurrent cell.
        output_size (int): Number of output channels.
        ff_size (int): Number of units in the link predictor.
        exog_size (int): Size of the exogenous variables.
        rec_layers (int): Number of RNN layers.
        ff_layers (int): Number of hidden layers in the decoder.
        rec_dropout (float, optional): Dropout probability in the RNN encoder.
        ff_dropout (float, optional): Dropout probability int the GCN decoder.
        horizon (int): Forecasting horizon.
        cell_type (str, optional): Type of cell that should be use (options: [`gru`, `lstm`]). (default: `gru`)
        activation (str, optional): Activation function.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 ff_size,
                 exog_size,
                 rec_layers,
                 ff_layers,
                 rec_dropout,
                 ff_dropout,
                 horizon,
                 cell_type='gru',
                 activation='relu'):
        super(RNNModel, self).__init__()

        if exog_size > 0:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            )

        self.rnn = RNN(input_size=hidden_size,
                       hidden_size=hidden_size,
                       n_layers=rec_layers,
                       dropout=rec_dropout,
                       cell=cell_type)

        self.readout = MLPDecoder(
            input_size=hidden_size,
            hidden_size=ff_size,
            output_size=output_size,
            horizon=horizon,
            n_layers=ff_layers,
            activation=activation,
            dropout=ff_dropout
        )

    def forward(self, x, u=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s f -> b s 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)

        x = self.rnn(x, return_last_state=True)

        return self.readout(x)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True, options=[16, 32, 64, 128, 256])
        parser.opt_list('--ff-size', type=int, default=64, tunable=True, options=[32, 64, 128, 256, 512, 1024])
        parser.opt_list('--rec-layers', type=int, default=1, tunable=True, options=[1, 2, 3])
        parser.opt_list('--ff-layers', type=int, default=1, tunable=True, options=[1, 2, 3])
        parser.opt_list('--rec-dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.2])
        parser.opt_list('--ff-dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.25, 0.5])
        parser.opt_list('--cell-type', type=str, default='gru', tunable=True, options=['gru', 'lstm'])
        return parser


class FCRNNModel(RNNModel):
    r"""
    A simple fully connected RNN for multi-step forecasting that simply flattens data along the spatial diemnesion.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the recurrent cell.
        output_size (int): Number of output channels.
        ff_size (int): Number of units in the link predictor.
        exog_size (int): Size of the exogenous variables.
        rec_layers (int): Number of RNN layers.
        ff_layers (int): Number of hidden layers in the decoder.
        rec_dropout (float, optional): Dropout probability in the RNN encoder.
        ff_dropout (float, optional): Dropout probability int the GCN decoder.
        horizon (int): Forecasting horizon.
        cell_type (str, optional): Type of cell that should be use (options: [`gru`, `lstm`]). (default: `gru`)
        activation (str, optional): Activation function.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 ff_size,
                 exog_size,
                 rec_layers,
                 ff_layers,
                 rec_dropout,
                 ff_dropout,
                 horizon,
                 n_nodes,
                 cell_type='gru',
                 activation='relu'):
        super(FCRNNModel, self).__init__(input_size=input_size * n_nodes,
                                         hidden_size=hidden_size,
                                         output_size=output_size * n_nodes,
                                         ff_size=ff_size,
                                         exog_size=exog_size,
                                         rec_layers=rec_layers,
                                         ff_layers=ff_layers,
                                         rec_dropout=rec_dropout,
                                         ff_dropout=ff_dropout,
                                         horizon=horizon,
                                         cell_type=cell_type,
                                         activation=activation)

    def forward(self, x, u=None, **kwargs):
        """"""
        # x: [batches, steps, nodes, features]
        # u: [batches, steps, (nodes), features]
        b, _, n, _ = x.size()
        x = rearrange(x, 'b s n f -> b s 1 (n f)')
        if u is not None and u.dim() == 4:
            u = rearrange(u, 'b s n f -> b s 1 (n f)')
        x = super(FCRNNModel, self).forward(x, u, **kwargs)
        # [b, h, 1, (n f)]
        return rearrange(x, 'b h 1 (n f) -> b h n f', n=n)
