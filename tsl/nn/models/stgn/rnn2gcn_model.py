from torch import nn
from tsl.utils.parser_utils import ArgParser

from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.decoders.gcn_decoder import GCNDecoder
from tsl.nn.blocks.encoders.rnn import RNN

from einops import rearrange


class RNNEncGCNDecModel(nn.Module):
    """
    Simple time-then-space model.

    Input time series are encoded in vectors using an RNN and then decoded using a stack of GCN layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int): Size of the optional readout.
        exog_size (int): Size of the exogenous variables.
        rnn_layers (int): Number of recurrent layers in the encoder.
        gcn_layers (int): Number of graph convolutional layers in the decoder.
        rnn_dropout (float, optional): Dropout probability in the RNN encoder.
        gcn_dropout (float, optional): Dropout probability int the GCN decoder.
        horizon (int): Forecasting horizon.
        cell_type (str, optional): Type of cell that should be use (options: [`gru`, `lstm`]). (default: `gru`)
        activation (str, optional): Activation function.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 exog_size,
                 rnn_layers,
                 gcn_layers,
                 rnn_dropout,
                 gcn_dropout,
                 horizon,
                 cell_type='gru',
                 activation='relu'):
        super(RNNEncGCNDecModel, self).__init__()

        if exog_size > 0:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
            )

        self.encoder = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           dropout=rnn_dropout,
                           cell=cell_type)

        self.decoder = GCNDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            horizon=horizon,
            n_layers=gcn_layers,
            activation=activation,
            dropout=gcn_dropout
        )

    def forward(self, x, edge_index, edge_weight, u=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s f -> b s 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)

        x = self.encoder(x, return_last_state=True)

        return self.decoder(x, edge_index, edge_weight)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True, options=[16, 32, 64, 128, 256])
        parser.opt_list('--rnn-layers', type=int, default=1, tunable=True, options=[1, 2, 3])
        parser.opt_list('--gcn-layers', type=int, default=1, tunable=True, options=[1, 2, 3])
        parser.opt_list('--rnn-dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.2])
        parser.opt_list('--gcn-dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.25, 0.5])
        parser.opt_list('--cell-type', type=str, default='gru', tunable=True, options=['gru', 'lstm'])
        return parser
