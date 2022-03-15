from torch import nn
from einops import rearrange

from tsl.utils.parser_utils import ArgParser

from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import Transformer
from tsl.nn.ops.ops import Select
from tsl.nn.layers.positional_encoding import PositionalEncoding

from einops.layers.torch import Rearrange


class TransformerModel(nn.Module):
    r"""
    Simple Transformer for multi-step time series forecasting.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        output_size (int): Dimension of the output.
        ff_size (int): Units in the MLP after self attention.
        exog_size (int): Dimension of the exogenous variables.
        horizon (int): Number of forecasting steps.
        n_heads (int, optional): Number of parallel attention heads.
        n_layers (int, optional): Number of layers.
        dropout (float, optional): Dropout probability.
        axis (str, optional): Dimension on which to apply attention to update the representations.
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 ff_size,
                 exog_size,
                 horizon,
                 n_heads,
                 n_layers,
                 dropout,
                 axis,
                 activation='elu'):
        super(TransformerModel, self).__init__()

        if exog_size > 0:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        self.pe = PositionalEncoding(hidden_size, max_len=100)

        self.transformer_encoder = nn.Sequential(
            Transformer(input_size=hidden_size,
                        hidden_size=hidden_size,
                        ff_size=ff_size,
                        n_heads=n_heads,
                        n_layers=n_layers,
                        activation=activation,
                        dropout=dropout,
                        axis=axis),
            Select(1, -1)
        )

        self.readout = nn.Sequential(
            MLP(input_size=hidden_size,
                hidden_size=ff_size,
                output_size=output_size * horizon,
                dropout=dropout),
            Rearrange('b n (h c) -> b h n c', c=output_size, h=horizon)
        )

    def forward(self, x, u=None, **kwargs):
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        b, *_ = x.size()
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s f -> b s 1 f')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        x = self.pe(x)
        x = self.transformer_encoder(x)

        return self.readout(x)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True, options=[16, 32, 64, 128, 256])
        parser.opt_list('--ff-size', type=int, default=32, tunable=True, options=[32, 64, 128, 256, 512, 1024])
        parser.opt_list('--n-layers', type=int, default=1, tunable=True, options=[1, 2, 3])
        parser.opt_list('--n-heads', type=int, default=1, tunable=True, options=[1, 2, 3])
        parser.opt_list('--dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.25, 0.5])
        parser.opt_list('--axis', type=str, default='steps', tunable=True, options=['steps', 'both'])
        return parser
