from torch import nn

from .conditional import ConditionalBlock
from .mlp import MLP
from .recurrent import RNN
from .tcn import TemporalConvNet


class InputEncoder(nn.Module):

    def __init__(self,
                 enc_type,
                 input_size,
                 exog_size,
                 output_size,
                 dropout=0.,
                 activation=None,
                 **kwargs):
        super(InputEncoder, self).__init__()
        if enc_type == 'mlp':
            self.input_encoder = MLP(input_size=input_size,
                                     exog_size=exog_size,
                                     hidden_size=output_size,
                                     activation=activation,
                                     dropout=dropout,
                                     **kwargs)
        elif enc_type == 'conditional':
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=output_size,
                                                  dropout=dropout,
                                                  activation=activation,
                                                  **kwargs)
        elif enc_type == 'rnn':
            assert activation is None
            self.input_encoder = RNN(input_size=input_size,
                                     exog_size=exog_size,
                                     output_size=output_size,
                                     dropout=dropout,
                                     **kwargs)
        elif enc_type == 'tcn':
            self.input_encoder = TemporalConvNet(input_channels=input_size,
                                                 exog_channels=exog_size,
                                                 output_channels=output_size,
                                                 activation=activation,
                                                 dropout=dropout,
                                                 **kwargs)
        else:
            raise NotImplementedError(
                f"Encoder type {enc_type} not implemented.")

    def forward(self, x, u=None):
        """"""
        return self.input_encoder(x, u)
