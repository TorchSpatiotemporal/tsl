from tsl.nn.blocks.encoders.nri_dcrnn import NeuRelInfDCRNN
from tsl.utils.parser_utils import ArgParser

from einops import rearrange
from torch import nn

from tsl.nn.blocks.encoders import ConditionalBlock
from tsl.nn.blocks.decoders.mlp_decoder import MLPDecoder


class NRIModel(nn.Module):
    """
    Simple model performing graph learning with a binary sampler and a DCRNN backbone.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the recurrent cell.
        ff_size (int): Number of units in the link predictor.
        emb_size (int): Number of features for the node embeddings.
        output_size (int): Number of output channels.
        n_layers (int): Number of DCRNN layers.
        exog_size (int): Size of the exogenous variables.
        horizon (int): Number of forecasting steps.
        n_nodes (int): Number of nodes in the input graph.
        sampler_tau (float, optional): Temperature of the binary sampler.
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
        kernel_size (int, optional): Order of the spatial diffusion process.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size,
                 emb_size,
                 output_size,
                 n_layers,
                 exog_size,
                 horizon,
                 n_nodes,
                 sampler_tau=0.25,
                 activation='relu',
                 dropout=0.,
                 kernel_size=2):
        super(NRIModel, self).__init__()
        self.tau = sampler_tau
        if exog_size:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        self.nri_dcrnn = NeuRelInfDCRNN(input_size=hidden_size,
                                        hidden_size=hidden_size,
                                        n_layers=n_layers,
                                        n_nodes=n_nodes,
                                        emb_size=emb_size,
                                        k=kernel_size)

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=ff_size,
                                  output_size=output_size,
                                  horizon=horizon,
                                  activation=activation,
                                  dropout=dropout)

    def forward(self, x, u=None, **kwargs):
        """"""
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s c -> b s 1 c')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)

        h, _ = self.nri_dcrnn(x, tau=self.tau)
        return self.readout(h)

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True, options=[16, 32, 64, 128])
        parser.opt_list('--ff-size', type=int, default=256, tunable=True, options=[64, 128, 256, 512])
        parser.opt_list('--emb-size', type=int, default=10, tunable=True, options=[8, 10, 16, 32, 64])
        parser.opt_list('--sampler-tau', type=float, default=0.25, tunable=True, options=[0.1, 0.25, 0.5, 0.75, 1])
        parser.opt_list('--n-layers', type=int, default=1, tunable=True, options=[1, 2])
        parser.opt_list('--dropout', type=float, default=0., tunable=True, options=[0., 0.1, 0.25, 0.5])
        parser.opt_list('--kernel-size', type=int, default=2, tunable=True, options=[1, 2])
        return parser
