from einops.layers.torch import Rearrange
from torch import nn

from tsl.nn import utils
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers.graph_convs import GraphConv
from tsl.nn.models.base_model import BaseModel


class GRUGCNModel(BaseModel):
    r"""Simple time-then-space model with a GRU encoder and a GCN decoder from
    the paper `"On the Equivalence Between Temporal and Static Equivariant
    Graph Representations" <https://arxiv.org/abs/2103.07016>`_ (Guo et al.,
    ICML 2022).

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of hidden units in each hidden layer.
        output_size (int): Size of the output.
        horizon (int): Forecasting steps.
        exog_size (int): Size of the optional exogenous variables.
        enc_layers (int): Number of layers in the GRU encoder.
        gcn_layers (int): Number of GCN layers in GCN decoder.
        norm (str): Normalization used by the graph convolutional layers.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 horizon,
                 exog_size,
                 enc_layers,
                 gcn_layers,
                 norm='mean',
                 encode_edges=False,
                 activation='softplus'):
        super(GRUGCNModel, self).__init__()

        input_size += exog_size
        self.input_encoder = RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 n_layers=enc_layers,
                                 return_only_last_state=True,
                                 cell='gru')

        if encode_edges:
            self.edge_encoder = nn.Sequential(
                RNN(input_size=input_size,
                    hidden_size=hidden_size,
                    n_layers=enc_layers,
                    return_only_last_state=True,
                    cell='gru'),
                nn.Linear(hidden_size, 1),
                nn.Softplus(),
                Rearrange('e f -> (e f)', f=1),
            )
        else:
            self.register_parameter('edge_encoder', None)

        self.gcn_layers = nn.ModuleList([
            GraphConv(hidden_size,
                      hidden_size,
                      root_weight=False,
                      norm=norm,
                      activation=activation) for _ in range(gcn_layers)
        ])

        self.skip_con = nn.Linear(hidden_size, hidden_size)

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size,
                                  activation=activation,
                                  horizon=horizon)

    def forward(self,
                x,
                edge_index,
                edge_weight=None,
                edge_features=None,
                u=None):
        """"""
        # x: [batches steps nodes features]
        x = utils.maybe_cat_exog(x, u)

        # flat time dimension
        x = self.input_encoder(x)
        if self.edge_encoder is not None:
            assert edge_weight is None
            edge_weight = self.edge_encoder(edge_features)

        out = x
        for layer in self.gcn_layers:
            out = layer(out, edge_index, edge_weight)

        out = out + self.skip_con(x)

        return self.readout(out)
