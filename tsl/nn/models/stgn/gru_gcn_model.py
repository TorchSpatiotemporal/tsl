from torch import nn

from tsl.nn.base import GraphConv
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.layers import Select
from tsl.nn.models import BaseModel

from einops.layers.torch import Rearrange

from tsl.nn.blocks.encoders import RNN
from tsl.nn.utils import utils


class GRUGCNModel(BaseModel):
    r"""
    Simple time-then-space model with a GRU encoder and a GCN decoder.

    From Guo et al., "On the Equivalence Between Temporal and Static Equivariant Graph Representations", ICML 2022.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of hidden units in each hidden layer.
        output_size (int): Size of the output.
        horizon (int): Forecasting steps.
        exog_size (int): Size of the optional exogenous variables.
        enc_layers (int): Number of layers in the GRU encoder.
        gcn_layers (int): Number of GCN layers in GCN decoder.
        asymmetric_norm (bool): Whether to use asymmetric or GCN normalization.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 horizon,
                 exog_size,
                 enc_layers,
                 gcn_layers,
                 asymmetric_norm,
                 encode_edges=False,
                 activation='softplus'):
        super(GRUGCNModel, self).__init__()

        input_size += exog_size
        self.input_encoder = RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 n_layers=enc_layers,
                                 cell='gru')

        if encode_edges:
            self.edge_encoder = nn.Sequential(
                RNN(input_size=input_size,
                    hidden_size=hidden_size,
                    n_layers=enc_layers,
                    cell='gru'),
                Select(1, -1),
                nn.Linear(hidden_size, 1),
                nn.Softplus(),
                Rearrange('e f -> (e f)', f=1)
            )
        else:
            self.register_parameter('edge_encoder', None)

        self.gcn_layers = nn.ModuleList(
            [
                GraphConv(hidden_size,
                          hidden_size,
                          root_weight=False,
                          asymmetric_norm=asymmetric_norm,
                          activation=activation) for _ in range(gcn_layers)
            ]
        )

        self.skip_con = nn.Linear(hidden_size, hidden_size)

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  output_size=output_size,
                                  activation=activation,
                                  horizon=horizon)

    def forward(self, x, edge_index, edge_weight=None, edge_features=None, u=None):
        """"""
        # x: [batches steps nodes features]
        x = utils.maybe_cat_exog(x, u)

        # flat time dimension
        x = self.input_encoder(x, return_last_state=True)
        if self.edge_encoder is not None:
            assert edge_weight is None
            edge_weight = self.edge_encoder(edge_features)

        out = x
        for layer in self.gcn_layers:
            out = layer(out, edge_index, edge_weight)

        out = out + self.skip_con(x)

        return self.readout(out)