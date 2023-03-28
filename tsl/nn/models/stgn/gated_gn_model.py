import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.layers.graph_convs import GatedGraphNetwork
from tsl.nn.models import BaseModel
from tsl.nn.utils import get_layer_activation, maybe_cat_exog


class GatedGraphNetworkModel(BaseModel):
    r"""Simple time-then-space model with an MLP with residual connections as
    encoder (flattened time dimension) and a gated GN decoder with node
    identification.

    Inspired by the FC-GNN model from the paper `"Multivariate Time Series
    Forecasting with Latent Graph Inference"
    <https://arxiv.org/abs/2203.03423>`_ (Satorras et al., 2022).

    Args:
        input_size (int): Size of the input.
        input_window_size (int): Size of the input window (this model cannot
            process sequences of variable lenght).
        hidden_size (int): Number of hidden units in each hidden layer.
        output_size (int): Size of the output.
        horizon (int): Forecasting steps.
        n_nodes (int): Number of nodes.
        exog_size (int): Size of the optional exogenous variables.
        enc_layers (int): Number of layers in the MLP encoder.
        gnn_layers (int): Number of GNN layers in the decoder.
        full_graph (int): Whether to use a full graph for the GNN. In that case,
            the model turns into a dense spatial attention layer.
    """

    def __init__(self,
                 input_size: int,
                 input_window_size: int,
                 horizon: int,
                 n_nodes: int,
                 hidden_size: int,
                 output_size: int = None,
                 exog_size: int = 0,
                 enc_layers: int = 1,
                 gnn_layers: int = 1,
                 full_graph: bool = True,
                 activation: str = 'silu'):
        super(GatedGraphNetworkModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size or input_size
        self.input_window_size = input_window_size
        self.full_graph = full_graph

        input_size += exog_size
        self.input_encoder = nn.Sequential(
            nn.Linear(input_size * input_window_size, hidden_size), )

        self.encoder_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size),
                          get_layer_activation(activation)(),
                          nn.Linear(hidden_size, hidden_size))
            for _ in range(enc_layers)
        ])

        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_size)

        self.gcn_layers = nn.ModuleList([
            GatedGraphNetwork(hidden_size, hidden_size, activation=activation)
            for _ in range(gnn_layers)
        ])

        self.decoder = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     get_layer_activation(activation)())

        self.readout = nn.Sequential(
            nn.Linear(hidden_size, horizon * self.output_size),
            Rearrange('b n (h f) -> b h n f', h=horizon, f=self.output_size))

    def forward(self, x, edge_index=None, u=None):
        """"""
        # x: [batches steps nodes features]
        x = maybe_cat_exog(x, u)

        if self.full_graph or edge_index is None:
            num_nodes = x.size(-2)
            nodes = torch.arange(num_nodes, device=x.device)
            edge_index = torch.cartesian_prod(nodes, nodes).T

        # flat time dimension
        x = rearrange(x[:, -self.input_window_size:], 'b s n f -> b n (s f)')

        x = self.input_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x) + x
        # add encoding
        x = self.emb() + x
        for layer in self.gcn_layers:
            x = layer(x, edge_index)

        x = self.decoder(x) + x

        return self.readout(x)
