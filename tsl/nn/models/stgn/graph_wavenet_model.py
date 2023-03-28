from typing import Optional

import torch
from einops import repeat
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.typing import Adj, OptTensor

from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks.encoders import TemporalConvNet
from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.layers.graph_convs import DenseGraphConvOrderK, DiffConv
from tsl.nn.layers.norm import Norm
from tsl.nn.models.base_model import BaseModel


class GraphWaveNetModel(BaseModel):
    r"""The Graph WaveNet model from the paper `"Graph WaveNet for Deep
    Spatial-Temporal Graph Modeling" <https://arxiv.org/abs/1906.00121>`_
    (Wu et al., IJCAI 2019).

    Args:
        input_size (int): Number of features of the input sample.
        output_size (int): Number of output channels.
        horizon (int): Number of future time steps to forecast.
        exog_size (int): Number of features of the input covariate,
            if any. (default: :obj:`0`)
        hidden_size (int): Number of hidden units.
            (default: :obj:`32`)
        ff_size (int): Number of units in the nonlinear readout.
            (default: :obj:`256`)
        n_layers (int): Number of Graph WaveNet blocks.
            (default: :obj:`8`)
        temporal_kernel_size (int): Size of the temporal convolution kernel.
            (default: :obj:`2`)
        spatial_kernel_size (int): Order of the spatial diffusion process.
            (default: :obj:`2`)
        learned_adjacency (bool):  If :obj:`True`, then consider an additional
            learned adjacency matrix.
            (default: :obj:`True`)
        n_nodes (int, optional): Number of nodes in the input graph, required
            only when :attr:`learned_adjacency` is :obj:`True`.
            (default: :obj:`None`)
        emb_size (int): Number of features in the node embeddings used for
            graph learning.
            (default: :obj:`10`)
        dilation (int): Dilation of the temporal convolutional kernels.
            (default: :obj:`2`)
        dilation_mod (int): Length of the cycle for the dilation coefficient.
            (default: :obj:`2`)
        norm (str): Normalization strategy.
            (default: :obj:`'batch'`)
        dropout (float): Dropout probability.
            (default: :obj:`0.3`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 256,
                 n_layers: int = 8,
                 temporal_kernel_size: int = 2,
                 spatial_kernel_size: int = 2,
                 learned_adjacency: bool = True,
                 n_nodes: Optional[int] = None,
                 emb_size: int = 10,
                 dilation: int = 2,
                 dilation_mod: int = 2,
                 norm: str = 'batch',
                 dropout: float = 0.3):
        super(GraphWaveNetModel, self).__init__(return_type=Tensor)

        if learned_adjacency:
            assert n_nodes is not None
            self.source_embeddings = NodeEmbedding(n_nodes, emb_size)
            self.target_embeddings = NodeEmbedding(n_nodes, emb_size)
        else:
            self.register_parameter('source_embedding', None)
            self.register_parameter('target_embedding', None)

        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)

        temporal_conv_blocks = []
        spatial_convs = []
        skip_connections = []
        norms = []
        receptive_field = 1
        for i in range(n_layers):
            d = dilation**(i % dilation_mod)
            temporal_conv_blocks.append(
                TemporalConvNet(input_channels=hidden_size,
                                hidden_channels=hidden_size,
                                kernel_size=temporal_kernel_size,
                                dilation=d,
                                exponential_dilation=False,
                                n_layers=1,
                                causal_padding=False,
                                gated=True))

            spatial_convs.append(
                DiffConv(in_channels=hidden_size,
                         out_channels=hidden_size,
                         k=spatial_kernel_size))

            skip_connections.append(nn.Linear(hidden_size, ff_size))
            norms.append(Norm(norm, hidden_size))
            receptive_field += d * (temporal_kernel_size - 1)
        self.tconvs = nn.ModuleList(temporal_conv_blocks)
        self.sconvs = nn.ModuleList(spatial_convs)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

        self.receptive_field = receptive_field

        dense_sconvs = []
        if learned_adjacency:
            for _ in range(n_layers):
                dense_sconvs.append(
                    DenseGraphConvOrderK(input_size=hidden_size,
                                         output_size=hidden_size,
                                         support_len=1,
                                         order=spatial_kernel_size,
                                         include_self=False,
                                         channel_last=True))
        self.dense_sconvs = nn.ModuleList(dense_sconvs)
        self.readout = nn.Sequential(
            nn.ReLU(),
            MLPDecoder(input_size=ff_size,
                       hidden_size=2 * ff_size,
                       output_size=output_size,
                       horizon=horizon,
                       activation='relu'))

    def get_learned_adj(self):
        logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
        adj = torch.softmax(logits, dim=1)
        return adj

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None) -> Tensor:
        """"""
        # x: [b t n f]

        if u is not None:
            if u.dim() == 3:
                u = repeat(u, 'b t f -> b t n f', n=x.size(-2))
            x = torch.cat([x, u], -1)

        if self.receptive_field > x.size(1):
            # pad temporal dimension
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))

        if len(self.dense_sconvs):
            adj_z = self.get_learned_adj()

        x = self.input_encoder(x)

        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for i, (tconv, sconv, skip_conn, norm) in enumerate(
                zip(self.tconvs, self.sconvs, self.skip_connections,
                    self.norms)):
            res = x
            # temporal conv
            x = tconv(x)
            # residual connection -> out
            out = skip_conn(x) + out[:, -x.size(1):]
            # spatial conv
            xs = sconv(x, edge_index, edge_weight)
            if len(self.dense_sconvs):
                x = xs + self.dense_sconvs[i](x, adj_z)
            else:
                x = xs
            x = self.dropout(x)
            # residual connection -> next layer
            x = x + res[:, -x.size(1):]
            x = norm(x)

        return self.readout(out)
