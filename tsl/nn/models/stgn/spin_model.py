from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.nn import inits
from torch_geometric.typing import OptTensor

from tsl.nn.blocks.encoders import MLP
from tsl.nn.layers.base import NodeEmbedding, PositionalEncoding
from tsl.nn.layers.graph_convs import (
    HierarchicalSpatiotemporalCrossAttention, SpatiotemporalCrossAttention)
from tsl.nn.models.base_model import BaseModel


class SPINPositionalEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None):
        super(SPINPositionalEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels,
                       out_channels,
                       out_channels,
                       n_layers=n_layers,
                       activation='relu')
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = NodeEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter('node_emb', None)

    def forward(self,
                x: Tensor,
                node_emb: OptTensor = None,
                node_index: OptTensor = None) -> Tensor:
        if node_emb is None:
            node_emb = self.node_emb(node_index=node_index)
        # x: [b s c], node_emb: [n c] -> [b s n c]
        x = self.lin(x)
        x = self.activation(x.unsqueeze(-2) + node_emb)
        out = self.mlp(x)
        out = self.positional(out)
        return out


class SPINModel(BaseModel):
    r"""The Spatiotemporal Point Inference Network (SPIN) from the paper
    `"Learning to Reconstruct Missing Data from Spatiotemporal Graphs with
    Sparse Observations" <https://arxiv.org/abs/2205.13479>`_ (Marisca et al.,
    NeurIPS 2022).
    """

    return_type = tuple

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 exog_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 temporal_self_attention: bool = True,
                 reweigh: Optional[str] = 'softmax',
                 n_layers: int = 4,
                 eta: int = 3,
                 message_layers: int = 1):
        super(SPINModel, self).__init__()

        exog_size = exog_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        self.u_enc = SPINPositionalEncoder(in_channels=exog_size,
                                           out_channels=hidden_size,
                                           n_layers=2,
                                           n_nodes=n_nodes)

        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)

        self.valid_emb = NodeEmbedding(n_nodes, hidden_size)
        self.mask_emb = NodeEmbedding(n_nodes, hidden_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for layer in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            encoder = SpatiotemporalCrossAttention(
                input_size=hidden_size,
                output_size=hidden_size,
                msg_size=hidden_size,
                msg_layers=message_layers,
                temporal_self_attention=temporal_self_attention,
                reweigh=reweigh,
                mask_temporal=True,
                mask_spatial=layer < eta,
                norm=True,
                root_weight=True,
                dropout=0.0)
            readout = MLP(hidden_size, hidden_size, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def forward(self,
                x: Tensor,
                u: Tensor,
                mask: Tensor,
                edge_index: Tensor,
                node_index: OptTensor = None,
                target_nodes: OptTensor = None) -> Tuple[Tensor, List[Tensor]]:
        """"""
        if target_nodes is None:
            target_nodes = slice(None)

        # Whiten missing values
        x = x * mask

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        imputations = []

        for layer in range(self.n_layers):
            if layer == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(node_index=node_index)
                masked = self.mask_emb(node_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[layer](x) * mask  # skip connection for valid x
            h = self.encoder[layer](h, edge_index, mask=mask)
            # Read from H to get imputations
            target_readout = self.readout[layer](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)

        return x_hat, imputations

    def predict(self,
                x: Tensor,
                u: Tensor,
                mask: Tensor,
                edge_index: Tensor,
                node_index: OptTensor = None,
                target_nodes: OptTensor = None) -> Tensor:
        """"""
        imputation = self.forward(x=x,
                                  u=u,
                                  mask=mask,
                                  edge_index=edge_index,
                                  node_index=node_index,
                                  target_nodes=target_nodes)[0]
        return imputation


class SPINHierarchicalModel(BaseModel):
    r"""The Hierarchical Spatiotemporal Point Inference Network (SPIN-H) from
    the paper `"Learning to Reconstruct Missing Data from Spatiotemporal Graphs
    with Sparse Observations" <https://arxiv.org/abs/2205.13479>`_
    (Marisca et al., NeurIPS 2022).
    """
    return_type = tuple

    def __init__(self,
                 input_size: int,
                 h_size: int,
                 z_size: int,
                 n_nodes: int,
                 z_heads: int = 1,
                 exog_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 n_layers: int = 5,
                 eta: int = 3,
                 message_layers: int = 1,
                 reweigh: Optional[str] = 'softmax',
                 update_z_cross: bool = True,
                 norm: bool = True,
                 spatial_aggr: str = 'add'):
        super(SPINHierarchicalModel, self).__init__()

        exog_size = exog_size or input_size
        output_size = output_size or input_size
        self.h_size = h_size
        self.z_size = z_size

        self.n_nodes = n_nodes
        self.z_heads = z_heads
        self.n_layers = n_layers
        self.eta = eta

        self.v = NodeEmbedding(n_nodes, h_size)
        self.lin_v = nn.Linear(h_size, z_size, bias=False)
        self.z = nn.Parameter(torch.Tensor(1, z_heads, n_nodes, z_size))
        inits.uniform(z_size, self.z)
        self.z_norm = LayerNorm(z_size)

        self.u_enc = SPINPositionalEncoder(in_channels=exog_size,
                                           out_channels=h_size,
                                           n_layers=2)

        self.h_enc = MLP(input_size, h_size, n_layers=2)
        self.h_norm = LayerNorm(h_size)

        self.v1 = NodeEmbedding(n_nodes, h_size)
        self.m1 = NodeEmbedding(n_nodes, h_size)

        self.v2 = NodeEmbedding(n_nodes, h_size)
        self.m2 = NodeEmbedding(n_nodes, h_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for layer in range(n_layers):
            x_skip = nn.Linear(input_size, h_size)
            encoder = HierarchicalSpatiotemporalCrossAttention(
                h_size=h_size,
                z_size=z_size,
                msg_size=h_size,
                msg_layers=message_layers,
                reweigh=reweigh,
                mask_temporal=True,
                mask_spatial=layer < eta,
                update_z_cross=update_z_cross,
                norm=norm,
                root_weight=True,
                aggr=spatial_aggr,
                dropout=0.0)
            readout = MLP(h_size, z_size, output_size, n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def forward(self,
                x: Tensor,
                u: Tensor,
                mask: Tensor,
                edge_index: Tensor,
                node_index: OptTensor = None,
                target_nodes: OptTensor = None) -> Tuple[Tensor, List[Tensor]]:
        """"""
        if target_nodes is None:
            target_nodes = slice(None)
        if node_index is None:
            node_index = slice(None)

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #
        # Condition also embeddings Z on V.                                   #

        v_nodes = self.v(node_index=node_index)
        z = self.z[..., node_index, :] + self.lin_v(v_nodes)

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index, node_emb=v_nodes)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q. Then, condition H on two
        # different embeddings to distinguish valid values from masked ones.
        h = torch.where(mask.bool(), h + self.v1(), q + self.m1())
        # Normalize features
        h, z = self.h_norm(h), self.z_norm(z)

        imputations = []

        for layer in range(self.n_layers):
            if layer == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                h = torch.where(mask.bool(), h + self.v2(), h + self.m2())
            # Skip connection from input x
            h = h + self.x_skip[layer](x) * mask
            # Masked Temporal GAT for encoding representation
            h, z = self.encoder[layer](h, z, edge_index, mask=mask)
            target_readout = self.readout[layer](h[..., target_nodes, :])
            imputations.append(target_readout)

        x_hat = imputations.pop(-1)

        return x_hat, imputations

    def predict(self,
                x: Tensor,
                u: Tensor,
                mask: Tensor,
                edge_index: Tensor,
                node_index: OptTensor = None,
                target_nodes: OptTensor = None) -> Tensor:
        """"""
        imputation = self.forward(x=x,
                                  u=u,
                                  mask=mask,
                                  edge_index=edge_index,
                                  node_index=node_index,
                                  target_nodes=target_nodes)[0]
        return imputation
