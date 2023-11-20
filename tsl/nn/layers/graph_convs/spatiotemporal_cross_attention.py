from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor

from tsl.nn.blocks.encoders import TemporalMLPAttention
from tsl.nn.functional import sparse_softmax
from tsl.nn.layers.norm import LayerNorm


class SpatiotemporalCrossAttention(MessagePassing):

    def __init__(self,
                 input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweigh: Optional[str] = None,
                 temporal_self_attention: bool = True,
                 mask_temporal: bool = True,
                 mask_spatial: bool = True,
                 norm: bool = True,
                 dropout: float = 0.,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SpatiotemporalCrossAttention, self).__init__(node_dim=-2,
                                                           **kwargs)

        # store dimensions
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size
        self.output_size = output_size
        self.msg_size = msg_size or self.output_size

        self.mask_temporal = mask_temporal
        self.mask_spatial = mask_spatial

        self.root_weight = root_weight
        self.dropout = dropout

        if temporal_self_attention:
            self.self_attention = TemporalMLPAttention(input_size=input_size,
                                                       output_size=output_size,
                                                       msg_size=msg_size,
                                                       msg_layers=msg_layers,
                                                       reweigh=reweigh,
                                                       dropout=dropout,
                                                       root_weight=False,
                                                       norm=False)
        else:
            self.register_parameter('self_attention', None)

        self.cross_attention = TemporalMLPAttention(
            input_size=input_size,
            output_size=output_size,
            msg_size=msg_size,
            msg_layers=msg_layers,
            reweigh=reweigh,
            dropout=dropout,
            root_weight=False,
            norm=False,
        )

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size,
                                   self.output_size,
                                   bias_initializer='zeros')
        else:
            self.register_parameter('lin_skip', None)

        if norm:
            self.norm = LayerNorm(output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        if self.self_attention is not None:
            self.self_attention.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(self,
                x: OptPairTensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                mask: OptTensor = None):
        # inputs: [batch, steps, nodes, channels]
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        n_src, n_tgt = x_src.size(-2), x_tgt.size(-2)

        # propagate query, key and value
        out = self.propagate(x=(x_src, x_tgt),
                             edge_index=edge_index,
                             edge_weight=edge_weight,
                             mask=mask if self.mask_spatial else None,
                             size=(n_src, n_tgt))

        if self.self_attention is not None:
            s, t = x_src.size(1), x_tgt.size(1)
            if s == t:
                attn_mask = ~torch.eye(
                    t, t, dtype=torch.bool, device=x_tgt.device)
            else:
                attn_mask = None
            temp = self.self_attention(
                x=(x_src, x_tgt),
                mask=mask if self.mask_temporal else None,
                temporal_mask=attn_mask,
            )
            out = out + temp

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor,
                mask_j: OptTensor) -> Tensor:
        # [batch, steps, edges, channels]

        out = self.cross_attention((x_j, x_i), mask=mask_j)

        if edge_weight is not None:
            out = out * edge_weight.view(-1, 1)
        return out


class HierarchicalSpatiotemporalCrossAttention(MessagePassing):

    def __init__(self,
                 h_size: int,
                 z_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweigh: Optional[str] = None,
                 update_z_cross: bool = True,
                 mask_temporal: bool = True,
                 mask_spatial: bool = True,
                 norm: bool = True,
                 dropout: float = 0.,
                 aggr: str = 'add',
                 **kwargs):
        self.spatial_aggr = aggr
        if aggr == 'softmax':
            aggr = 'add'
        super(HierarchicalSpatiotemporalCrossAttention,
              self).__init__(node_dim=-2, aggr=aggr, **kwargs)

        # store dimensions
        self.h_size = h_size
        self.z_size = z_size
        self.msg_size = msg_size or z_size

        self.mask_temporal = mask_temporal
        self.mask_spatial = mask_spatial

        self.root_weight = root_weight
        self.norm = norm
        self.dropout = dropout
        self._z_cross = None

        self.zh_self = TemporalMLPAttention(
            input_size=(h_size, z_size),
            output_size=z_size,
            msg_size=msg_size,
            msg_layers=msg_layers,
            reweigh=reweigh,
            dropout=dropout,
            root_weight=True,
            norm=True,
        )

        self.hz_self = TemporalMLPAttention(
            input_size=(z_size, h_size),
            output_size=h_size,
            msg_size=msg_size,
            msg_layers=msg_layers,
            reweigh=reweigh,
            dropout=dropout,
            root_weight=True,
            norm=False,
        )

        if update_z_cross:
            self.zh_cross = TemporalMLPAttention(
                input_size=(h_size, z_size),
                output_size=z_size,
                msg_size=msg_size,
                msg_layers=msg_layers,
                reweigh=reweigh,
                dropout=dropout,
                root_weight=True,
                norm=True,
            )
        else:
            self.register_parameter('zh_cross', None)

        self.hz_cross = TemporalMLPAttention(
            input_size=(z_size, h_size),
            output_size=h_size,
            msg_size=msg_size,
            msg_layers=msg_layers,
            reweigh=None,
            dropout=dropout,
            root_weight=True,
            norm=False,
        )

        if self.spatial_aggr == 'softmax':
            self.lin_alpha_h = Linear(h_size, 1, bias=False)
            self.lin_alpha_z = Linear(z_size, 1, bias=False)
        else:
            self.register_parameter('lin_alpha_h', None)
            self.register_parameter('lin_alpha_z', None)

        if self.root_weight:
            self.h_skip = Linear(h_size, h_size, bias_initializer='zeros')
            self.z_skip = Linear(z_size, z_size, bias_initializer='zeros')
        else:
            self.register_parameter('h_skip', None)
            self.register_parameter('z_skip', None)

        if self.norm:
            self.h_norm = LayerNorm(h_size)
            self.z_norm = LayerNorm(z_size)
        else:
            self.register_parameter('h_norm', None)
            self.register_parameter('z_norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.zh_self.reset_parameters()
        self.hz_self.reset_parameters()
        if self.zh_cross is not None:
            self.zh_cross.reset_parameters()
        self.hz_cross.reset_parameters()
        if self.spatial_aggr == 'softmax':
            self.lin_alpha_h.reset_parameters()
            self.lin_alpha_z.reset_parameters()
        if self.root_weight:
            self.h_skip.reset_parameters()
            self.z_skip.reset_parameters()
        if self.norm:
            self.h_norm.reset_parameters()
            self.z_norm.reset_parameters()

    def forward(self,
                h: Tensor,
                z: Tensor,
                edge_index: Adj,
                mask: OptTensor = None):
        # inputs: [batch, steps, nodes, channels]

        z_out = self.zh_self(x=(h, z),
                             mask=mask if self.mask_temporal else None)
        h_self = self.hz_self(x=(z_out, h))

        # propagate query, key and value
        n_src, n_tgt = h.size(-2), z.size(-2)
        h_out = self.propagate(h=h_self,
                               z=z_out,
                               edge_index=edge_index,
                               mask=mask if self.mask_spatial else None,
                               size=(n_src, n_tgt))

        if self._z_cross is not None:
            z_out = self.aggregate(self._z_cross,
                                   edge_index[1],
                                   dim_size=n_tgt)
            self._z_cross = None

        # skip connection
        if self.root_weight:
            h_out = h_out + self.h_skip(h)
            z_out = z_out + self.z_skip(z)

        if self.norm:
            h_out = self.h_norm(h_out)
            z_out = self.z_norm(z_out)

        return h_out, z_out

    def h_cross_message(self, h_i: Tensor, z_j: Tensor, index,
                        size_i) -> Tensor:
        # [batch, steps, edges, channels]
        h_cross = self.hz_cross((z_j, h_i))
        if self.spatial_aggr == 'softmax':
            alpha_h = self.lin_alpha_h(h_cross)
            alpha_h = sparse_softmax(alpha_h,
                                     index,
                                     num_nodes=size_i,
                                     dim=self.node_dim)
            h_cross = alpha_h * h_cross
        return h_cross

    def hz_cross_message(self, h_i: Tensor, h_j: Tensor, z_i: Tensor, index,
                         size_i, mask_j: OptTensor) -> Tensor:
        # [batch, steps, edges, channels]
        z_cross = self.zh_cross((h_j, z_i), mask=mask_j)
        h_cross = self.hz_cross((z_cross, h_i))
        if self.spatial_aggr == 'softmax':
            # reweigh z
            alpha_z = self.lin_alpha_z(z_cross)
            alpha_z = sparse_softmax(alpha_z,
                                     index,
                                     num_nodes=size_i,
                                     dim=self.node_dim)
            z_cross = alpha_z * z_cross
            # reweigh h
            alpha_h = self.lin_alpha_h(h_cross)
            alpha_h = sparse_softmax(alpha_h,
                                     index,
                                     num_nodes=size_i,
                                     dim=self.node_dim)
            h_cross = alpha_h * h_cross
        self._z_cross = z_cross
        return h_cross

    def message(self, h_i: Tensor, h_j: Tensor, z_i: Tensor, z_j: Tensor,
                index, size_i, mask_j: OptTensor) -> Tensor:
        if self.zh_cross is not None:
            return self.hz_cross_message(h_i, h_j, z_i, index, size_i, mask_j)
        return self.h_cross_message(h_i, z_j, index, size_i)
