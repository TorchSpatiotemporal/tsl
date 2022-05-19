import torch.nn as nn
from torch_geometric.typing import OptTensor
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch
import math
from einops import rearrange
from functools import partial
from typing import Optional
from tsl.nn.models.transformer_model import TransformerModel
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.base.centrality_encoding import CentralityEncoding
from tsl.nn.ops.ops import Select
from tsl.nn.blocks.encoders import TransformerLayer, SpatioTemporalTransformerLayer, Transformer
import numpy as np
from tsl.nn.base.attention.graphormer_attention import multi_head_attention_forward
from typing import Tuple
from torch import Tensor

@torch.jit.script
def _get_causal_mask(seq_len: int, diagonal: int = 0,
                     device: Optional[torch.device] = None):
    # mask keeping only previous steps
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    causal_mask = torch.triu(ones, diagonal)
    return causal_mask

def discretize_values(dist):
    matrix = torch.from_numpy(dist)
    nonzero_threshold=np.partition(np.unique(dist.flatten()), 1)[1]
    matrix = matrix.double()
    matrix = torch.where(matrix < nonzero_threshold, 0., matrix)
    matrix = torch.where((matrix < 0.1) & (matrix > nonzero_threshold) , 1., matrix)
    matrix = torch.where((matrix < 0.2) & (matrix > 0.1) , 2., matrix)
    matrix = torch.where((matrix < 0.3) & (matrix > 0.2) , 3., matrix)
    matrix = torch.where((matrix < 0.4) & (matrix > 0.3) , 4., matrix)
    matrix = torch.where((matrix < 0.5) & (matrix > 0.4) , 5., matrix)
    matrix = torch.where((matrix < 0.6) & (matrix > 0.5) , 6., matrix)
    matrix = torch.where((matrix < 0.7) & (matrix > 0.6) , 7., matrix)
    matrix = torch.where((matrix < 0.8) & (matrix > 0.7) , 8., matrix)
    matrix = torch.where((matrix < 0.9) & (matrix > 0.8) , 9., matrix)
    matrix = torch.where((matrix < 1.0) & (matrix > 0.9) , 10., matrix)

    return matrix


# class CentralityEncoding(nn.Module):
#     def __init__(self,
#                  hidden_size,
#                  max_in_degree,
#                  max_out_degree,
#                  in_degree_list,
#                  out_degree_list):
#         super(CentralityEncoding, self).__init__()
        
#         self.in_degree_encoder = nn.Embedding(max_in_degree, hidden_size, padding_idx=0)
#         self.out_degree_encoder = nn.Embedding(max_out_degree, hidden_size, padding_idx=0)
#         self.in_degree_list = in_degree_list.to('cuda' if torch.cuda.is_available() else 'cpu')
#         self.out_degree_list = out_degree_list.to('cuda' if torch.cuda.is_available() else 'cpu')

#     def forward(self):
#         return self.in_degree_encoder(self.in_degree_list) + self.out_degree_encoder(self.out_degree_list)

class GraphAttnBias(nn.Module):
    def __init__(self,
                 hidden_size,
                 max_dist,
                 spd_matrix):
        super(GraphAttnBias, self).__init__()
        
        self.spatial_pos_encoder = nn.Embedding(max_dist, hidden_size, padding_idx=0)
        self.spd_matrix = spd_matrix

    def forward(self):
        return self.spatial_pos_encoder(self.spd_matrix)

class GraphormerMHA(MultiheadAttention):
    def __init__(self, embed_dim, heads,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 attn_bias: Optional[Tensor] = None,
                 axis='steps',
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 device=None,
                 dtype=None,
                 causal=False) -> None:
        super(GraphormerMHA, self).__init__(
            embed_dim, heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=False,
            device=device,
            dtype=dtype)

        if axis in ['steps', 0]:
            shape = 's (b n) c'
        elif axis in ['nodes', 1]:
            if causal:
                raise ValueError(f'Cannot use causal attention for axis "{axis}".')
            shape = 'n (b s) c'
        else:
            raise ValueError("Axis can either be 'steps' (0) or 'nodes' (1), "
                             f"not '{axis}'.")
        self._in_pattern = f'b s n c -> {shape}'
        self._out_pattern = f'{shape} -> b s n c'
        self.causal = causal

        # change projections
        if qdim is not None and qdim != embed_dim:
            self.qdim = qdim
            self.q_proj = nn.Linear(self.qdim, embed_dim)
        else:
            self.qdim = embed_dim
            self.q_proj = nn.Identity()

        self.attn_bias = attn_bias
    
    def forward(self, query: Tensor,
                key: OptTensor = None,
                value: OptTensor = None,
                key_padding_mask: OptTensor = None,
                need_weights: bool = True, attn_mask: OptTensor = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        # inputs: [batches, steps, nodes, channels] -> [s (b n) c]
        if key is None:
            key = query
        if value is None:
            value = key
        batch = value.shape[0]
        query, key, value = [rearrange(x, self._in_pattern)
                                for x in (query, key, value)]

        if self.causal:
            causal_mask = _get_causal_mask(
                key.size(0), diagonal=1, device=query.device)
            if attn_mask is None:
                attn_mask = causal_mask
            else:
                attn_mask = torch.logical_and(attn_mask, causal_mask)
        
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_weights = multi_head_attention_forward(
                self.q_proj(query), key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, attn_bias=self.attn_bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_weights = multi_head_attention_forward(
                self.q_proj(query), key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, attn_bias=self.attn_bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            attn_output  = attn_output.transpose(1, 0)

        attn_output = rearrange(attn_output, self._out_pattern, b=batch)\
            .contiguous()
        if attn_weights is not None:
            attn_weights = rearrange(attn_weights, '(b d) l m -> b d l m',
                                     b=batch).contiguous()
        return attn_output, attn_weights

class GraphormerLayer(TransformerLayer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 attn_bias=None,
                 ff_size=None,
                 n_heads=1,
                 axis='steps',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(GraphormerLayer, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_heads=n_heads,
            axis=axis,
            causal=causal,
            activation=activation,
            dropout=dropout
        )
        
        self.att = GraphormerMHA(embed_dim=hidden_size,
                                 qdim=input_size,
                                 kdim=input_size,
                                 vdim=input_size,
                                 attn_bias = attn_bias,
                                 heads=n_heads,
                                 axis=axis,
                                 causal=causal)
        
class SpatioTemporalGraphormerLayer(SpatioTemporalTransformerLayer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 attn_bias=None,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(SpatioTemporalGraphormerLayer, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            n_heads=n_heads,
            causal=causal,
            activation=activation,
            dropout=dropout
        )
        
        """
        self.temporal_att = GraphormerMHA(embed_dim=hidden_size,
                                          qdim=input_size,
                                          kdim=input_size,
                                          vdim=input_size,
                                          attn_bias = attn_bias,
                                          heads=n_heads,
                                          axis='steps',
                                          causal=causal)
        """
        self.spatial_att = GraphormerMHA(embed_dim=hidden_size,
                                         qdim=hidden_size,
                                         kdim=hidden_size,
                                         vdim=hidden_size,
                                         attn_bias = attn_bias,
                                         heads=n_heads,
                                         axis='nodes',
                                         causal=False)
        
class Graphormer(Transformer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 attn_bias=None,
                 ff_size=None,
                 output_size=None,
                 n_layers=1,
                 n_heads=1,
                 axis='steps',
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(Graphormer, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            ff_size=ff_size,
            output_size=output_size,
            n_layers=n_layers,
            n_heads=n_heads,
            axis=axis,
            causal=causal,
            activation=activation,
            dropout=dropout
        )
        self.f = getattr(F, activation)

        if ff_size is None:
            ff_size = hidden_size

        if axis in ['steps', 'nodes']:
            transformer_layer = partial(GraphormerLayer, axis=axis)
        elif axis == 'both':
            transformer_layer = SpatioTemporalGraphormerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        layers = []
        for i in range(n_layers):
            layers.append(transformer_layer(input_size=input_size if i == 0 else hidden_size,
                                            hidden_size=hidden_size,
                                            ff_size=ff_size,
                                            n_heads=n_heads,
                                            causal=causal,
                                            activation=activation,
                                            dropout=dropout))

        self.net = nn.Sequential(*layers)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

class GraphormerModel(TransformerModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 ff_size,
                 exog_size,
                 horizon,
                 n_nodes,
                 n_heads,
                 n_layers,
                 max_in_degree,
                 max_out_degree,
                 max_dist,
                 in_degree_list,
                 out_degree_list,
                 spd_matrix,
                 dropout,
                 axis,
                 activation='gelu'):
        super(GraphormerModel, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            ff_size=ff_size,
            exog_size=exog_size,
            horizon=horizon,
            n_heads=n_heads,
            n_layers=n_layers,
            n_nodes=n_nodes,
            dropout=dropout,
            axis=axis,
            activation=activation)
        
        assert in_degree_list.shape[0] == out_degree_list.shape[0], \
            'the number of in_degrees and out_degrees must be equal'
        
        _attn_bias = GraphAttnBias(hidden_size, max_dist, spd_matrix)

        self.transformer_encoder = nn.Sequential(
            Graphormer(input_size=hidden_size,
                       hidden_size=hidden_size,
                       attn_bias=_attn_bias(),
                       ff_size=ff_size,
                       n_heads=n_heads,
                       n_layers=n_layers,
                       activation=activation,
                       dropout=dropout,
                       axis=axis),
            Select(1, -1)
        )

        self.sge = StaticGraphEmbedding(n_nodes, hidden_size)
        self.ce = CentralityEncoding(
            hidden_size, max_in_degree, max_out_degree, in_degree_list, out_degree_list)
        
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
        # x = self.pe(x) + self.ce() + self.sge()
        # x = self.pe(x)
        x = self.pe(x) + self.sge()
        x = self.transformer_encoder(x)

        return self.readout(x)

import sys

class Graph():
    def __init__(self, graph):
        self.V = graph.shape[0]
        self.graph = graph
 
    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = sys.maxsize
        min_index = -1
 
        # Search not nearest vertex not in the
        # shortest path tree
        for u in range(self.V):
            if dist[u] < min and sptSet[u] == False:
                min = dist[u]
                min_index = u
 
        return min_index
    
    def set_disconnected_nodes_val(self, dist, val = 0):
        min = sys.maxsize
        for u in range(self.V):
            if dist[u] == min:
                dist[u] = val
        return dist 
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src, disconnected_val = 0):
 
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # x is always equal to src in first iteration
            x = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[x] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for y in range(self.V):
                if self.graph[x][y] > 0 and sptSet[y] == False and \
                dist[y] > dist[x] + self.graph[x][y]:
                        dist[y] = dist[x] + self.graph[x][y]

        dist = self.set_disconnected_nodes_val(dist, disconnected_val)
        return dist
