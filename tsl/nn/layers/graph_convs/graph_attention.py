import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch import nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import GATConv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops

from tsl.nn.functional import sparse_softmax


class AttentionScores(MessagePassing):

    def __init__(self,
                 embed_dim,
                 heads: int = 1,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 edge_dim: Optional[int] = None,
                 add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(AttentionScores, self).__init__(node_dim=-3, **kwargs)

        self.embed_dim = int(embed_dim)
        self.qdim = int(qdim) if qdim is not None else self.embed_dim
        self.kdim = int(kdim) if kdim is not None else self.embed_dim

        self.heads = heads
        self.edge_dim = edge_dim
        self.add_self_loops = add_self_loops
        self._alpha = None

        self.lin_key = Linear(self.kdim, heads * embed_dim)
        self.lin_query = Linear(self.qdim, heads * embed_dim)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * embed_dim, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()

    def forward(self,
                query: Tensor,
                key: Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None):
        n = (key if key.size(-2) > query.size(-2) else query).size(-2)
        h_and_c = self.heads, self.embed_dim
        # project and split heads
        query = self.lin_query(query).view(query.shape[:-1] + h_and_c)
        key = self.lin_key(key).view(key.shape[:-1] + h_and_c)
        # add self loops (i.e., attend also to self)
        if self.add_self_loops:
            edge_index, edge_attr = add_remaining_self_loops(edge_index,
                                                             edge_attr,
                                                             num_nodes=n)
        # project edge_attr
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(edge_attr.shape[:-1] +
                                                      h_and_c)
        # propagate and attend
        self.propagate(edge_index,
                       q=query,
                       k=key,
                       edge_attr=edge_attr,
                       size=(n, n))
        # retrieve scores
        alpha = self._alpha.mean(-1)
        self._alpha = None

        return alpha

    def message(self, q_i: Tensor, k_j: OptTensor, edge_attr: OptTensor,
                index: Tensor, size_i: int) -> Tensor:

        # cat edge_attr to query and key
        if edge_attr is not None:
            q_i = q_i + edge_attr
            k_j = k_j + edge_attr

        # compute scores
        alpha = (q_i * k_j).sum(dim=-1) / math.sqrt(self.embed_dim)
        alpha = sparse_softmax(alpha, index, num_nodes=size_i, dim=-2)
        self._alpha = alpha

        return alpha


class MultiHeadGraphAttention(MessagePassing):
    """The multi-head attention from the paper `Attention Is All You Need
    <https://arxiv.org/abs/1706.03762>`_ (Vaswani et al., NeurIPS 2017) for
    graph-structured data.

    Args:
        embed_dim (int): Size of the embedding dimension.
        num_heads (int): Number of attention heads.
            (default: :obj:`1`)
        qdim (int, optional): Number of features of the query. If :obj:`None`,
            then defaults to :attr:`embed_dim`.
            (default: :obj:`None`)
        kdim (int, optional): Number of features of the key. If :obj:`None`,
            then defaults to :attr:`embed_dim`.
            (default: :obj:`None`)
        vdim (int, optional): Number of features of the value. If :obj:`None`,
            then defaults to :attr:`embed_dim`.
            (default: :obj:`None`)
        edge_dim (int, optional): Number of edge features (:obj:`None` if there
            are no edge features).
            (default: :obj:`None`)
        concat (bool): If :obj:`True`, then the heads' outputs are concatenated
            along the feature dimension, and the dimension of each head's output
            is :obj:`embed_dim / num_heads`. Note that the total number of
            features in output is :attr:`embed_dim` in both cases.
            (default: :obj:`True`)
        dropout (float, optional): The dropout rate.
            (default: :obj:`0`)
        root_weight (bool): If :obj:`True`, then add a skip connection from the
            input with a linear transformation.
            (default :obj:`True`)
        bias (bool, optional): If :obj:`True`, then add a bias vector in output.
            (default: :obj:`True`)
        **kwargs: keyword arguments for the ``super(MessagePassing)`` call.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 qdim: Optional[int] = None,
                 kdim: Optional[int] = None,
                 vdim: Optional[int] = None,
                 edge_dim: Optional[int] = None,
                 concat: bool = True,
                 dropout: float = 0.,
                 root_weight: bool = True,
                 bias: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MultiHeadGraphAttention, self).__init__(node_dim=-3, **kwargs)

        self.embed_dim = int(embed_dim)
        self.qdim = int(qdim) if qdim is not None else self.embed_dim
        self.kdim = int(kdim) if kdim is not None else self.embed_dim
        self.vdim = int(vdim) if vdim is not None else self.embed_dim
        self.edge_dim = edge_dim

        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.root_weight = root_weight
        self._alpha = None

        if self.concat:
            self.head_dim = self.embed_dim // self.num_heads
            assert self.head_dim * self.num_heads == self.embed_dim, \
                "embed_dim must be divisible by heads"
        else:
            self.head_dim = self.embed_dim

        # key bias is discarded in softmax
        self.lin_key = Linear(self.kdim, num_heads * self.head_dim, bias=False)
        self.lin_query = Linear(self.qdim,
                                num_heads * self.head_dim,
                                bias_initializer='zeros')
        self.lin_value = Linear(self.vdim,
                                num_heads * self.head_dim,
                                bias_initializer='zeros')

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim,
                                   num_heads * self.head_dim,
                                   bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if self.root_weight:
            self.lin_skip = Linear(self.vdim, self.embed_dim, bias=bias)
        else:
            self.lin_skip = self.register_parameter('lin_skip', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: Optional[bool] = False,
                return_attention_matrix: Optional[bool] = False):
        """"""
        # inputs: [*batch, nodes, channels]
        n = value.size(-2)
        x = value  # save original input for skip connection
        # project and split heads
        h_and_c = self.num_heads, self.head_dim
        query = self.lin_query(query).view(query.shape[:-1] + h_and_c)
        key = self.lin_key(key).view(key.shape[:-1] + h_and_c)
        value = self.lin_value(value).view(value.shape[:-1] + h_and_c)
        # project edge_attr
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(edge_attr.shape[:-1] +
                                                      h_and_c)
        # propagate and attend
        out = self.propagate(edge_index,
                             q=query,
                             k=key,
                             v=value,
                             edge_attr=edge_attr,
                             size=(n, n))
        # Concatenate or average heads
        if self.concat:
            out = out.view(*out.shape[:-2], self.embed_dim)
        else:
            out = out.mean(dim=-2)
        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x)

        # retrieve scores
        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            return out, alpha
        elif return_attention_matrix:
            # arrange weights as (batched) dense matrix
            W = torch.zeros((*alpha.shape[:-2], n, n),
                            dtype=alpha.dtype,
                            device=alpha.device)
            W[..., edge_index[0], edge_index[1]] = alpha.mean(-1)
            return out, W
        else:
            return out

    def message(self, q_i: Tensor, k_j: OptTensor, v_j: OptTensor,
                edge_attr: OptTensor, index: Tensor, size_i: int) -> Tensor:
        """"""
        # cat edge_attr to query and key
        if edge_attr is not None:
            q_i = q_i + edge_attr
            k_j = k_j + edge_attr

        # compute scores
        alpha = (q_i * k_j).sum(dim=-1) / math.sqrt(self.embed_dim)
        alpha = sparse_softmax(alpha, index, num_nodes=size_i, dim=-2)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # attend
        out = v_j * alpha[..., None]
        return out


class GATLayer(nn.Module):

    def __init__(self, d_model, n_heads, concat=False, dropout=0.1):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels=d_model,
                                out_channels=d_model,
                                heads=n_heads,
                                dropout=dropout,
                                concat=concat)

    def forward(self, x, edge_index):
        # x:[b, s, n, f]
        b, s, n, _ = x.size()
        x = rearrange(x, 'b s n f -> (b s n) f')
        rep = b * s
        batch_edge_index = Batch.from_data_list([
            Data(edge_index=edge_index, num_nodes=n),
        ] * rep)
        x = self.gat_conv(x, batch_edge_index.edge_index)
        x = rearrange(x, '(b s n) f -> b s n f', b=b, s=s)
        return x
