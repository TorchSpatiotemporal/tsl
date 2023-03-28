from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (from_scipy_sparse_matrix, remove_self_loops,
                                   to_scipy_sparse_matrix)
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tsl.nn.layers.base import NodeEmbedding
from tsl.nn.layers.graph_convs import DiffConv
from tsl.nn.layers.norm import LayerNorm
from tsl.ops.connectivity import asymmetric_norm, power_series, transpose

from .dcrnn import DCRNNCell


def compute_support(edge_index: LongTensor,
                    edge_weight: OptTensor = None,
                    order: int = 1,
                    num_nodes: Optional[int] = None,
                    add_backward: bool = True):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    ei, ew = asymmetric_norm(edge_index,
                             edge_weight,
                             dim=1,
                             num_nodes=num_nodes)
    a = to_scipy_sparse_matrix(ei, ew, num_nodes)
    support = []
    ak = a
    for i in range(order - 1):
        ak = ak * a
        ak.setdiag(0.)
        ak.eliminate_zeros()
        support.append(ak)
    support = [(ei, ew)] + [from_scipy_sparse_matrix(ak) for ak in support]
    if add_backward:
        ei_t, ew_t = transpose(edge_index, edge_weight)
        return support + compute_support(ei_t, ew_t, order, num_nodes, False)
    return support


class SpatialDecoder(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: Optional[int] = None,
                 exog_size: int = 0,
                 order: int = 1):
        super(SpatialDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.order = order

        exog_size = exog_size
        in_channels = input_size * 2 + hidden_size + exog_size

        self.lin_in = nn.Linear(in_channels, hidden_size)
        self.graph_conv = DiffConv(in_channels=hidden_size,
                                   out_channels=hidden_size,
                                   root_weight=False,
                                   k=1)
        self.lin_out = nn.Linear(2 * hidden_size, hidden_size)
        self.read_out = nn.Linear(2 * hidden_size, self.output_size)
        self.activation = nn.PReLU()

    def __repr__(self):
        attrs = ['input_size', 'hidden_size', 'output_size', 'order']
        attrs = ', '.join([f'{attr}={getattr(self, attr)}' for attr in attrs])
        return f"{self.__class__.__name__}({attrs})"

    def compute_support(self,
                        edge_index: LongTensor,
                        edge_weight: OptTensor = None,
                        num_nodes: Optional[int] = None,
                        add_backward: bool = True):
        ei, ew = asymmetric_norm(edge_index,
                                 edge_weight,
                                 dim=1,
                                 num_nodes=num_nodes)
        ei, ew = power_series(ei, ew, self.order)
        ei, ew = remove_self_loops(ei, ew)
        if add_backward:
            ei_t, ew_t = transpose(edge_index, edge_weight)
            return (ei, ew), self.compute_support(ei_t, ew_t, num_nodes, False)
        return ei, ew

    def forward(self,
                x: Tensor,
                mask: Tensor,
                h: Tensor,
                edge_index: LongTensor,
                edge_weight: OptTensor = None,
                u: OptTensor = None):
        # x: [batch, nodes, channels]
        x_in = [x, mask, h]
        if u is not None:
            x_in += [u]
        x_in = torch.cat(x_in, -1)
        x_in = self.lin_in(x_in)
        if self.order > 1:
            support = self.compute_support(edge_index,
                                           edge_weight,
                                           x.size(1),
                                           add_backward=True)
            self.graph_conv._support = support
            out = self.graph_conv(x_in, edge_index=None)
            self.graph_conv._support = None
        else:
            edge_index, edge_weight = remove_self_loops(
                edge_index, edge_weight)
            out = self.graph_conv(x_in, edge_index, edge_weight)
        # out MLP
        out = torch.cat([out, h], -1)
        out = self.activation(self.lin_out(out))
        out = torch.cat([out, h], -1)
        return self.read_out(out), out


class GRINCell(nn.Module):
    r"""The Graph Recurrent Imputation cell with `Diffusion Convolution
    <https://arxiv.org/abs/1707.01926>`_ from the paper
    `"Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural
    Networks" <https://arxiv.org/abs/2108.00298>`_ (Cini et al., ICLR 2022).

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the DCRNN hidden layer.
            (default: :obj:`64`)
        exog_size (int): Number of channels in the exogenous variables, if any.
            (default: :obj:`0`)
        n_layers (int): Number of stacked DCRNN cells.
            (default: :obj:`1`)
        n_nodes (int, optional): Number of nodes in the input graph.
            (default: :obj:`None`)
        kernel_size (int): Order of the spatial diffusion process in the DCRNN
            cells. (default: :obj:`2`)
        decoder_order (int): Order of the spatial diffusion process in the
            spatial decoder.
            (default: :obj:`1`)
        layer_norm (bool, optional): If :obj:`True`, then use layer
            normalization.
            (default: :obj:`False`)
        dropout (float, optional): Dropout probability in the DCRNN cells.
            (default: :obj:`0`)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 exog_size: int = 0,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None,
                 kernel_size: int = 2,
                 decoder_order: int = 1,
                 layer_norm: bool = False,
                 dropout: float = 0.):
        super(GRINCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.u_size = exog_size
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # input + mask + (eventually) exogenous
        rnn_input_size = 2 * self.input_size + exog_size

        # Spatio-temporal encoder (rnn_input_size -> hidden_size)
        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            in_channels = rnn_input_size if i == 0 else self.hidden_size
            cell = DCRNNCell(input_size=in_channels,
                             hidden_size=self.hidden_size,
                             k=kernel_size,
                             root_weight=True)
            self.cells.append(cell)
            norm = LayerNorm(self.hidden_size) if layer_norm else nn.Identity()
            self.norms.append(norm)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # Fist stage readout
        self.first_stage = nn.Linear(self.hidden_size, self.input_size)

        # Spatial decoder (rnn_input_size + hidden_size -> hidden_size)
        self.spatial_decoder = SpatialDecoder(input_size=input_size,
                                              hidden_size=hidden_size,
                                              exog_size=exog_size,
                                              order=decoder_order)

        # Hidden state initialization embedding
        if n_nodes is not None:
            self.h0 = nn.ModuleList()
            for _ in range(self.n_layers):
                self.h0.append(NodeEmbedding(n_nodes, self.hidden_size))
        else:
            self.register_parameter('h0', None)

    def __repr__(self):
        attrs = ['input_size', 'hidden_size', 'kernel_size', 'n_layers']
        attrs = ', '.join([f'{attr}={getattr(self, attr)}' for attr in attrs])
        return f"{self.__class__.__name__}({attrs})"

    def get_h0(self, x):
        if self.h0 is not None:
            return [h(expand=(x.shape[0], -1, -1)) for h in self.h0]
        size = (self.n_layers, x.shape[0], x.shape[2], self.hidden_size)
        return [*torch.zeros(size, device=x.device)]

    def update_state(self, x, h, edge_index, edge_weight):
        # x: [batch, nodes, channels]
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            h[layer] = norm(cell(rnn_in, h[layer], edge_index, edge_weight))
            rnn_in = h[layer]
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self,
                x: Tensor,
                edge_index: LongTensor,
                edge_weight: OptTensor = None,
                mask: OptTensor = None,
                u: OptTensor = None,
                h: Union[List[Tensor], Tensor] = None):
        """"""
        # x: [batch, steps, nodes, channels]
        steps = x.size(1)

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)

        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x)  # [[b n h] * n_layers]
        elif not isinstance(h, list):
            h = [*h]

        # Temporal conv
        predictions, imputations, states = [], [], []
        representations = []
        for step in range(steps):
            x_s = x[:, step]
            m_s = mask[:, step]
            h_s = h[-1]
            u_s = u[:, step] if u is not None else None
            # firstly impute missing values with predictions from state
            xs_hat_1 = self.first_stage(h_s)
            # fill missing values in input with prediction
            x_s = torch.where(m_s.bool(), x_s, xs_hat_1)
            # prepare inputs
            # retrieve maximum information from neighbors
            xs_hat_2, repr_s = self.spatial_decoder(x_s,
                                                    m_s,
                                                    h_s,
                                                    u=u_s,
                                                    edge_index=edge_index,
                                                    edge_weight=edge_weight)
            # readout of imputation state + mask to retrieve imputations
            # prepare inputs
            x_s = torch.where(m_s.bool(), x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=-1)  # x_hat_2 + mask + exogenous
            # update state with original sequence filled using imputations
            h = self.update_state(inputs, h, edge_index, edge_weight)
            # store imputations and states
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)

        # Aggregate outputs -> [batch, steps, nodes, channels]
        imputations = torch.stack(imputations, dim=1)
        predictions = torch.stack(predictions, dim=1)
        states = torch.stack(states, dim=1)
        representations = torch.stack(representations, dim=1)

        return imputations, predictions, representations, states
