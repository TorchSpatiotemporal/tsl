import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from tsl.nn.utils.connectivity import transpose, normalize


class DiffConv(MessagePassing):
    r"""An implementation of the Diffusion Convolution Layer.
    For details see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        k (int): Filter size :math:`K`.
        bias (bool, optional): If set to :obj:`False`, the layer
            will not learn an additive bias (default :obj:`True`).

    """

    def __init__(self, in_channels, out_channels, k,
                 root_weight: bool = True, bias=True):
        super(DiffConv, self).__init__(aggr="add", node_dim=-2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.root_weight = root_weight

        n_filters = 2 * k if not root_weight else 2 * k + 1

        self.filters = nn.Linear(in_channels * n_filters, out_channels, bias=bias)

        self._support = None
        self.reset_parameters()

    @staticmethod
    def compute_support_index(edge_index, edge_weights=None, num_nodes=None, add_backward=True):
        _, normalized_ew = normalize(edge_index, edge_weights, dim=1, num_nodes=num_nodes)
        # Add backward matrices
        if add_backward:
            return (edge_index, normalized_ew), \
                   DiffConv.compute_support_index(*transpose(edge_index, edge_weights),
                                                  num_nodes=num_nodes,
                                                  add_backward=False)
        # Normalize
        return (edge_index, normalized_ew)

    def reset_parameters(self):
        self.filters.reset_parameters()
        self._support = None

    def message(self, x_j: torch.Tensor, weight) -> torch.Tensor:
        # x_j: [batch, edges, channels]
        return weight.view(-1, 1) * x_j

    def forward(self,
                x: torch.FloatTensor,
                edge_index: torch.LongTensor, edge_weight=None, cache_support=False) -> torch.FloatTensor:
        """"""
        # x: [batch, (steps), nodes, nodes]
        n = x.size(-2)
        if self._support is None:
            support = self.compute_support_index(edge_index, edge_weight, num_nodes=n, add_backward=True)
            if cache_support:
                self._support = support
        else:
            support = self._support

        (edge_index_fwd, edge_weight_fwd), (edge_index_bwd, edge_weight_bwd) = support

        out = []
        if self.root_weight:
            out += [x]

        xk_f = x
        xk_b = x
        for _ in range(self.k):
            xk_f = self.propagate(edge_index_fwd, x=xk_f, weight=edge_weight_fwd)
            xk_b = self.propagate(edge_index_bwd, x=xk_b, weight=edge_weight_bwd)
            out += [xk_f, xk_b]

        out = torch.cat(out, -1)
        return self.filters(out)
