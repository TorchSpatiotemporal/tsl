import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

from tsl.nn.utils import get_layer_activation


class GatedGraphNetwork(MessagePassing):
    r"""Gate Graph Neural Network layer (with residual connections) inspired by
    the FC-GNN model from the paper `"Multivariate Time Series Forecasting with
    Latent Graph Inference" <https://arxiv.org/abs/2203.03423>`_ (Satorras et
    al., 2022).

    Args:
        input_size (int): Input channels.
        output_size (int): Output channels.
        activation (str, optional): Activation function.
        parametrized_skip_conn (bool, optional): If :obj:`True`, then add a
            linear layer in the residual connection even if input and output
            dimensions match.
            (default: :obj:`False`)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: str = 'silu',
                 parametrized_skip_conn: bool = False):
        super(GatedGraphNetwork, self).__init__(aggr="add", node_dim=-2)

        self.in_channels = input_size
        self.out_channels = output_size

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * input_size, output_size // 2),
            get_layer_activation(activation)(),
            nn.Linear(output_size // 2, output_size),
            get_layer_activation(activation)(),
        )

        self.gate_mlp = nn.Sequential(nn.Linear(output_size, 1), nn.Sigmoid())

        self.update_mlp = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            get_layer_activation(activation)(),
            nn.Linear(output_size, output_size))

        if (input_size != output_size) or parametrized_skip_conn:
            self.skip_conn = nn.Linear(input_size, output_size)
        else:
            self.skip_conn = nn.Identity()

    def forward(self, x: Tensor, edge_index: Adj):
        """"""
        out = self.propagate(edge_index, x=x)

        out = self.update_mlp(torch.cat([out, x], -1)) + self.skip_conn(x)

        return out

    def message(self, x_i: Tensor, x_j: Tensor):
        """"""
        mij = self.msg_mlp(torch.cat([x_i, x_j], -1))
        return self.gate_mlp(mij) * mij
