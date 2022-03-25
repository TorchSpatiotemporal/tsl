import torch
from torch import nn

from torch_geometric.nn import MessagePassing

from tsl.nn.utils import get_layer_activation


class GatedGraphNetwork(MessagePassing):
    r"""

    Gate Graph Neural Network model inspired by
    Satorras et al., "Multivariate Time Series Forecasting with Latent Graph Inference", arxiv 2022.

    Args:
        input_size (int): Input channels.
        output_size (int): Output channels.
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation='silu'):
        super(GatedGraphNetwork, self).__init__(aggr="add", node_dim=-2)

        self.in_channels = input_size
        self.out_channels = output_size

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * input_size, output_size // 2),
            get_layer_activation(activation)(),
            nn.Linear(output_size // 2, output_size),
            get_layer_activation(activation)(),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            get_layer_activation(activation)(),
            nn.Linear(output_size, output_size)
        )

        if input_size != output_size:
            self.skip_conn = nn.Linear(input_size, output_size)
        else:
            self.skip_conn = nn.Identity()

    def forward(self, x, edge_index):
        """"""

        out = self.propagate(edge_index, x=x)

        out = self.update_mlp(torch.cat([out, x], -1)) + self.skip_conn(x)

        return out

    def message(self, x_i, x_j):
        mij = self.msg_mlp(torch.cat([x_i, x_j], -1))
        return self.gate_mlp(mij) * mij
