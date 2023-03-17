import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import inits


class InstanceNorm(torch.nn.Module):
    r"""Applies graph-wise instance normalization.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
    """

    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor(in_channels))
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.ones(self.weight)
        inits.zeros(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        # x : [*, nodes, features]
        mean = torch.mean(x, dim=-2, keepdim=True)
        std = torch.std(x, dim=-2, unbiased=False, keepdim=True)

        out = (x - mean) / (std + self.eps)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'
