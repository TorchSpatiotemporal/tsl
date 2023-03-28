import torch
from einops import rearrange
from torch import Tensor


class BatchNorm(torch.nn.Module):
    r"""Applies graph-wise batch normalization.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, bool): Running stats momentum.
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): Whether to track stats to perform
            batch norm.
            (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats)

    def reset_parameters(self):
        self.module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        b, *_ = x.size()
        x = rearrange(x, 'b ... n c -> (b n) c ...')
        x = self.module(x)
        return rearrange(x, '(b n) c ... -> b ... n c', b=b)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'
