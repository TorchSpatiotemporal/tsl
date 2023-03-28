import torch
from torch import Tensor, nn

from .batch_norm import BatchNorm
from .instance_norm import InstanceNorm
from .layer_norm import LayerNorm


class Norm(torch.nn.Module):
    r"""Applies a normalization of the specified type.

    Args:
        in_channels (int): Size of each input sample.
    """

    def __init__(self, norm_type, in_channels, **kwargs):
        super().__init__()
        self.norm_type = norm_type
        self.in_channels = in_channels

        if norm_type == 'instance':
            norm_layer = InstanceNorm
        elif norm_type == 'batch':
            norm_layer = BatchNorm
        elif norm_type == 'layer':
            norm_layer = LayerNorm
        elif norm_type == 'none':
            norm_layer = nn.Identity
        else:
            raise NotImplementedError(
                f'"{norm_type}" is not a valid normalization option.')

        self.norm = norm_layer(in_channels, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return self.norm(x)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.norm_type},'
                f' {self.in_channels})')
