from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
import numpy as np


class Lambda(nn.Module):

    def __init__(self, action):
        super(Lambda, self).__init__()
        self.action = action

    def forward(self, input: Tensor) -> Tensor:
        return self.action(input)


class Concatenate(nn.Module):

    def __init__(self, dim: int = 0):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, tensors: Union[Tuple[Tensor, ...], List[Tensor]]) \
            -> Tensor:
        shapes = [t.shape for t in tensors]
        expand_dims = list(np.max(shapes, 0))
        expand_dims[self.dim] = -1
        tensors = [t.expand(*expand_dims) for t in tensors]
        return torch.cat(tensors, dim=self.dim)


class Select(nn.Module):
    """
    Select one element along a dimension.
    """
    def __init__(self, dim, index):
        super(Select, self).__init__()
        self.dim = dim
        self.index = index

    def forward(self, tensor: Tensor) \
            -> Tensor:
        return tensor.select(self.dim, self.index)
