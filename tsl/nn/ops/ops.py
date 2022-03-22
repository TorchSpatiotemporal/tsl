from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
import numpy as np

from ..functional import expand_then_cat

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
        return expand_then_cat(tensors, self.dim)


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
