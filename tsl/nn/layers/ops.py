from typing import Union, Tuple, List, Callable

import torch
from torch import nn, Tensor

from tsl.nn.functional import expand_then_cat
from tsl.nn.utils import get_layer_activation

__all__ = [
    'Lambda',
    'Concatenate',
    'Select',
    'GradNorm',
    'Activation'
]


class Lambda(nn.Module):
    """Call a generic function on the input.

    Args:
        function (callable): The function to call in :obj:`forward(input)`.
    """

    def __init__(self, function: Callable):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, input: Tensor) -> Tensor:
        """Returns :obj:`self.function(input)`."""
        return self.function(input)


class Concatenate(nn.Module):
    """Concatenate tensors along dimension :attr:`dim`.

    The tensors dimensions are matched (i.e., broadcasted if necessary) before
    concatenation.

    Args:
        dim (int): The dimension to concatenate on.
            (default: :obj:`0`)
    """

    def __init__(self, dim: int = 0):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, tensors: Union[Tuple[Tensor, ...], List[Tensor]]) \
            -> Tensor:
        """Returns :func:`~tsl.nn.functional.expand_then_cat` on input
        tensors."""
        return expand_then_cat(tensors, self.dim)


class Select(nn.Module):
    """Apply :func:`~torch.select` to select one element from a
    :class:`~torch.Tensor` along a dimension.

    This layer returns a view of the original tensor with the given dimension
    removed.

    Args:
        dim (int): The dimension to slice.
        index (int): The index to select with.
    """

    def __init__(self, dim: int, index: int):
        super(Select, self).__init__()
        self.dim = dim
        self.index = index

    def forward(self, tensor: Tensor) -> Tensor:
        """Returns :func:`~torch.select` on input tensor."""
        return tensor.select(self.dim, self.index)


class GradNorm(torch.autograd.Function):
    """Scales the gradient in back-propagation. In the forward pass is an
    identity operation."""

    @staticmethod
    def forward(ctx, x, norm):
        """"""
        ctx.save_for_backward(x)
        ctx.norm = norm  # save normalization coefficient
        return x  # identity

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        norm = ctx.norm
        return grad_output / norm, None  # return the normalized gradient


class Activation(nn.Module):
    r"""
    A utility layer for any activation function.

    Args:
        activation (str): Name of the activation function.
        **kwargs: Keyword arguments for the activation layer.
    """

    def __init__(self, activation: str, **kwargs):
        super(Activation, self).__init__()
        activation_class = get_layer_activation(activation)
        self.activation: nn.Module = activation_class(**kwargs)

    def forward(self, x):
        return self.activation(x)
