from typing import Union, Tuple, List

from torch import nn, Tensor

from tsl.nn.functional import expand_then_cat


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
