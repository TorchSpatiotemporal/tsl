from torch import Tensor, nn


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
