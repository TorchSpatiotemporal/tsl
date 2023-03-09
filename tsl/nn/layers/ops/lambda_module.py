from typing import Callable

from torch import nn, Tensor


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
