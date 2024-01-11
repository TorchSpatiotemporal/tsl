from typing import Callable

from torch import Tensor, nn


class Lambda(nn.Module):
    """Call a generic function on the input.

    Args:
        function (callable): The function to call in :obj:`forward(input)`.
    """

    def __init__(self, function: Callable):
        super(Lambda, self).__init__()
        self.function = function
        self.__function_src__ = ''
        if function.__name__ == '<lambda>':
            import inspect
            src = inspect.getsource(function)
            src = src[src.find('lambda') + 7:]  # cut from 'lambda ' on
            src = src[:len(src) - src[::-1].find(')') -
                      1]  # cut until last ')'
            self.__function_src__ = src

    def extra_repr(self) -> str:
        return self.__function_src__

    def forward(self, input: Tensor) -> Tensor:
        """Returns :obj:`self.function(input)`."""
        return self.function(input)
