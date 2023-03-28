from typing import Tuple

import torch
from torch import Tensor

from ..recurrent.base import GRUCellBase, LSTMCellBase
from .linear import MultiLinear


class MultiGRUCell(GRUCellBase):
    r"""Multiple parallel gated recurrent unit (GRU) cells.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard
    product.

    Args:
        input_size (int): The number of features in the instance input sample.
        hidden_size (int): The number of features in the instance hidden state.
        n_instances (int): The number of parallel GRU cells. Each cell has
            different weights.
        bias (bool): If :obj:`True`, then the layer will learn an additive
            bias for each instance gate.
            (default: :obj:`True`)
        device (optional): The device of the parameters.
            (default: :obj:`None`)
        dtype (optional): The data type of the parameters.
            (default: :obj:`None`)

    Examples::

        >>> rnn = MultiGRUCell(20, 32, 10)
        >>> input = torch.randn(64, 12, 10, 20)
        >>> h = None
        >>> output = []
        >>> for i in range(12):
        ...     h = rnn(input[:, i], h)
        ...     output.append(h)
        >>> output = torch.stack(output, dim=1)
        >>> print(output.size())
        torch.Size([64, 12, 10, 32])
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_instances: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.n_instances = n_instances
        # instantiate gates
        in_size = input_size + hidden_size
        forget_gate = MultiLinear(in_size,
                                  hidden_size,
                                  n_instances,
                                  bias=bias,
                                  **factory_kwargs)
        update_gate = MultiLinear(in_size,
                                  hidden_size,
                                  n_instances,
                                  bias=bias,
                                  **factory_kwargs)
        candidate_gate = MultiLinear(in_size,
                                     hidden_size,
                                     n_instances,
                                     bias=bias,
                                     **factory_kwargs)
        super().__init__(hidden_size, forget_gate, update_gate, candidate_gate)

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0),
                           self.n_instances,
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)


class MultiLSTMCell(LSTMCellBase):
    r"""Multiple parallel long short-term memory (LSTM) cells.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard
    product.

    Args:
        input_size (int): The number of features in the instance input sample.
        hidden_size (int): The number of features in the instance hidden state.
        n_instances (int): The number of parallel LSTM cells. Each cell has
            different weights.
        bias (bool): If :obj:`True`, then the layer will learn an additive
            bias for each instance gate.
            (default: :obj:`True`)
        device (optional): The device of the parameters.
            (default: :obj:`None`)
        dtype (optional): The data type of the parameters.
            (default: :obj:`None`)

    Examples::

        >>> rnn = MultiLSTMCell(20, 32, 10)
        >>> input = torch.randn(64, 12, 10, 20)
        >>> h = None
        >>> output = []
        >>> for i in range(12):
        ...     h = rnn(input[:, i], h)  # h = h, c
        ...     output.append(h[0])      # i-th output is h_i
        >>> output = torch.stack(output, dim=1)
        >>> print(output.size())
        torch.Size([64, 12, 10, 32])
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_instances: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.n_instances = n_instances
        # instantiate gates
        in_size = input_size + hidden_size
        input_gate = MultiLinear(in_size,
                                 hidden_size,
                                 n_instances,
                                 bias=bias,
                                 **factory_kwargs)
        forget_gate = MultiLinear(in_size,
                                  hidden_size,
                                  n_instances,
                                  bias=bias,
                                  **factory_kwargs)
        cell_gate = MultiLinear(in_size,
                                hidden_size,
                                n_instances,
                                bias=bias,
                                **factory_kwargs)
        output_gate = MultiLinear(in_size,
                                  hidden_size,
                                  n_instances,
                                  bias=bias,
                                  **factory_kwargs)
        super().__init__(hidden_size, input_gate, forget_gate, cell_gate,
                         output_gate)

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0),
                            self.n_instances,
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device),
                torch.zeros(x.size(0),
                            self.n_instances,
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device))
