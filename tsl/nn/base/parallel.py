import math
from typing import Union, Tuple

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import init, functional as F

from tsl.nn.base.recurrent import GRUCell, LSTMCell
from tsl.nn.layers.ops import Activation


class ParallelLinear(nn.Module):
    """Applies multiple different linear transformations to the incoming
    data.

    .. math::

        \mathbf{y} = [x_i W_i^T + b_i]_{i=0,\ldots,N}

    Args:
        in_channels (int): Size of instance input sample.
        out_channels (int): Size of instance output sample.
        n_instances (int): The number :math:`N` of parallel linear
            operations. Each operation has different weights and biases.
        parallel_dim (int): Dimension of the instances (must match
            :attr:`n_instances` at runtime).
            (default: :obj:`-2`)
        channels_dim (int): Dimension of the input channels.
            (default: :obj:`-1`)
        bias (bool): If :obj:`True`, then the layer will learn an additive
            bias for each instance.
            (default: :obj:`True`)
        device (optional): The device of the parameters.
            (default: :obj:`None`)
        dtype (optional): The data type of the parameters.
            (default: :obj:`None`)

    Examples:

        >>> m = ParallelLinear(20, 32, 10)
        >>> input = torch.randn(64, 12, 10, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([64, 24, 10, 32])
    """

    def __init__(self, in_channels: int, out_channels: int, n_instances: int,
                 parallel_dim: int = -2,
                 channels_dim: int = -1,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_instances = n_instances

        self.parallel_dim = parallel_dim
        self.channels_dim = channels_dim

        self.weight = nn.Parameter(
            torch.empty((n_instances, in_channels, out_channels),
                        **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_instances, out_channels,
                                                 **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.einsum_pattern = None
        self._hook = self.register_forward_pre_hook(self.initialize_module)

        self.reset_parameters()

    def extra_repr(self) -> str:
        """"""
        return 'in_channels={}, out_channels={}, n_instances={}'.format(
            self.in_channels, self.out_channels, self.n_instances
        )

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_channels)
        init.uniform_(self.weight.data, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias.data, -bound, bound)

    @torch.no_grad()
    def initialize_module(self, module, input):
        pattern = [chr(s + 97) for s in range(input[0].ndim)]  # 'a', 'b', ...
        pattern[self.parallel_dim] = 'x'
        pattern[self.channels_dim] = 'y'
        input_pattern = ''.join(pattern)
        pattern[self.channels_dim] = 'z'
        output_pattern = ''.join(pattern)
        self.einsum_pattern = f"{input_pattern},xyz->{output_pattern}"

        if self.bias is not None:
            shape = [1] * input[0].ndim
            shape[self.parallel_dim] = self.n_instances
            shape[self.channels_dim] = self.out_channels
            self.bias = nn.Parameter(self.bias.view(*shape).contiguous())

        self._hook.remove()
        delattr(self, '_hook')

    def forward(self, input: Tensor) -> Tensor:
        r"""Compute :math:`\mathbf{y} = [x_i W_i^T + b_i]_{i=0,\ldots,N}`"""
        out = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class ParallelDense(ParallelLinear):

    def __init__(self, in_channels: int, out_channels: int, n_instances: int,
                 parallel_dim: int = -2,
                 channels_dim: int = -1,
                 bias: bool = True,
                 dropout: float = 0.,
                 activation: str = 'relu',
                 device=None, dtype=None) -> None:
        super(ParallelDense, self).__init__(in_channels, out_channels,
                                            n_instances=n_instances,
                                            parallel_dim=parallel_dim,
                                            channels_dim=channels_dim,
                                            bias=bias,
                                            device=device,
                                            dtype=dtype)
        activation = activation or 'linear'
        self.activation = activation.lower()
        if dropout > 0.:
            self.out = nn.Sequential(Activation(self.activation),
                                     nn.Dropout(dropout))
        else:
            self.out = Activation(self.activation)

    def forward(self, input: Tensor) -> Tensor:
        r"""Compute :math:`\mathbf{y} = \sigmoid\left([x_i W_i^T + b_i]
        _{i=0,\ldots,N}\right)`"""
        out = super().forward(input)
        return self.out(out)


class ParallelConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_instances: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[str, int] = 0,
                 dilation: int = 1,
                 bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ParallelConv1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_instances = n_instances
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty((n_instances * out_channels, in_channels, kernel_size),
                        **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(n_instances, out_channels, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        return f'{self.in_channels}, {self.out_channels}, ' \
               f'kernel_size={self.kernel_size}, n_instances={self.n_instances}'

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_channels * self.kernel_size)
        init.uniform_(self.weight.data, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias.data, -bound, bound)

    def forward(self, x):
        x = rearrange(x, 'b t n f -> b (n f) t')

        out = F.conv1d(x, weight=self.weight, bias=None, stride=self.stride,
                       dilation=self.dilation, groups=self.n_instances,
                       padding=self.padding)

        out = rearrange(out, 'b (n f) t -> b t n f', f=self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out


class ParallelGRUCell(GRUCell):
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

        >>> rnn = ParallelGRUCell(20, 32, 10)
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

    def __init__(self, input_size: int, hidden_size: int, n_instances: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.n_instances = n_instances
        # instantiate gates
        in_size = input_size + hidden_size
        forget_gate = ParallelLinear(in_size, hidden_size, n_instances,
                                     bias=bias, **factory_kwargs)
        update_gate = ParallelLinear(in_size, hidden_size, n_instances,
                                     bias=bias, **factory_kwargs)
        candidate_gate = ParallelLinear(in_size, hidden_size, n_instances,
                                        bias=bias, **factory_kwargs)
        super().__init__(hidden_size, forget_gate, update_gate, candidate_gate)

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0), self.n_instances, self.hidden_size,
                           dtype=x.dtype, device=x.device)


class ParallelLSTMCell(LSTMCell):
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

        >>> rnn = ParallelLSTMCell(20, 32, 10)
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

    def __init__(self, input_size: int, hidden_size: int, n_instances: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.n_instances = n_instances
        # instantiate gates
        in_size = input_size + hidden_size
        input_gate = ParallelLinear(in_size, hidden_size, n_instances,
                                    bias=bias, **factory_kwargs)
        forget_gate = ParallelLinear(in_size, hidden_size, n_instances,
                                     bias=bias, **factory_kwargs)
        cell_gate = ParallelLinear(in_size, hidden_size, n_instances,
                                   bias=bias, **factory_kwargs)
        output_gate = ParallelLinear(in_size, hidden_size, n_instances,
                                     bias=bias, **factory_kwargs)
        super().__init__(hidden_size, input_gate, forget_gate,
                         cell_gate, output_gate)

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0), self.n_instances, self.hidden_size,
                            dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0), self.n_instances, self.hidden_size,
                            dtype=x.dtype, device=x.device))
