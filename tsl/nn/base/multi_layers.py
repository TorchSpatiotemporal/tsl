import math
from typing import Union, Tuple

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import init, functional as F

from tsl.nn.base.recurrent import GRUCell, LSTMCell
from tsl.nn.layers.ops import Activation


class MultiLinear(nn.Module):
    """Applies multiple different linear transformations to the incoming
    data.

    .. math::

        \mathbf{y} = [x_i W_i^T + b_i]_{i=0,\ldots,N}

    Args:
        in_channels (int): Size of instance input sample.
        out_channels (int): Size of instance output sample.
        n_instances (int): The number :math:`N` of parallel linear
            operations. Each operation has different weights and biases.
        instance_dim (int or str): Dimension of the instances (must match
            :attr:`n_instances` at runtime).
            (default: :obj:`-2`)
        channel_dim (int or str): Dimension of the input channels.
            (default: :obj:`-1`)
        bias (bool): If :obj:`True`, then the layer will learn an additive
            bias for each instance.
            (default: :obj:`True`)
        device (optional): The device of the parameters.
            (default: :obj:`None`)
        dtype (optional): The data type of the parameters.
            (default: :obj:`None`)

    Examples:

        >>> m = MultiLinear(20, 32, 10, pattern='t n f', instance_dim='n')
        >>> input = torch.randn(64, 12, 10, 20)  # shape: [b t n f]
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([64, 24, 10, 32])
    """

    def __init__(self, in_channels: int, out_channels: int, n_instances: int,
                 *,
                 ndim: int = None, pattern: str = None,
                 instance_dim: Union[int, str] = -2,
                 channel_dim: Union[int, str] = -1,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_instances = n_instances

        self.ndim = ndim
        self.instance_dim = instance_dim
        self.channel_dim = channel_dim

        # initialize by pattern, e.g.:
        #   pattern='t n f', instance_dim='n', instance_dim=-1
        #   pattern='t n f', instance_dim=-2, instance_dim=-1
        if pattern is not None:
            pattern = pattern.replace(' ', '')
            self.instance_dim = instance_dim if isinstance(instance_dim, str) \
                else pattern[instance_dim]
            self.channel_dim = channel_dim if isinstance(channel_dim, str) \
                else pattern[channel_dim]
            self.einsum_pattern = self._compute_einsum_pattern(pattern=pattern)
            self.bias_shape = self._compute_bias_shape(pattern=pattern)
            self.reshape_bias = False
        # initialize negative dim indexing (default), e.g.:
        #   instance_dim=-2, instance_dim=-1 (pattern=None, ndim=None)
        elif ndim is None and instance_dim < 0 and channel_dim < 0:
            ndim = abs(min(instance_dim, channel_dim))
            self.einsum_pattern = self._compute_einsum_pattern(ndim)
            self.bias_shape = self._compute_bias_shape(ndim)
            self.reshape_bias = False
        # initialize with ndim and dim (positive/negative) indexing, e.g.:
        #   ndim=3, instance_dim=1, instance_dim=-1
        elif ndim is not None:
            # initialize with lazy ndim calculation, e.g.:
            #   ndim=-1, instance_dim=1, instance_dim=-1
            if ndim < 0:
                # lazy initialize einsum pattern
                self.einsum_pattern = None
                self.bias_shape = (n_instances, out_channels)
                self.reshape_bias = True
                self._hook = self.register_forward_pre_hook(
                    self.initialize_module)
            else:
                self.einsum_pattern = self._compute_einsum_pattern(ndim)
                self.bias_shape = self._compute_bias_shape(ndim)
                self.reshape_bias = False
        # cannot initialize if all:
        #   1. pattern is None
        #   2. ndim is None and instance_dim >= 0 or channel_dim >= 0
        else:
            raise ValueError("One of 'pattern' or 'ndim' must be given if one "
                             "of 'instance_dim' or 'channel_dim' is positive.")

        self.weight: nn.Parameter = nn.Parameter(
            torch.empty((n_instances, in_channels, out_channels),
                        **factory_kwargs))

        if bias:
            self.bias: nn.Parameter = nn.Parameter(torch.empty(*self.bias_shape,
                                                               **factory_kwargs))
        else:
            self.register_parameter('bias', None)

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

    def _compute_bias_shape(self, ndim: int = None, pattern: str = None):
        if ndim is not None:
            bias_shape = [1] * ndim
            bias_shape[self.instance_dim] = self.n_instances
            bias_shape[self.channel_dim] = self.out_channels
        elif pattern is not None:
            pattern = pattern.replace(' ', '')
            bias_shape = []
            for token in pattern:
                if token == self.channel_dim:
                    bias_shape.append(self.out_channels)
                elif token == self.instance_dim:
                    bias_shape.append(self.n_instances)
                else:
                    bias_shape.append(1)
        else:
            raise ValueError("One of 'pattern' or 'ndim' must be given.")
        return tuple(bias_shape)

    def _compute_einsum_pattern(self, ndim: int = None, pattern: str = None):
        if ndim is not None:
            pattern = [chr(s + 97) for s in range(ndim)]  # 'a', 'b', ...
            pattern[self.instance_dim] = 'x'
            pattern[self.channel_dim] = 'y'
            input_pattern = ''.join(pattern)
            pattern[self.channel_dim] = 'z'
            output_pattern = ''.join(pattern)
            weight_pattern = 'xyz'
        elif pattern is not None:
            input_pattern = pattern.replace(' ', '')
            output_pattern = input_pattern.replace(self.channel_dim, 'z')
            weight_pattern = f'{self.instance_dim}{self.channel_dim}z'
        else:
            raise ValueError("One of 'pattern' or 'ndim' must be given.")
        return f"...{input_pattern},{weight_pattern}->...{output_pattern}"

    @torch.no_grad()
    def initialize_module(self, module, input):
        self.ndim = input[0].ndim
        self.einsum_pattern = self._compute_einsum_pattern(self.ndim)
        self.bias_shape = self._compute_bias_shape(self.ndim)
        self._hook.remove()
        delattr(self, '_hook')

    def forward(self, input: Tensor) -> Tensor:
        r"""Compute :math:`\mathbf{y} = [x_i W_i^T + b_i]_{i=0,\ldots,N}`"""
        out = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            if self.reshape_bias:
                out = out + self.bias.view(*self.bias_shape).contiguous()
            else:
                out = out + self.bias
        return out


class MultiDense(MultiLinear):

    def __init__(self, in_channels: int, out_channels: int, n_instances: int,
                 *,
                 ndim: int = None, pattern: str = None,
                 instance_dim: int = -2,
                 channel_dim: int = -1,
                 bias: bool = True,
                 dropout: float = 0.,
                 activation: str = 'relu',
                 device=None, dtype=None) -> None:
        super(MultiDense, self).__init__(in_channels, out_channels,
                                         n_instances=n_instances,
                                         ndim=ndim, pattern=pattern,
                                         instance_dim=instance_dim,
                                         channel_dim=channel_dim, bias=bias,
                                         device=device, dtype=dtype)
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


class MultiConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_instances: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[str, int] = 0,
                 dilation: int = 1,
                 bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiConv1d, self).__init__()

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
        x = rearrange(x, '... t n f -> ... (n f) t')

        out = F.conv1d(x, weight=self.weight, bias=None, stride=self.stride,
                       dilation=self.dilation, groups=self.n_instances,
                       padding=self.padding)

        out = rearrange(out, '... (n f) t -> ... t n f', f=self.out_channels)

        if self.bias is not None:
            out = out + self.bias

        return out


class MultiGRUCell(GRUCell):
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

    def __init__(self, input_size: int, hidden_size: int, n_instances: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.n_instances = n_instances
        # instantiate gates
        in_size = input_size + hidden_size
        forget_gate = MultiLinear(in_size, hidden_size, n_instances, bias=bias,
                                  **factory_kwargs)
        update_gate = MultiLinear(in_size, hidden_size, n_instances, bias=bias,
                                  **factory_kwargs)
        candidate_gate = MultiLinear(in_size, hidden_size, n_instances,
                                     bias=bias, **factory_kwargs)
        super().__init__(hidden_size, forget_gate, update_gate, candidate_gate)

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0), self.n_instances, self.hidden_size,
                           dtype=x.dtype, device=x.device)


class MultiLSTMCell(LSTMCell):
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

    def __init__(self, input_size: int, hidden_size: int, n_instances: int,
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.n_instances = n_instances
        # instantiate gates
        in_size = input_size + hidden_size
        input_gate = MultiLinear(in_size, hidden_size, n_instances, bias=bias,
                                 **factory_kwargs)
        forget_gate = MultiLinear(in_size, hidden_size, n_instances, bias=bias,
                                  **factory_kwargs)
        cell_gate = MultiLinear(in_size, hidden_size, n_instances, bias=bias,
                                **factory_kwargs)
        output_gate = MultiLinear(in_size, hidden_size, n_instances, bias=bias,
                                  **factory_kwargs)
        super().__init__(hidden_size, input_gate, forget_gate,
                         cell_gate, output_gate)

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0), self.n_instances, self.hidden_size,
                            dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0), self.n_instances, self.hidden_size,
                            dtype=x.dtype, device=x.device))
