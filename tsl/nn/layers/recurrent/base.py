from typing import Tuple, Union

import torch
from torch import Tensor, nn

StateType = Union[Tensor, Tuple[Tensor]]


class RNNCellBase(nn.Module):
    """Base class for implementing recurrent neural networks (RNN) cells."""

    def initialize_state(self, *args, **kwargs) -> StateType:
        raise NotImplementedError


class GRUCellBase(RNNCellBase):
    """Base class for implementing gated recurrent unit (GRU) cells."""

    def __init__(self, hidden_size: int, forget_gate: nn.Module,
                 update_gate: nn.Module, candidate_gate: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = forget_gate
        self.update_gate = update_gate
        self.candidate_gate = candidate_gate

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_size={self.hidden_size})'

    def reset_parameters(self):
        self.forget_gate.reset_parameters()
        self.update_gate.reset_parameters()
        self.candidate_gate.reset_parameters()

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0),
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)

    def forward(self, x: Tensor, h: Tensor, *args, **kwargs) -> Tensor:
        """"""
        # x: [batch, *, channels]
        # h: [batch, *, channels]
        x_gates = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.forget_gate(x_gates, *args, **kwargs))
        u = torch.sigmoid(self.update_gate(x_gates, *args, **kwargs))
        x_c = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.candidate_gate(x_c, *args, **kwargs))
        h_new = u * h + (1. - u) * c
        return h_new


class GRUCell(nn.GRUCell, RNNCellBase):

    __doc__ = nn.GRUCell.__doc__

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_size={self.hidden_size})'

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0),
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)


class GraphGRUCellBase(GRUCellBase):
    """Base class for implementing graph-based gated recurrent unit (GRU)
    cells."""

    def initialize_state(self, x) -> Tensor:
        return torch.zeros(x.size(0),
                           x.size(-2),
                           self.hidden_size,
                           dtype=x.dtype,
                           device=x.device)


class LSTMCellBase(RNNCellBase):
    """Base class for implementing long short-term memory (LSTM) cells."""

    def __init__(self, hidden_size: int, input_gate: nn.Module,
                 forget_gate: nn.Module, cell_gate: nn.Module,
                 output_gate: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gate = input_gate
        self.forget_gate = forget_gate
        self.cell_gate = cell_gate
        self.output_gate = output_gate

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_size={self.hidden_size})'

    def reset_parameters(self):
        self.input_gate.reset_parameters()
        self.forget_gate.reset_parameters()
        self.cell_gate.reset_parameters()
        self.output_gate.reset_parameters()

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0),
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device),
                torch.zeros(x.size(0),
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device))

    def forward(self, x: Tensor, hc: Tuple[Tensor, Tensor], *args,
                **kwargs) -> Tuple[Tensor, Tensor]:
        """"""
        # x: [batch, *, channels]
        # hc: (h=[batch, *, channels], c=[batch, *, channels])
        h, c = hc
        x_gates = torch.cat([x, h], dim=-1)
        i = torch.sigmoid(self.input_gate(x_gates, *args, **kwargs))
        f = torch.sigmoid(self.forget_gate(x_gates, *args, **kwargs))
        g = torch.tanh(self.cell_gate(x_gates, *args, **kwargs))
        o = torch.sigmoid(self.output_gate(x_gates, *args, **kwargs))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class LSTMCell(nn.LSTMCell, RNNCellBase):

    __doc__ = nn.LSTMCell.__doc__

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_size={self.hidden_size})'

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0),
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device),
                torch.zeros(x.size(0),
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device))


class GraphLSTMCellBase(LSTMCellBase):
    """Base class for implementing graph-based long short-term memory (LSTM)
     cells."""

    def initialize_state(self, x) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(x.size(0),
                            x.size(-2),
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device),
                torch.zeros(x.size(0),
                            x.size(-2),
                            self.hidden_size,
                            dtype=x.dtype,
                            device=x.device))
