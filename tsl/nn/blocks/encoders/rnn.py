import torch

from torch import nn
from einops import rearrange


class RNN(nn.Module):
    r"""
        Simple RNN encoder with optional linear readout.

        Args:
            input_size (int): Input size.
            hidden_size (int): Units in the hidden layers.
            output_size (int, optional): Size of the optional readout.
            n_layers (int, optional): Number of hidden layers. (default: 1)
            cell (str, optional): Type of cell that should be use (options: [`gru`, `lstm`]). (default: `gru`)
            dropout (float, optional): Dropout probability.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 n_layers=1,
                 dropout=0.,
                 cell='gru'):
        super(RNN, self).__init__()

        if cell == 'gru':
            cell = nn.GRU
        elif cell == 'lstm':
            cell = nn.LSTM
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        self.rnn = cell(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=n_layers,
                        dropout=dropout)

        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x, return_last_state=False):
        """

        Args:
            x (torch.Tensor): Input tensor.
            return_last_state: Whether to return only the state corresponding to the last time step.
        """
        # x: [batches, steps, nodes, features]
        b, *_ = x.size()
        x = rearrange(x, 'b s n f -> s (b n) f')
        x, *_ = self.rnn(x)
        # [steps batches * nodes, features] -> [steps batches, nodes, features]
        x = rearrange(x, 's (b n) f -> b s n f', b=b)
        if return_last_state:
            x = x[:, -1]
        if self.readout is not None:
            return self.readout(x)
        return x
