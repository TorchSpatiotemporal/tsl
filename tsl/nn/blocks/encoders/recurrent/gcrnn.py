from tsl.nn.layers.recurrent import GraphConvGRUCell, GraphConvLSTMCell

from .base import RNNBase


class GraphConvRNN(RNNBase):
    r"""The Graph Convolutional Recurrent Network based on the paper
    `"Structured Sequence Modeling with Graph Convolutional Recurrent Networks"
    <https://arxiv.org/abs/1612.07659>`_ (Seo et al., ICONIP 2017), using
    :class:`~tsl.nn.layers.graph_convs.GraphConv` as graph convolution.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the hidden state.
        n_layers (int): Number of hidden layers.
            (default: ``1``)
        cat_states_layers (bool): If :obj:`True`, then the states of each layer
            are concatenated along the feature dimension.
            (default: :obj:`False`)
        return_only_last_state (bool): If :obj:`True`, then the ``forward()``
            method returns only the state at the end of the processing, instead
            of the full sequence of states.
            (default: :obj:`False`)
        cell (str): Type of graph recurrent cell that should be use
            (options: ``'gru'``, ``'lstm'``).
            (default: ``'gru'``)
        bias (bool): If :obj:`False`, then the layer will not learn an additive
            bias vector for each gate.
            (default: :obj:`True`)
        asymmetric_norm (bool): If :obj:`True`, then normalize the edge weights
            as :math:`a_{j \rightarrow i} =  \frac{a_{j \rightarrow i}}
            {deg_{i}}`, otherwise apply the GCN normalization.
            (default: :obj:`True`)
        root_weight (bool): If :obj:`True`, then add a filter (with different
            weights) for the root node itself.
            (default :obj:`True`)
        activation (str, optional): Activation function to be used, :obj:`None`
            for identity function (i.e., no activation).
            (default: :obj:`None`)
        cached (bool): If :obj:`True`, then cached the normalized edge weights
            computed in the first call.
            (default :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        cat_states_layers: bool = False,
        return_only_last_state: bool = False,
        cell: str = "gru",
        bias: bool = True,
        asymmetric_norm: bool = True,
        root_weight: bool = True,
        activation: str = None,
        cached: bool = False,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

        if cell == "gru":
            cell = GraphConvGRUCell
        elif cell == "lstm":
            cell = GraphConvLSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        rnn_cells = [
            cell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                asymmetric_norm=asymmetric_norm,
                root_weight=root_weight,
                activation=activation,
                bias=bias,
                cached=cached,
                **kwargs,
            )
            for i in range(n_layers)
        ]
        super(GraphConvRNN, self).__init__(
            rnn_cells, cat_states_layers, return_only_last_state
        )
