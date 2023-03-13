from tsl.nn.layers.graph_convs import GraphConv
from .base import GraphGRUCell, GraphLSTMCell


class GraphConvGRUCell(GraphGRUCell):
    r"""Gated Recurrent Unit with :class:`~tsl.nn.layers.graph_convs.GraphConv`
    as graph convolution in the gates, based on the paper
    `"Structured Sequence Modeling with Graph Convolutional Recurrent Networks"
    <https://arxiv.org/abs/1612.07659>`_ (Seo et al., ICONIP 2017).

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the hidden state.
        bias (bool): If :obj:`True`, then the layer will learn an additive bias
            for each gate.
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

    def __init__(self, input_size: int, hidden_size: int,
                 bias: bool = True,
                 asymmetric_norm: bool = True,
                 root_weight: bool = True,
                 activation: str = None,
                 cached: bool = False,
                 **kwargs):
        self.input_size = input_size
        # instantiate gates
        forget_gate = GraphConv(input_size + hidden_size, hidden_size,
                                asymmetric_norm=asymmetric_norm,
                                root_weight=root_weight,
                                activation=activation,
                                bias=bias, cached=cached,
                                **kwargs)
        update_gate = GraphConv(input_size + hidden_size, hidden_size,
                                asymmetric_norm=asymmetric_norm,
                                root_weight=root_weight,
                                activation=activation,
                                bias=bias, cached=cached,
                                **kwargs)
        candidate_gate = GraphConv(input_size + hidden_size, hidden_size,
                                   asymmetric_norm=asymmetric_norm,
                                   root_weight=root_weight,
                                   activation=activation,
                                   bias=bias, cached=cached,
                                   **kwargs)
        super(GraphConvGRUCell, self).__init__(hidden_size=hidden_size,
                                               forget_gate=forget_gate,
                                               update_gate=update_gate,
                                               candidate_gate=candidate_gate)


class GraphConvLSTMCell(GraphLSTMCell):
    r"""LSTM with :class:`~tsl.nn.layers.graph_convs.GraphConv` as graph
    convolution in the gates, based on the paper `"Structured Sequence Modeling
    with Graph Convolutional Recurrent Networks"
    <https://arxiv.org/abs/1612.07659>`_ (Seo et al., ICONIP 2017).

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the hidden state.
        bias (bool): If :obj:`True`, then the layer will learn an additive bias
            for each gate.
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

    def __init__(self, input_size: int, hidden_size: int,
                 bias: bool = True,
                 asymmetric_norm: bool = True,
                 root_weight: bool = True,
                 activation='linear',
                 cached: bool = False,
                 **kwargs):
        self.input_size = input_size
        # instantiate gates
        input_gate = GraphConv(input_size + hidden_size, hidden_size,
                               asymmetric_norm=asymmetric_norm,
                               root_weight=root_weight,
                               activation=activation,
                               bias=bias, cached=cached,
                               **kwargs)
        forget_gate = GraphConv(input_size + hidden_size, hidden_size,
                                asymmetric_norm=asymmetric_norm,
                                root_weight=root_weight,
                                activation=activation,
                                bias=bias, cached=cached,
                                **kwargs)
        cell_gate = GraphConv(input_size + hidden_size, hidden_size,
                              asymmetric_norm=asymmetric_norm,
                              root_weight=root_weight,
                              activation=activation,
                              bias=bias, cached=cached,
                              **kwargs)
        output_gate = GraphConv(input_size + hidden_size, hidden_size,
                                asymmetric_norm=asymmetric_norm,
                                root_weight=root_weight,
                                activation=activation,
                                bias=bias, cached=cached,
                                **kwargs)
        super(GraphConvLSTMCell, self).__init__(hidden_size=hidden_size,
                                                input_gate=input_gate,
                                                forget_gate=forget_gate,
                                                cell_gate=cell_gate,
                                                output_gate=output_gate)
