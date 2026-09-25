"""Composable base classes for spatiotemporal graph neural networks."""

from typing import Dict, List, Optional, Tuple, Union

from torch import nn

from tsl.nn.layers import NodeEmbedding
from tsl.nn.models import BaseModel
from tsl.typing import Adj, Tensor

ModuleArguments = Tuple[Tuple[object, ...], Dict[str, object]]


class STGNN(BaseModel):
    """Base class for SpatioTemporal Graph Neural Networks (STGNNs) following the
    template architecture of the paper `"Taming Local Effects in Graph-based
    Spatiotemporal Forecasting" <https://arxiv.org/abs/2302.04071>`_ (Cini et al.,
    NeurIPS 2023).

    Subclasses construct the three stages by overriding :meth:`build_encoder`,
    :meth:`build_stmp_layer`, and :meth:`build_decoder`. The ``map_*_args``
    hooks adapt the model inputs to the interface of each constructed module;
    therefore subclasses need not implement :meth:`forward`. ``emb`` is the
    selected node-embedding tensor, or :obj:`None` when embeddings are disabled.

    Args:
        input_size (int): Number of input features.
        horizon (int): Forecasting horizon.
        n_nodes (int, optional): Number of global nodes. Required when
            ``emb_size`` is positive. (default: :obj:`None`)
        output_size (int, optional): Number of output features. (default:
            :obj:`None`)
        exog_size (int, optional): Number of encoder exogenous features.
            (default: :obj:`0`)
        emb_size (int, optional): Size of each learnable node embedding.
            (default: :obj:`0`)
        n_layers (int, optional): Number of STMP layers. (default: :obj:`1`)
        add_embedding_before (str or list[str], optional): Retained for
            backwards compatibility. Embedding placement is determined by the
            custom encoder and decoder. (default: :obj:`'encoding'`)
        **kwargs: Additional build-time configuration. Each item is stored as
            an attribute before the build methods are called.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int = None,
        output_size: int = None,
        exog_size: int = 0,
        emb_size: int = 0,
        n_layers: int = 1,
        add_embedding_before: Optional[Union[str, List[str]]] = 'encoding',
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.output_size = output_size or input_size
        self.exog_size = exog_size
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.add_embedding_before = add_embedding_before
        for name, value in kwargs.items():
            setattr(self, name, value)

        if emb_size > 0:
            if n_nodes is None:
                raise ValueError(
                    "'n_nodes' must be provided when emb_size is positive."
                )
            self.emb = NodeEmbedding(n_nodes, emb_size)
        else:
            self.register_module('emb', None)

        self.encoder = self.build_encoder()
        self.stmp_layers = nn.ModuleList(
            self.build_stmp_layer(layer) for layer in range(n_layers)
        )
        self.decoder = self.build_decoder()

    def build_encoder(self) -> nn.Module:
        """Build the encoder module.

        Returns:
            nn.Module: An encoder called with the arguments returned by
            :meth:`map_encoder_args` and returning hidden states with shape
            ``[B, T, N, d_h]``.

        Raises:
            NotImplementedError: Always. Subclasses must provide the encoder.
        """
        raise NotImplementedError

    def build_stmp_layer(self, layer: int) -> nn.Module:
        """Build an STMP layer.

        Args:
            layer (int): Zero-based index of the layer being constructed.

        Returns:
            nn.Module: An STMP layer called with the arguments returned by
            :meth:`map_stmp_args` and returning hidden states.

        Raises:
            NotImplementedError: Always. Subclasses must provide each layer.
        """
        raise NotImplementedError

    def build_decoder(self) -> nn.Module:
        """Build the decoder module.

        Returns:
            nn.Module: A decoder called with the arguments returned by
            :meth:`map_decoder_args` and returning a forecast with shape
            ``[B, H, N, d_out]``.

        Raises:
            NotImplementedError: Always. Subclasses must provide the decoder.
        """
        raise NotImplementedError

    @staticmethod
    def _stmp_summary(layer: nn.Module, index: int) -> List[str]:
        """Return schematic lines describing one STMP layer."""
        if not isinstance(layer, nn.ModuleDict):
            return [f'  │   [{index}] {layer.__class__.__name__}']

        lines = [f'  │   [{index}]']
        for name, modules in layer.items():
            module_names = ', '.join(module.__class__.__name__ for module in modules)
            lines.append(f'  │   ├─ {name}: {module_names or "Identity"}')
        return lines

    def summary(self, show_workflow: bool = False) -> str:
        """Return a schematic representation of the model architecture.

        Args:
            show_workflow (bool, optional): Whether to append the common
                encoder--STMP--decoder tensor workflow. (default: :obj:`False`)

        Returns:
            str: An encoder--STMP--decoder architecture tree. The summary
            displays module types rather than their parameter values.
        """
        lines = [
            f'{self.__class__.__name__} architecture',
            f'  ├─ encoder: {self.encoder.__class__.__name__}',
            f'  ├─ STMP stack ({len(self.stmp_layers)} layer'
            f'{"s" if len(self.stmp_layers) != 1 else ""})',
        ]
        for index, layer in enumerate(self.stmp_layers):
            lines.extend(self._stmp_summary(layer, index))
        lines.append(f'  └─ decoder: {self.decoder.__class__.__name__}')
        if show_workflow:
            lines.extend(
                [
                    '',
                    'Workflow:',
                    '  (x, u, v, emb) → encoder → h⁽⁰⁾',
                    '  h⁽ˡ⁾ → STMP[l] → h⁽ˡ⁺¹⁾,  l = 0, …, L - 1',
                    '  (h⁽ᴸ⁾, u_h, emb) → decoder → y_hat',
                ]
            )
        return '\n'.join(lines)

    @staticmethod
    def _call_module(
        module: nn.Module, module_arguments: ModuleArguments, hook_name: str
    ) -> Tensor:
        """Call ``module`` with arguments returned by an argument-mapping hook."""
        if not isinstance(module_arguments, tuple) or len(module_arguments) != 2:
            raise TypeError(
                f"'{hook_name}' must return a tuple containing positional and "
                "keyword arguments."
            )
        args, kwargs = module_arguments
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise TypeError(f"'{hook_name}' must return a tuple of ``(tuple, dict)``.")
        return module(*args, **kwargs)

    def map_encoder_args(
        self,
        x: Tensor,
        u: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        emb: Optional[Tensor] = None,
        **kwargs,
    ) -> ModuleArguments:
        """Map forward inputs to arguments for the encoder module.

        Override this hook to select a subset of inputs, rename them as keyword
        arguments, or concatenate them before calling the encoder. The default
        mapping preserves the original positional interface and discards forward
        keyword arguments.

        Args:
            x (Tensor): Input sequence.
            u (Tensor, optional): Encoder covariates. (default: :obj:`None`)
            v (Tensor, optional): Static attributes. (default: :obj:`None`)
            emb (Tensor, optional): Selected node embeddings. (default:
                :obj:`None`)
            **kwargs: Additional arguments passed to :meth:`forward`.

        Returns:
            tuple[tuple, dict]: Positional and keyword arguments for
            :attr:`encoder`.
        """
        return (x, u, v, emb), {}

    def map_stmp_args(
        self,
        h: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        *,
        layer: int,
        **kwargs,
    ) -> ModuleArguments:
        """Map inputs to arguments for one STMP layer.

        Override this hook to adapt the inputs for a particular layer. The
        default mapping forwards all :meth:`forward` keyword arguments to every
        STMP layer, preserving the previous behavior.

        Args:
            h (Tensor): Hidden states.
            edge_index (Adj): Graph connectivity.
            edge_weight (Tensor, optional): Edge weights. (default: :obj:`None`)
            layer (int): Zero-based STMP layer index.
            **kwargs: Additional arguments passed to :meth:`forward`.

        Returns:
            tuple[tuple, dict]: Positional and keyword arguments for the STMP
            layer at ``layer``.
        """
        return (h, edge_index, edge_weight), kwargs

    def map_decoder_args(
        self,
        h: Tensor,
        u_h: Optional[Tensor] = None,
        emb: Optional[Tensor] = None,
        **kwargs,
    ) -> ModuleArguments:
        """Map forward inputs to arguments for the decoder module.

        Override this hook to select a subset of inputs, rename them as keyword
        arguments, or combine future covariates and embeddings before calling
        the decoder. The default mapping preserves the original positional
        interface and discards forward keyword arguments.

        Args:
            h (Tensor): Final hidden states.
            u_h (Tensor, optional): Future covariates. (default: :obj:`None`)
            emb (Tensor, optional): Selected node embeddings.
                (default: :obj:`None`)
            **kwargs: Additional arguments passed to :meth:`forward`.

        Returns:
            tuple[tuple, dict]: Positional and keyword arguments for
            :attr:`decoder`.
        """
        return (h, u_h, emb), {}

    def stmp(
        self,
        h: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Apply the constructed STMP stack.

        Args:
            h (Tensor): Hidden states with shape ``[B, T, N, d_h]``.
            edge_index (Adj): Graph connectivity.
            edge_weight (Tensor, optional): Edge weights. (default: :obj:`None`)
            **kwargs: Additional arguments available to :meth:`map_stmp_args`.

        Returns:
            Tensor: Updated hidden states.

        Shapes:
            h: :math:`(B, T, N, F_h)`, where :math:`B` is the batch size,
                :math:`T` is the input-window length, :math:`N` is the number
                of nodes, and :math:`F_h` is the hidden size.
            edge_index: Graph connectivity, for example COO indices with shape
                :math:`(2, E)`, where :math:`E` is the number of edges.
            edge_weight: Optional edge weights with shape :math:`(E,)`.
            return: :math:`(B, T, N, F_h)`.
        """
        for layer_index, layer in enumerate(self.stmp_layers):
            layer_args = self.map_stmp_args(
                h, edge_index, edge_weight, layer=layer_index, **kwargs
            )
            h = self._call_module(layer, layer_args, 'map_stmp_args')
        return h

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        u_h: Optional[Tensor] = None,
        node_idx: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Compute a forecast following the encoder--STMP--decoder workflow.

        Args:
            x (Tensor): Input sequence with shape ``[B, T, N, d_in]``.
            edge_index (Adj): Graph connectivity.
            edge_weight (Tensor, optional): Edge weights. (default: :obj:`None`)
            u (Tensor, optional): Encoder covariates with shape
                ``[B, T, N, d]`` or ``[B, T, d]``. (default: :obj:`None`)
            v (Tensor, optional): Static attributes with shape
                ``[B, N, d]`` or ``[B, d]``. (default: :obj:`None`)
            u_h (Tensor, optional): Future covariates with shape
                ``[B, H, N, d]`` or ``[B, H, d]``. (default: :obj:`None`)
            node_idx (Tensor, optional): Global node indices used to select
                embeddings. (default: :obj:`None`)
            **kwargs: Additional arguments available to the argument-mapping
                hooks.

        Returns:
            Tensor: Forecast with shape ``[B, H, N, d_out]``.

        Shapes:
            x: :math:`(B, T, N, F_{in})`, where :math:`B` is the batch size,
                :math:`T` is the input-window length, :math:`N` is the number
                of nodes, and :math:`F_{in}` is the input size.
            edge_index: Graph connectivity, for example COO indices with shape
                :math:`(2, E)`, where :math:`E` is the number of edges.
            edge_weight: Optional edge weights with shape :math:`(E,)`.
            u: Optional encoder covariates with shape :math:`(B, T, N, F_u)`
                or :math:`(B, T, F_u)`.
            v: Optional static attributes with shape :math:`(B, N, F_v)` or
                :math:`(B, F_v)`.
            u_h: Optional future covariates with shape :math:`(B, H, N, F_c)` or
                :math:`(B, H, F_c)`, where :math:`H` is the horizon.
            node_idx: Optional global-node indices with shape :math:`(N,)`.
            return: :math:`(B, H, N, F_{out})`, where :math:`F_{out}` is the
                output size.
        """
        emb = self.emb(node_index=node_idx) if self.emb is not None else None
        encoder_args = self.map_encoder_args(x, u, v, emb, **kwargs)
        h = self._call_module(self.encoder, encoder_args, 'map_encoder_args')
        h = self.stmp(h, edge_index, edge_weight, **kwargs)
        decoder_args = self.map_decoder_args(h, u_h, emb, **kwargs)
        return self._call_module(self.decoder, decoder_args, 'map_decoder_args')


class DisjointSTGNN(STGNN):
    """STGNN with each STMP layer split into temporal then spatial stacks.

    Args:
        input_size (int): Number of input features.
        horizon (int): Forecasting horizon.
        n_nodes (int, optional): Number of global nodes. (default: :obj:`None`)
        output_size (int, optional): Number of output features. (default:
            :obj:`None`)
        exog_size (int, optional): Number of encoder exogenous features.
            (default: :obj:`0`)
        emb_size (int, optional): Size of node embeddings. (default: :obj:`0`)
        n_layers (int, optional): Number of STMP layers. (default: :obj:`1`)
        n_temporal_layers (int, optional): Number of temporal layers in each
            STMP layer. (default: :obj:`1`)
        n_spatial_layers (int, optional): Number of spatial layers in each STMP
            layer. (default: :obj:`1`)
        add_embedding_before (str or list[str], optional): Retained for
            backwards compatibility. (default: :obj:`'encoding'`)
        **kwargs: Additional build-time configuration.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int = None,
        output_size: int = None,
        exog_size: int = 0,
        emb_size: int = 0,
        n_layers: int = 1,
        n_temporal_layers: int = 1,
        n_spatial_layers: int = 1,
        add_embedding_before: Optional[Union[str, List[str]]] = 'encoding',
        **kwargs,
    ):
        self.n_temporal_layers = n_temporal_layers
        self.n_spatial_layers = n_spatial_layers
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            emb_size=emb_size,
            n_layers=n_layers,
            add_embedding_before=add_embedding_before,
            **kwargs,
        )

    def build_stmp_layer(self, layer: int) -> nn.Module:
        """Build the temporal and spatial sublayers for one STMP layer.

        Args:
            layer (int): Zero-based outer STMP layer index.

        Returns:
            nn.Module: A registry containing the temporal and spatial sublayers.
        """
        return nn.ModuleDict(
            temporal=nn.ModuleList(
                self.build_tmp_layer(layer, inner_layer)
                for inner_layer in range(self.n_temporal_layers)
            ),
            spatial=nn.ModuleList(
                self.build_smp_layer(layer, inner_layer)
                for inner_layer in range(self.n_spatial_layers)
            ),
        )

    def build_tmp_layer(self, layer: int, inner_layer: int) -> nn.Module:
        """Build one temporal sublayer.

        Args:
            layer (int): Zero-based outer STMP layer index.
            inner_layer (int): Zero-based temporal sublayer index.

        Returns:
            nn.Module: A temporal sublayer called with the arguments returned
            by :meth:`map_tmp_args`.
        """
        raise NotImplementedError

    def build_smp_layer(self, layer: int, inner_layer: int) -> nn.Module:
        """Build one spatial sublayer.

        Args:
            layer (int): Zero-based outer STMP layer index.
            inner_layer (int): Zero-based spatial sublayer index.

        Returns:
            nn.Module: A spatial sublayer called with the arguments returned by
            :meth:`map_smp_args`.
        """
        raise NotImplementedError

    def map_tmp_args(
        self,
        h: Tensor,
        *,
        layer: int,
        inner_layer: int,
        **kwargs,
    ) -> ModuleArguments:
        """Map inputs to arguments for one temporal sublayer.

        Args:
            h (Tensor): Hidden states.
            layer (int): Zero-based outer STMP layer index.
            inner_layer (int): Zero-based temporal sublayer index.
            **kwargs: Additional arguments passed to :meth:`forward`.

        Returns:
            tuple[tuple, dict]: Positional and keyword arguments for the
            temporal sublayer.
        """
        return (h,), {}

    def tmp(self, h: Tensor, *, layer: int, inner_layer: int, **kwargs) -> Tensor:
        """Apply one constructed temporal sublayer.

        Args:
            h (Tensor): Hidden states.
            layer (int): Outer STMP layer index.
            inner_layer (int): Temporal sublayer index.
            **kwargs: Additional arguments passed to :meth:`forward`.

        Returns:
            Tensor: Updated hidden states.

        Shapes:
            h: :math:`(B, T, N, F_h)`, where :math:`B` is the batch size,
                :math:`T` is the input-window length, :math:`N` is the number
                of nodes, and :math:`F_h` is the hidden size.
            return: :math:`(B, T, N, F_h)`.
        """
        layer_args = self.map_tmp_args(
            h, layer=layer, inner_layer=inner_layer, **kwargs
        )
        return self._call_module(
            self.stmp_layers[layer]['temporal'][inner_layer],
            layer_args,
            'map_tmp_args',
        )

    def map_smp_args(
        self,
        h: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        *,
        layer: int,
        inner_layer: int,
        **kwargs,
    ) -> ModuleArguments:
        """Map inputs to arguments for one spatial sublayer.

        Args:
            h (Tensor): Hidden states.
            edge_index (Adj): Graph connectivity.
            edge_weight (Tensor, optional): Edge weights. (default: :obj:`None`)
            layer (int): Zero-based outer STMP layer index.
            inner_layer (int): Zero-based spatial sublayer index.
            **kwargs: Additional arguments passed to :meth:`forward`.

        Returns:
            tuple[tuple, dict]: Positional and keyword arguments for the spatial
            sublayer.
        """
        return (h, edge_index, edge_weight), kwargs

    def smp(
        self,
        h: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        *,
        layer: int,
        inner_layer: int,
        **kwargs,
    ) -> Tensor:
        """Apply one constructed spatial sublayer.

        Args:
            h (Tensor): Hidden states.
            edge_index (Adj): Graph connectivity.
            edge_weight (Tensor, optional): Edge weights. (default: :obj:`None`)
            layer (int): Outer STMP layer index.
            inner_layer (int): Spatial sublayer index.
            **kwargs: Additional arguments available to :meth:`map_smp_args`.

        Returns:
            Tensor: Updated hidden states.

        Shapes:
            h: :math:`(B, T, N, F_h)`, where :math:`B` is the batch size,
                :math:`T` is the input-window length, :math:`N` is the number
                of nodes, and :math:`F_h` is the hidden size.
            edge_index: Graph connectivity, for example COO indices with shape
                :math:`(2, E)`, where :math:`E` is the number of edges.
            edge_weight: Optional edge weights with shape :math:`(E,)`.
            return: :math:`(B, T, N, F_h)`.
        """
        layer_args = self.map_smp_args(
            h,
            edge_index,
            edge_weight,
            layer=layer,
            inner_layer=inner_layer,
            **kwargs,
        )
        return self._call_module(
            self.stmp_layers[layer]['spatial'][inner_layer],
            layer_args,
            'map_smp_args',
        )

    def stmp(
        self,
        h: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Apply each temporal stack followed by its spatial stack.

        Args:
            h (Tensor): Hidden states.
            edge_index (Adj): Graph connectivity.
            edge_weight (Tensor, optional): Edge weights. (default: :obj:`None`)
            **kwargs: Additional arguments available to :meth:`map_tmp_args` and
                :meth:`map_smp_args`.

        Returns:
            Tensor: Updated hidden states.

        Shapes:
            h: :math:`(B, T, N, F_h)`, where :math:`B` is the batch size,
                :math:`T` is the input-window length, :math:`N` is the number
                of nodes, and :math:`F_h` is the hidden size.
            edge_index: Graph connectivity, for example COO indices with shape
                :math:`(2, E)`, where :math:`E` is the number of edges.
            edge_weight: Optional edge weights with shape :math:`(E,)`.
            return: :math:`(B, T, N, F_h)`.
        """
        for layer in range(self.n_layers):
            for inner_layer in range(self.n_temporal_layers):
                h = self.tmp(h, layer=layer, inner_layer=inner_layer, **kwargs)
            for inner_layer in range(self.n_spatial_layers):
                h = self.smp(
                    h,
                    edge_index,
                    edge_weight,
                    layer=layer,
                    inner_layer=inner_layer,
                    **kwargs,
                )
        return h


class TimeThenSpace(DisjointSTGNN):
    """Disjoint STGNN composed from one temporal and multiple spatial modules.

    Args:
        input_size (int): Number of input features.
        horizon (int): Forecasting horizon.
        n_nodes (int, optional): Number of global nodes. (default: :obj:`None`)
        output_size (int, optional): Number of output features. (default:
            :obj:`None`)
        exog_size (int, optional): Number of encoder exogenous features.
            (default: :obj:`0`)
        emb_size (int, optional): Size of node embeddings. (default: :obj:`0`)
        n_temporal_layers (int, optional): Number of temporal layers in each
            STMP layer. (default: :obj:`1`)
        n_spatial_layers (int, optional): Number of spatial layers in each STMP
            layer. (default: :obj:`1`)
        add_embedding_before (str or list[str], optional): Retained for
            backwards compatibility. (default: :obj:`'encoding'`)
        **kwargs: Additional build-time configuration.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_nodes: int = None,
        output_size: int = None,
        exog_size: int = 0,
        emb_size: int = 0,
        n_temporal_layers: int = 1,
        n_spatial_layers: int = 1,
        add_embedding_before: Optional[Union[str, List[str]]] = 'encoding',
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            emb_size=emb_size,
            n_layers=1,
            n_temporal_layers=n_temporal_layers,
            n_spatial_layers=n_spatial_layers,
            add_embedding_before=add_embedding_before,
            **kwargs,
        )
