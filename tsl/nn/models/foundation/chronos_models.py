from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from tsl.imports import require_optional_dependency
from tsl.nn.models.base_model import BaseModel


class ChronosModel(BaseModel):
    r"""Adapter for Amazon Chronos pretrained forecasting models.

    Chronos and Chronos-Bolt process each node-feature series independently.
    Chronos-2 instead keeps features together as variates and can optionally
    keep all nodes of a sample together as well. Different batch samples are
    always independent Chronos-2 tasks to prevent cross-sample leakage.

    Args:
        horizon (int): Number of forecasting steps.
        model_id (str, optional): Hugging Face model identifier or path to a
            local Chronos checkpoint.
            (default: :obj:`"amazon/chronos-bolt-small"`)
        quantile (float, optional): Quantile to use as the point forecast. If
            :obj:`None`, use the point forecast supplied by Chronos.
            (default: :obj:`None`)
        return_quantiles (bool, optional): Whether :meth:`forward` returns a
            ``(prediction, quantiles)`` tuple. This can be overridden for each
            call. (default: :obj:`False`)
        quantile_levels (sequence, optional): Quantile levels returned when
            ``return_quantiles`` is enabled. By default, use levels from
            :obj:`0.1` through :obj:`0.9` in increments of :obj:`0.1`.
            (default: :obj:`None`)
        nodes_as_covariates (bool, optional): For Chronos-2, whether node-level
            multivariate series within the same batch sample can attend to one
            another. Every node is represented as one ``[features, time]``
            item, and each batch sample is always predicted in a separate call
            to prevent cross-sample leakage. Chronos and Chronos-Bolt do not
            support cross-learning. (default: :obj:`True`)
        pipeline_kwargs (mapping, optional): Extra keyword arguments forwarded
            to :meth:`chronos.BaseChronosPipeline.from_pretrained`, such as
            ``device_map`` and ``torch_dtype``.
            (default: :obj:`None`)

    Raises:
        ImportError: If the optional ``chronos-forecasting`` package is not
            installed.
        ValueError: If an argument is outside its valid range.
    """

    def __init__(
        self,
        horizon: int,
        model_id: str = 'amazon/chronos-bolt-small',
        quantile: Optional[float] = None,
        return_quantiles: bool = False,
        quantile_levels: Optional[Sequence[float]] = None,
        nodes_as_covariates: bool = True,
        pipeline_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        if horizon <= 0:
            raise ValueError(f'horizon must be positive, got {horizon}.')
        if quantile is not None and not 0 < quantile < 1:
            raise ValueError(f'quantile must be between 0 and 1, got {quantile}.')

        if quantile_levels is None:
            quantile_levels = [level / 10 for level in range(1, 10)]
        quantile_levels = tuple(float(level) for level in quantile_levels)
        if not quantile_levels or any(not 0 < level < 1 for level in quantile_levels):
            raise ValueError('quantile_levels must contain values between 0 and 1.')

        chronos = require_optional_dependency('chronos', 'chronos-forecasting')
        self.pipeline = chronos.BaseChronosPipeline.from_pretrained(
            model_id, **dict(pipeline_kwargs or {})
        )

        # BaseChronosPipeline is not an nn.Module. Register its inner model so
        # state/device operations performed by Predictor still reach the model.
        inner_model = getattr(self.pipeline, 'inner_model', None)
        if isinstance(inner_model, nn.Module):
            self.chronos_model = inner_model

        self.is_chronos2 = any(
            cls.__name__ == 'Chronos2Pipeline' for cls in type(self.pipeline).__mro__
        )
        self.horizon = horizon
        self.model_id = model_id
        self.quantile = quantile
        self.return_quantiles = return_quantiles
        self.quantile_levels = quantile_levels
        self.nodes_as_covariates = nodes_as_covariates

    @staticmethod
    def _stack_forecasts(
        forecasts: Union[Tensor, Sequence[Tensor]],
    ) -> Tensor:
        """Convert Chronos forecast containers to a tensor.

        Args:
            forecasts: A forecast tensor or sequence of forecast tensors.

        Returns:
            Tensor: Stacked forecasts.
        """
        if isinstance(forecasts, Tensor):
            return forecasts
        return torch.stack(list(forecasts), dim=0)

    @staticmethod
    def _prepare_inputs(x: Tensor, u: Optional[Tensor]) -> Tuple[Tensor, int, bool]:
        """Normalize target and exogenous inputs to ``[b, t, n, f]``.

        Args:
            x (Tensor): Target history in ``[b, t, f]`` or
                ``[b, t, n, f]`` format.
            u (Tensor, optional): Exogenous history in ``[b, t, f]`` or
                ``[b, t, n, f]`` format. Node-independent values are broadcast
                to all nodes. (default: :obj:`None`)

        Returns:
            tuple: The concatenated input, number of target features, and
            whether the input had no explicit node dimension.

        Raises:
            ValueError: If shapes are invalid or incompatible.
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                'ChronosModel expects x with shape [batch, time, features] or '
                f'[batch, time, nodes, features], got {tuple(x.shape)}.'
            )

        squeeze_nodes = x.ndim == 3
        if squeeze_nodes:
            x = x.unsqueeze(2)
        batch, steps, nodes, target_features = x.shape

        if u is None:
            return x, target_features, squeeze_nodes
        if u.ndim == 3:
            if u.shape[:2] != (batch, steps):
                raise ValueError('x and u must have matching batch and time axes.')
            u = u.unsqueeze(2).expand(-1, -1, nodes, -1)
        elif u.ndim == 4:
            if u.shape[:3] != (batch, steps, nodes):
                raise ValueError(
                    'x and u must have matching batch, time, and node axes.'
                )
        else:
            raise ValueError(
                'ChronosModel expects u with shape [batch, time, features] or '
                f'[batch, time, nodes, features], got {tuple(u.shape)}.'
            )
        return torch.cat([x, u], dim=-1), target_features, squeeze_nodes

    def _restore_layout(
        self,
        forecast: Tensor,
        batch: int,
        nodes: int,
        features: int,
        target_features: int,
        has_quantiles: bool,
    ) -> Tensor:
        """Restore a Chronos forecast to TSL layout and remove exogenous rows.

        Args:
            forecast (Tensor): Forecast in Chronos layout.
            batch (int): Batch size.
            nodes (int): Number of nodes.
            features (int): Total target and exogenous feature count.
            target_features (int): Number of target features.
            has_quantiles (bool): Whether the final axis stores quantiles.

        Returns:
            Tensor: Forecast in ``[b, h, n, f]`` or ``[b, h, n, f, q]``
            layout.
        """
        trailing = (forecast.shape[-1],) if has_quantiles else ()
        forecast = forecast.reshape(batch, nodes, features, self.horizon, *trailing)
        if has_quantiles:
            forecast = forecast.permute(0, 3, 1, 2, 4)
            return forecast[..., :target_features, :]
        forecast = forecast.permute(0, 3, 1, 2)
        return forecast[..., :target_features]

    def _predict_chronos2(
        self,
        x: Tensor,
        quantile_levels: Sequence[float],
    ) -> Tuple[Tensor, Tensor]:
        """Predict Chronos-2 samples without crossing batch boundaries.

        Each node is passed as one multivariate ``[features, time]`` item. A
        separate pipeline call is made for every batch sample, so enabling
        cross-learning can only share information among nodes of that sample.

        Args:
            x (Tensor): Input in ``[batch, time, nodes, features]`` format.
            quantile_levels (sequence): Quantiles requested from Chronos-2.

        Returns:
            tuple: Quantile and point forecasts in Chronos layout.
        """
        quantile_batches = []
        point_batches = []
        for sample in x:
            node_series = [node for node in sample.permute(1, 2, 0)]
            quantiles, point = self.pipeline.predict_quantiles(
                node_series,
                prediction_length=self.horizon,
                quantile_levels=list(quantile_levels),
                cross_learning=self.nodes_as_covariates,
            )
            quantile_batches.append(self._stack_forecasts(quantiles))
            point_batches.append(self._stack_forecasts(point))
        return torch.stack(quantile_batches), torch.stack(point_batches)

    def forward(
        self,
        x: Tensor,
        u: Optional[Tensor] = None,
        return_quantiles: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forecast a batch of temporal or spatiotemporal series.

        Args:
            x (Tensor): Input with shape ``[batch, time, features]`` or
                ``[batch, time, nodes, features]``.
            u (Tensor, optional): Exogenous input with shape
                ``[batch, time, features]`` or
                ``[batch, time, nodes, features]``. (default: :obj:`None`)
            return_quantiles (bool, optional): Override whether to return the
                quantile forecasts alongside the point forecast.
                (default: :obj:`None`)

        Returns:
            Tensor or tuple: Point forecasts, optionally paired with quantile
            forecasts. Three-dimensional inputs produce ``[b, h, f]`` and
            ``[b, h, f, q]`` outputs; four-dimensional inputs produce
            ``[b, h, n, f]`` and ``[b, h, n, f, q]`` outputs.
        """
        x_input, target_features, squeeze_nodes = self._prepare_inputs(x, u)
        batch, steps, nodes, features = x_input.shape

        if not self.is_chronos2:
            pipeline_input = x_input.permute(0, 2, 3, 1).reshape(
                batch * nodes * features, steps
            )

        should_return_quantiles = (
            self.return_quantiles if return_quantiles is None else return_quantiles
        )
        output_levels = self.quantile_levels if should_return_quantiles else ()
        requested_levels = set(output_levels)
        requested_levels.add(self.quantile if self.quantile is not None else 0.5)
        requested_levels = sorted(requested_levels)
        if self.is_chronos2:
            quantiles, point_forecast = self._predict_chronos2(
                x_input, requested_levels
            )
        else:
            quantiles, point_forecast = self.pipeline.predict_quantiles(
                pipeline_input,
                prediction_length=self.horizon,
                quantile_levels=requested_levels,
            )
            quantiles = self._stack_forecasts(quantiles)

        if self.quantile is None:
            prediction = self._stack_forecasts(point_forecast)
        else:
            prediction = quantiles[..., requested_levels.index(self.quantile)]

        prediction = self._restore_layout(
            prediction, batch, nodes, features, target_features, False
        )
        if squeeze_nodes:
            prediction = prediction.squeeze(2)
        if not should_return_quantiles:
            return prediction.to(device=x.device, dtype=x.dtype)

        output_indices = [
            requested_levels.index(level) for level in self.quantile_levels
        ]
        quantiles = quantiles[..., output_indices]
        quantiles = self._restore_layout(
            quantiles, batch, nodes, features, target_features, True
        )
        if squeeze_nodes:
            quantiles = quantiles.squeeze(2)

        prediction = prediction.to(device=x.device, dtype=x.dtype)
        quantiles = quantiles.to(device=x.device, dtype=x.dtype)
        return prediction, quantiles
