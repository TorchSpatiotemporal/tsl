from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from tsl.imports import require_optional_dependency
from tsl.nn.models.base_model import BaseModel


class TimesFMModel(BaseModel):
    r"""Adapter for Google Research TimesFM 3 forecasting models.

    Each ``[time, features]`` target is represented as one multivariate time
    series. With node inputs, ``nodes_as_covariates=False`` sends every node as
    an independent item, while ``nodes_as_covariates=True`` joins the
    node-feature axes so TimesFM's cross-variate attention can learn across
    nodes. Batch samples always remain separate items in both modes.

    Args:
        horizon (int): Number of forecasting steps.
        model_id (str, optional): Hugging Face model identifier or path to a
            local TimesFM 3 checkpoint.
            (default: :obj:`"google/timesfm-3.0-pytorch"`)
        quantile (float, optional): Native TimesFM quantile to use as the point
            forecast. If :obj:`None`, use TimesFM's point forecast.
            (default: :obj:`None`)
        return_quantiles (bool, optional): Whether :meth:`forward` returns a
            ``(prediction, quantiles)`` tuple. This can be overridden for each
            call. (default: :obj:`False`)
        quantile_levels (sequence, optional): Native TimesFM quantile levels to
            return. If :obj:`None`, return every level in the checkpoint.
            (default: :obj:`None`)
        nodes_as_covariates (bool, optional): Whether nodes within one batch
            sample are joined as variates and can attend to one another. Batch
            samples are never joined. (default: :obj:`True`)
        model_kwargs (mapping, optional): Extra keyword arguments forwarded to
            :meth:`timesfm3.TimesFM3Forecaster.from_pretrained`, such as
            ``device`` and ``cache_dir``. (default: :obj:`None`)
        forecast_kwargs (mapping, optional): Extra keyword arguments forwarded
            to :meth:`timesfm3.TimesFM3Forecaster.predict_batch`.
            (default: :obj:`None`)

    Raises:
        ImportError: If the optional ``timesfm`` package is not installed.
        ValueError: If an argument is invalid or requests a quantile not
            supplied by the checkpoint.
    """

    def __init__(
        self,
        horizon: int,
        model_id: str = 'google/timesfm-3.0-pytorch',
        quantile: Optional[float] = None,
        return_quantiles: bool = False,
        quantile_levels: Optional[Sequence[float]] = None,
        nodes_as_covariates: bool = True,
        model_kwargs: Optional[Mapping[str, Any]] = None,
        forecast_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        if horizon <= 0:
            raise ValueError(f'horizon must be positive, got {horizon}.')
        if quantile is not None and not 0 < quantile < 1:
            raise ValueError(f'quantile must be between 0 and 1, got {quantile}.')

        reserved = {
            'contexts',
            'horizon',
            'past_only_covariates',
            'return_quantiles',
        }
        forecast_kwargs = dict(forecast_kwargs or {})
        conflicting = reserved.intersection(forecast_kwargs)
        if conflicting:
            names = ', '.join(sorted(conflicting))
            raise ValueError(f'forecast_kwargs cannot override {names}.')

        timesfm = require_optional_dependency('timesfm3', 'timesfm')
        self.forecaster = timesfm.TimesFM3Forecaster.from_pretrained(
            model_id, **dict(model_kwargs or {})
        )
        inner_model = getattr(self.forecaster, 'model', None)
        if isinstance(inner_model, nn.Module):
            self.timesfm_model = inner_model

        native_levels = tuple(
            float(level) for level in self.forecaster.config.quantiles
        )
        if quantile_levels is None:
            quantile_levels = native_levels
        quantile_levels = tuple(float(level) for level in quantile_levels)
        if not quantile_levels or any(not 0 < level < 1 for level in quantile_levels):
            raise ValueError('quantile_levels must contain values between 0 and 1.')
        requested = quantile_levels + (() if quantile is None else (quantile,))
        missing = [level for level in requested if level not in native_levels]
        if missing:
            raise ValueError(
                'TimesFM can only return checkpoint quantiles; unavailable levels: '
                f'{missing}. Available levels: {list(native_levels)}.'
            )

        self.horizon = horizon
        self.model_id = model_id
        self.quantile = quantile
        self.return_quantiles = return_quantiles
        self.quantile_levels = quantile_levels
        self.native_quantile_levels = native_levels
        self.nodes_as_covariates = nodes_as_covariates
        self.forecast_kwargs = forecast_kwargs

    @staticmethod
    def _prepare_inputs(
        x: Tensor, u: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], bool]:
        """Normalize target and exogenous inputs to ``[b, t, n, f]``.

        Args:
            x (Tensor): Target history in ``[b, t, f]`` or
                ``[b, t, n, f]`` format.
            u (Tensor, optional): Exogenous history in ``[b, t, f]`` or
                ``[b, t, n, f]`` format. Node-independent values are broadcast
                to all nodes. (default: :obj:`None`)

        Returns:
            tuple: Normalized targets, normalized exogenous inputs, and whether
            the target had no explicit node dimension.

        Raises:
            ValueError: If shapes are invalid or incompatible.
        """
        if x.ndim not in (3, 4):
            raise ValueError(
                'TimesFMModel expects x with shape [batch, time, features] or '
                f'[batch, time, nodes, features], got {tuple(x.shape)}.'
            )
        squeeze_nodes = x.ndim == 3
        if squeeze_nodes:
            x = x.unsqueeze(2)
        batch, steps, nodes, _ = x.shape

        if u is None:
            return x, None, squeeze_nodes
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
                'TimesFMModel expects u with shape [batch, time, features] or '
                f'[batch, time, nodes, features], got {tuple(u.shape)}.'
            )
        return x, u, squeeze_nodes

    def _to_timesfm_layout(
        self, values: Tensor
    ) -> Tuple[Sequence[np.ndarray], int, int]:
        """Convert a TSL tensor into independent TimesFM items.

        Args:
            values (Tensor): Input in ``[batch, time, nodes, features]`` format.

        Returns:
            tuple: NumPy inputs, number of nodes, and feature count.
        """
        batch, steps, nodes, features = values.shape
        values = values.detach().float().cpu().permute(0, 2, 3, 1)
        if self.nodes_as_covariates:
            values = values.reshape(batch, nodes * features, steps)
        else:
            values = values.reshape(batch * nodes, features, steps)
        return [sample.numpy() for sample in values], nodes, features

    def _restore_layout(
        self,
        forecast: Tensor,
        batch: int,
        nodes: int,
        features: int,
        has_quantiles: bool,
    ) -> Tensor:
        """Restore TimesFM outputs to TSL layout.

        Args:
            forecast (Tensor): Forecast in TimesFM layout.
            batch (int): Batch size.
            nodes (int): Number of nodes.
            features (int): Number of target features.
            has_quantiles (bool): Whether the final axis stores quantiles.

        Returns:
            Tensor: Forecast in ``[b, h, n, f]`` or ``[b, h, n, f, q]``
            layout.
        """
        trailing = (forecast.shape[-1],) if has_quantiles else ()
        forecast = forecast.reshape(batch, nodes, features, self.horizon, *trailing)
        if has_quantiles:
            return forecast.permute(0, 3, 1, 2, 4)
        return forecast.permute(0, 3, 1, 2)

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
            u (Tensor, optional): Past exogenous input with shape
                ``[batch, time, features]`` or
                ``[batch, time, nodes, features]``. TimesFM receives these as
                past-only covariates on the same variate axis as ``x``.
                (default: :obj:`None`)
            return_quantiles (bool, optional): Override whether to return the
                quantile forecasts alongside the point forecast.
                (default: :obj:`None`)

        Returns:
            Tensor or tuple: Point forecasts, optionally paired with quantile
            forecasts. Three-dimensional inputs produce ``[b, h, f]`` and
            ``[b, h, f, q]`` outputs; four-dimensional inputs produce
            ``[b, h, n, f]`` and ``[b, h, n, f, q]`` outputs.
        """
        x_input, u_input, squeeze_nodes = self._prepare_inputs(x, u)
        batch, _, nodes, features = x_input.shape
        contexts, _, _ = self._to_timesfm_layout(x_input)
        covariates = None
        if u_input is not None:
            covariates, _, _ = self._to_timesfm_layout(u_input)

        should_return_quantiles = (
            self.return_quantiles if return_quantiles is None else return_quantiles
        )
        need_quantiles = should_return_quantiles or self.quantile is not None
        if hasattr(self, 'timesfm_model'):
            parameter = next(self.timesfm_model.parameters(), None)
            if parameter is not None:
                self.forecaster.device = parameter.device
        outputs = list(
            self.forecaster.predict_batch(
                contexts=contexts,
                horizon=self.horizon,
                past_only_covariates=covariates,
                return_quantiles=need_quantiles,
                **self.forecast_kwargs,
            )
        )
        point = torch.as_tensor(np.stack([output.forecast for output in outputs]))

        quantiles = None
        if need_quantiles:
            quantiles = torch.as_tensor(
                np.stack([output.quantiles for output in outputs])
            )
        if self.quantile is not None:
            index = self.native_quantile_levels.index(self.quantile)
            point = quantiles[..., index]

        prediction = self._restore_layout(point, batch, nodes, features, False)
        if squeeze_nodes:
            prediction = prediction.squeeze(2)
        prediction = prediction.to(device=x.device, dtype=x.dtype)
        if not should_return_quantiles:
            return prediction

        indices = [
            self.native_quantile_levels.index(level) for level in self.quantile_levels
        ]
        quantiles = quantiles[..., indices]
        quantiles = self._restore_layout(quantiles, batch, nodes, features, True)
        if squeeze_nodes:
            quantiles = quantiles.squeeze(2)
        return prediction, quantiles.to(device=x.device, dtype=x.dtype)
