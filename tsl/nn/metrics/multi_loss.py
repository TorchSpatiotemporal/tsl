import torch
from tsl.nn.metrics.metric_base import MaskedMetric
import torch.nn as nn

class MaskedMultiLoss(MaskedMetric):
    r"""
    Adapted from: https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/metrics.py
    Metric that can be used to combine multiple metrics.

    Args:
        metrics: List of metrics.
        weights (optional): List of weights for the corresponding metrics.
    """
    def __init__(self, metrics, weights=None):
        super().__init__(None, compute_on_step=True)
        assert len(metrics) > 0, "at least one metric has to be specified"
        if weights is None:
            weights = [1.0 for _ in metrics]
        assert len(weights) == len(metrics), "Number of weights has to match number of metrics"

        self.metrics = nn.ModuleList(metrics)
        self.weights = weights

    def __repr__(self):
        name = (
            f"{self.__class__.__name__}("
            + ", ".join([f"{w:.3g} * {repr(m)}" if w != 1.0 else repr(m) for w, m in zip(self.weights, self.metrics)])
            + ")"
        )
        return name

    def __iter__(self):
        """
        Iterate over metrics.
        """
        return iter(self.metrics)

    def __len__(self) -> int:
        """
        Number of metrics.
        Returns:
            int: number of metrics
        """
        return len(self.metrics)

    def update(self, y_hat: torch.Tensor, y: torch.Tensor, mask=None):
        """
        Update composite metric
        Args:
            y_hat: network output
            y: actual values
        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        assert len(self) == y_hat.size(0)
        for idx, metric in enumerate(self.metrics):
            metric.update(y_hat[idx], y, mask)

    def compute(self) -> torch.Tensor:
        """
        Get metric
        Returns:
            torch.Tensor: metric
        """
        results = []
        for weight, metric in zip(self.weights, self.metrics):
            results.append(metric.compute() * weight)

        if len(results) == 1:
            results = results[0]
        else:
            results = torch.stack(results, dim=0).sum(0)
        return results

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()
        super(MaskedMultiLoss, self).reset()

    def __getitem__(self, idx: int):
        """
        Return metric.
        Args:
            idx (int): metric index
        """
        return self.metrics[idx]
