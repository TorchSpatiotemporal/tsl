from typing import Any

from torch.nn import Identity

from ...nn.layers import Select
from .metric_base import MaskedMetric


class MaskedMetricWrapper(MaskedMetric):

    def __init__(self,
                 metric: MaskedMetric,
                 input_preprocessing=None,
                 target_preprocessing=None,
                 mask_preprocessing=None):
        super(MaskedMetricWrapper, self).__init__(None)
        self.metric = metric

        if input_preprocessing is None:
            input_preprocessing = Identity

        if target_preprocessing is None:
            target_preprocessing = Identity

        if mask_preprocessing is None:
            mask_preprocessing = Identity

        self.input_preprocessing = input_preprocessing
        self.target_preprocessing = target_preprocessing
        self.mask_preprocessing = mask_preprocessing

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.metric(*args, **kwargs)

    def update(self, y_hat, y, mask=None):
        y_hat = self.input_preprocessing(y_hat)
        y = self.target_preprocessing(y)
        if mask is not None:
            mask = self.mask_preprocessing(mask)
        return self.metric.update(y_hat, y, mask)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


class SelectMetricWrapper(MaskedMetricWrapper):

    def __init__(self,
                 metric,
                 dim,
                 input_idx=None,
                 target_idx=None,
                 mask_idx=None):
        if input_idx is not None:
            input_preprocessing = Select(dim, input_idx)
        else:
            input_preprocessing = None

        if target_idx is not None:
            target_preprocessing = Select(dim, target_idx)
        else:
            target_preprocessing = None

        if mask_idx is not None:
            mask_preprocessing = Select(dim, mask_idx)
        else:
            mask_preprocessing = None
        super(SelectMetricWrapper,
              self).__init__(metric,
                             input_preprocessing=input_preprocessing,
                             target_preprocessing=target_preprocessing,
                             mask_preprocessing=mask_preprocessing)
