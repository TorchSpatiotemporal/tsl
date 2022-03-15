from tsl.nn.metrics.metric_base import MaskedMetric
from tsl.utils.python_utils import ensure_list


class MaskedMetricWrapper(MaskedMetric):
    def __init__(self,
                 metric: MaskedMetric,
                 input_preprocessing=None,
                 target_preprocessing=None,
                 mask_preprocessing=None):
        super(MaskedMetricWrapper, self).__init__(None)
        self.metric = metric

        if input_preprocessing is None:
            input_preprocessing = lambda x: x

        if target_preprocessing is None:
            target_preprocessing = lambda x: x

        if mask_preprocessing is None:
            mask_preprocessing = lambda x: x

        self.input_preprocessing = input_preprocessing
        self.target_preprocessing = target_preprocessing
        self.mask_preprocessing = mask_preprocessing

    def update(self, y_hat, y, mask=None):
        y_hat = self.input_preprocessing(y_hat)
        y = self.target_preprocessing(y)
        mask = self.mask_preprocessing(mask)
        return self.metric.update(y_hat, y, mask)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
        super(MaskedMetricWrapper, self).reset()


class SplitMetricWrapper(MaskedMetricWrapper):
    def __init__(self, metric, input_idx=None, target_idx=None, mask_idx=None):
        if input_idx is not None:
            input_preprocessing = lambda x: x[input_idx]
        else:
            input_preprocessing = None

        if target_idx is not None:
            target_preprocessing = lambda x: x[target_idx]
        else:
            target_preprocessing = None

        if mask_idx is not None:
            map_preprocessing = lambda x: x[mask_idx]
        else:
            map_preprocessing = None
        super(SplitMetricWrapper, self).__init__(metric,
                                                 input_preprocessing=input_preprocessing,
                                                 target_preprocessing=target_preprocessing,
                                                 mask_preprocessing=map_preprocessing)


class ChannelSplitMetricWrapper(MaskedMetricWrapper):
    def __init__(self, metric, input_channels=None, target_channels=None, map_channels=None):
        if input_channels is not None:
            input_preprocessing = lambda x: x[..., ensure_list(input_channels)]
        else:
            input_preprocessing = None

        if target_channels is not None:
            target_preprocessing = lambda x: x[..., ensure_list(target_channels)]
        else:
            target_preprocessing = None

        if map_channels is not None:
            map_preprocessing = lambda x: x[..., ensure_list(map_channels)]
        else:
            map_preprocessing = None
        super(ChannelSplitMetricWrapper, self).__init__(metric,
                                                        input_preprocessing=input_preprocessing,
                                                        target_preprocessing=target_preprocessing,
                                                        mask_preprocessing=map_preprocessing)
