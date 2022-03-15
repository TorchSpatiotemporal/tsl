import torch
from .metric_base import MaskedMetric
from tsl.utils.python_utils import ensure_list


def _pinball_loss(y_hat, y, q):
    err = y - y_hat
    return torch.maximum((q - 1) * err, q * err)

def _multi_quantile_pinball_loss(y_hat, y, q):
    q = ensure_list(q)
    assert y_hat.size(0) == len(q)
    loss = torch.zeros_like(y_hat)
    for i, qi in enumerate(q):
        loss += _pinball_loss(y_hat[i], y, qi)
    return loss


class MaskedPinballLoss(MaskedMetric):
    """
        Quantile loss.

        Args:
            q (float): Target quantile.
            mask_nans (bool, optional): Whether to automatically mask nan values.
            mask_inf (bool, optional): Whether to automatically mask infinite values.
            compute_on_step (bool, optional): Whether to compute the metric right-away or if accumulate the results.
                             This should be `True` when using the metric to compute a loss function, `False` if the metric
                             is used for logging the aggregate error across different minibatches.
            at (int, optional): Whether to compute the metric only w.r.t. a certain time step.
    """
    def __init__(self,
                 q,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedPinballLoss, self).__init__(metric_fn=_pinball_loss,
                                                mask_nans=mask_nans,
                                                mask_inf=mask_inf,
                                                compute_on_step=compute_on_step,
                                                dist_sync_on_step=dist_sync_on_step,
                                                process_group=process_group,
                                                dist_sync_fn=dist_sync_fn,
                                                metric_kwargs={'q': q},
                                                at=at)
