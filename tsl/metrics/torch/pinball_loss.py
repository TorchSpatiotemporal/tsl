from tsl.metrics.torch import pinball_loss
from tsl.metrics.torch.metric_base import MaskedMetric


class MaskedPinballLoss(MaskedMetric):
    """Quantile loss.

    Args:
        q (float): Target quantile.
        mask_nans (bool, optional): Whether to automatically mask nan values.
        mask_inf (bool, optional): Whether to automatically mask infinite
            values.
        compute_on_step (bool, optional): Whether to compute the metric
            right-away or if accumulate the results. This should be :obj:`True`
            when using the metric to compute a loss function, :obj:`False` if
            the metric is used for logging the aggregate error across different
            mini-batches.
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 q,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedPinballLoss,
              self).__init__(metric_fn=pinball_loss,
                             mask_nans=mask_nans,
                             mask_inf=mask_inf,
                             compute_on_step=compute_on_step,
                             dist_sync_on_step=dist_sync_on_step,
                             process_group=process_group,
                             dist_sync_fn=dist_sync_fn,
                             metric_fn_kwargs={'q': q},
                             at=at)
