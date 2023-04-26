from typing import Any, Callable, Optional

from tsl.metrics.torch import pinball_loss
from tsl.metrics.torch.metric_base import MaskedMetric


class MaskedPinballLoss(MaskedMetric):
    """Quantile loss.

    Args:
        q (float): Target quantile.
        mask_nans (bool): Whether to automatically mask nan values.
            (default: :obj:`False`)
        mask_inf (bool): Whether to automatically mask infinite
            values.
            (default: :obj:`False`)
        compute_on_step (bool): Whether to compute the metric
            right-away or if accumulate the results. This should be :obj:`True`
            when using the metric to compute a loss function, :obj:`False` if
            the metric is used for logging the aggregate error across different
            mini-batches.
            (default: :obj:`True`)
        at (int, optional): Whether to compute the metric only w.r.t. a certain
            time step.
            (default: :obj:`None`)
        dim (int): The index of the dimension that represents time in a batch.
            Relevant only when also 'at' is defined.
            Default assumes [b t n f] format.
            (default: :obj:`1`)
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(self,
                 q: float,
                 mask_nans: bool = False,
                 mask_inf: bool = False,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Any = None,
                 dist_sync_fn: Callable = None,
                 at: Optional[int] = None,
                 dim: int = 1):
        super(MaskedPinballLoss,
              self).__init__(metric_fn=pinball_loss,
                             mask_nans=mask_nans,
                             mask_inf=mask_inf,
                             compute_on_step=compute_on_step,
                             dist_sync_on_step=dist_sync_on_step,
                             process_group=process_group,
                             dist_sync_fn=dist_sync_fn,
                             metric_fn_kwargs={'q': q},
                             at=at,
                             dim=dim)
