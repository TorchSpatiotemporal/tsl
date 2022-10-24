import torch
from tsl.utils import ensure_list

__all__ = [
    'mape',
    'pinball_loss',
    'multi_quantile_pinball_loss'
]


def mape(y_hat, y):
    return torch.abs((y_hat - y) / y)


def pinball_loss(y_hat, y, q):
    err = y - y_hat
    return torch.maximum((q - 1) * err, q * err)


def multi_quantile_pinball_loss(y_hat, y, q):
    q = ensure_list(q)
    assert y_hat.size(0) == len(q)
    loss = torch.zeros_like(y_hat)
    for i, qi in enumerate(q):
        loss += pinball_loss(y_hat[i], y, qi)
    return loss
