import torch

__all__ = [
    'mape'
]


def mape(y_hat, y):
    return torch.abs((y_hat - y) / y)
