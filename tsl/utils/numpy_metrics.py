import numpy as np

import tsl


def mae(y_hat, y):
    return np.abs(y_hat - y).mean()


def nmae(y_hat, y):
    delta = np.max(y) - np.min(y) + 1e-8
    return mae(y_hat, y) * 100 / delta


def mape(y_hat, y):
    return 100 * np.abs((y_hat - y) / (y + 1e-8)).mean()


def mse(y_hat, y):
    return np.square(y_hat - y).mean()


def rmse(y_hat, y):
    return np.sqrt(mse(y_hat, y))


def nrmse(y_hat, y):
    delta = np.max(y) - np.min(y) + 1e-8
    return rmse(y_hat, y) * 100 / delta


def nrmse_2(y_hat, y):
    nrmse_ = np.sqrt(np.square(y_hat - y).sum() / np.square(y).sum())
    return nrmse_ * 100


def r2(y_hat, y):
    return 1. - np.square(y_hat - y).sum() / (np.square(y.mean(0) - y).sum())


def masked_mae(y_hat, y, mask=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = y_hat[mask] - y[mask]
    return np.abs(err).mean()


def masked_mape(y_hat, y, mask=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = (y_hat[mask] - y[mask]) / (y[mask] + tsl.epsilon)
    return np.abs(err).mean()


def masked_mse(y_hat, y, mask=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = y_hat[mask] - y[mask]
    return np.square(err).mean()


def masked_rmse(y_hat, y, mask=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = np.square(y_hat[mask] - y[mask])
    return np.sqrt(err.mean())


def masked_mre(y_hat, y, mask=None):
    if mask is None:
        mask = np.ones(y.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    err = np.abs(y_hat[mask] - y[mask])
    return err.sum() / (y[mask].sum() + tsl.epsilon)
