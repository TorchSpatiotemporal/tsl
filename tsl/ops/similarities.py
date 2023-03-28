import numpy as np
import pandas as pd

import tsl
from tsl.typing import FrameArray


def _pearson_sim_matrix(unbiased_x, norms):
    n_samples = unbiased_x.shape[0]
    res = np.zeros(shape=(n_samples, n_samples))
    for i in range(n_samples):
        corr = (unbiased_x[i] @ unbiased_x[i + 1:].T) / (
            norms[i] * norms[i + 1:] + 1e-8)
        res[i, i + 1:] = corr
    return res + res.T + np.identity(n_samples)


def pearson_sim_matrix(X):
    unbiased_x = X - X.mean(1, keepdims=True)
    norms = np.linalg.norm(unbiased_x, axis=1)
    return _pearson_sim_matrix(unbiased_x, norms)


def correntropy(x, period, mask=None, gamma=0.05):
    """Computes similarity matrix by looking at the similarity of windows of
    length `period` using correntropy.

    See Liu et al., "Correntropy: Properties and Applications in Non-Gaussian
    Signal Processing", TSP 2007.

    Args:
        x: Input series.
        period: Length of window.
        mask: Missing value mask.
        gamma: Width of the kernel

    Returns:
        The similarity matrix.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if mask is None:
        mask = 1 - np.isnan(x, dtype='uint8')
        mask = mask[..., None]

    sim = np.zeros((x.shape[1], x.shape[1]))
    tot = np.zeros_like(sim)
    for i in range(period, len(x), period):
        xi = x[i - period:i].T
        m = mask[i - period:i].min(0)
        si = rbf_kernel(xi, gamma=gamma)
        m = m * m.T
        si = si * m
        sim += si
        tot += m
    return sim / (tot + tsl.epsilon)


def gaussian_kernel(x, theta=None):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    return weights


def geographical_distance(x: FrameArray, to_rad: bool = True):
    """Compute the as-the-crow-flies distance between every pair of samples in
    :obj:`x`. The first dimension of each point is assumed to be the latitude,
    the second is the longitude. The inputs is assumed to be in degrees. If it
    is not the case, :obj:`to_rad` must be set to :obj:`False`. The dimension of
    the data must be 2.

    Args:
        x (pd.DataFrame or np.ndarray): Array-like structure of shape :math:`(N,
            2)`.
        to_rad (bool): Whether to convert inputs to radians (provided that they
            are in degrees). (default :obj:`True`)

    Returns:
        The distance between the points in kilometers. The type is the same as
        :obj:`x`.
    """
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame
    # Assume it is 2-dim array_like of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    from sklearn.metrics.pairwise import haversine_distances
    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def top_k(matrix, k, include_self=False, keep_values=False):
    """Find the top :obj:`k` values for each row.

    Args:
        matrix: 2-dimensional array-like input.
        k (int): Number of values to keep.
        include_self (bool): Whether to include corresponding row (only if
            :obj:`matrix` is square). (default: :obj:`False`)
        keep_values (bool): Whether to keep the original values or to return a
            binary matrix with 1 in the top-k values. (default: :obj:`False`)
    """
    dim = matrix.shape[1]
    if not include_self:
        assert len(set(matrix.shape)) == 1
        matrix = matrix - np.diag([np.inf] * dim).astype(matrix.dtype)
    non_topk = np.argpartition(matrix, -k)[:, :-k]
    knn_matrix = matrix.copy() if keep_values else np.ones_like(matrix)
    knn_matrix[np.arange(dim).reshape(-1, 1), non_topk] = 0
    return knn_matrix


def thresholded_gaussian_kernel(x,
                                theta=None,
                                threshold=None,
                                threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights
