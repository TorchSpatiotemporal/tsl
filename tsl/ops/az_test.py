from collections import namedtuple

import numpy as np
import scipy.stats

from tsl.typing import Optional, OptTensArray, TensArray, Tuple, Union


def _twosided_std_gaussian_pval(stat: float):
    """Return the two-sided p-value associated with statistic `stat`
    distributed as a standard Gaussian."""
    return 2 * (1 - scipy.stats.norm.cdf(np.abs(stat)))


def _to_numpy(o: Union[TensArray, list, int, float, None]):
    """Cast the object `o` to `numpy.ndarray` or a `float`. If it is `None`,
    it will be left as `None`."""
    if isinstance(o, np.ndarray):
        return o
    if isinstance(o, list):
        return np.array(o)
    if isinstance(o, int) or isinstance(o, float):
        return float(o)
    if o is None:
        return o
    import torch
    if isinstance(o, torch.Tensor):
        return o.numpy()
    raise NotImplementedError(
        f"I don't know how to convert {type(o)} to numpy")


def _to_undirected_no_selfloops(
    edge_index: np.ndarray, edge_weight: Optional[Union[np.ndarray, int,
                                                        float]]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Remove self-loops, make the graph undirected, remove duplicated edges,
    and sum the weights corresponding to duplicated edges; it works with
    `numpy.ndarray`."""

    # Check inputs
    assert edge_index.shape[1] > 0
    if isinstance(edge_weight, int) or isinstance(edge_weight, float):
        edge_weight = edge_weight * np.ones(edge_index.shape[1])
    if edge_weight is not None:
        assert edge_index.shape[1] == edge_weight.shape[0]

    # Remove self-loops
    selfloop_mask = edge_index[0] != edge_index[1]
    edge_index_ = edge_index[:, selfloop_mask]
    edge_weight_ = edge_weight[selfloop_mask]

    # Sort edges to have an undirected graph
    edge_index_.sort(axis=0)
    order = np.lexsort(edge_index_)
    # edges
    edge_index_ = edge_index_[:, order]
    # weights
    if edge_weight is not None:
        edge_weight_ = edge_weight_[order]

    # Mask of unique edges (or the first of the duplicates) and non self-loops
    unique_mask = np.any(edge_index_[:, 1:] != edge_index_[:, :-1], axis=0)
    unique_mask = np.append(True, unique_mask)
    unique_mask_inds, = np.nonzero(unique_mask)
    # edges
    edge_index_ = edge_index_[:, unique_mask]
    # weights
    if edge_weight_ is not None:
        edge_weight_ = np.add.reduceat(edge_weight_,
                                       unique_mask_inds,
                                       dtype=edge_weight_.dtype)

    return edge_index_, edge_weight_


AZWhitenessTestResult = namedtuple('AZWhitenessTestResult',
                                   ('statistic', 'pvalue'))
AZWhitenessMultiTestResult = namedtuple(
    'AZWhitenessMultiTestResult',
    ('statistic', 'pvalue', 'componentwise_tests'))


def az_whiteness_test(
    x: TensArray,
    edge_index: TensArray,
    mask: OptTensArray = None,
    pattern: str = "t n f",
    edge_weight: Optional[Union[TensArray, float]] = None,
    edge_weight_temporal: Optional[float] = None,
    lamb: float = 0.5,
    multivariate: bool = False,
    remove_median: bool = False
) -> Union[AZWhitenessTestResult, AZWhitenessMultiTestResult]:
    """Implementation of the AZ-whiteness test from the paper `"AZ-whiteness
    test: a test for uncorrelated noise on spatio-temporal graphs"
    <https://arxiv.org/abs/2204.11135>`_ (D. Zambon and C. Alippi,
    NeurIPS 2022).

    Args:
        x (TensArray): graph signal, typically with pattern "t n f" and
            representing the prediction residuals.
        edge_index (TensArray): indices of the spatial edges with shape (2, E).
            Current implementation supports only a static topology.
        mask (TensArray, optional): boolean mask of signal :obj:`x`, with same
            size of :obj:`x`. The mask is :obj:`True` where the observations in
            :obj:`x` are valid and :obj:`False` otherwise.
            (default: :obj:`None`)
        pattern (str): string encoding the index pattern of `x`, typically
            "t n f" representing time, nodes and node features dimensions,
            respectively.
            (default: :obj:`"t n f"`)
        edge_weight (TensArray or float, optional): positive weights of the
            spatial edges. It can be a :obj:`TensArray` of shape (E,), or a
            scalar value (same weight for all edges).
            (default: :obj:`None`)
        edge_weight_temporal (float, optional): positive scalar weight for all
            temporal edges. If :obj:`None` or :obj:`"auto"`, the weight is
            computed to balance the contribution of the spatial and temporal
            components (see `Zambon and Alippi, 2022
            <https://arxiv.org/abs/2204.11135>`_).
            (default: :obj:`None`)
        lamb (float, optional): scalar factor in within :math:`0.0` and
            :math:`1.0` defining a convex combination of the spatial and
            temporal components; if :obj:`lamb == 1.0` the test is applied on
            the spatial topology only, for :obj:`lamb == 0.0` only the serial
            correlation is considered.
            (default: :obj:`0.5`)
        multivariate (bool): whether to run a single test on a multivariate
            signal or combine multiple scalar tests, one for each of the
            :obj:`f` features. It applies only when :obj:`f > 1`.
            (default: :obj:`False`)
        remove_median (bool): whether to manually fulfill --- where possible ---
            the assumption of null median or not.
            (default: :obj:`False`)

    Returns:
        AZWhitenessTestResult or AZWhitenessMultiTestResult: The test
        statistics.
    """

    # retrieve pattern
    dims = pattern.strip().split(' ')
    T_DIM, N_DIM, F_DIM = dims.index("t"), dims.index("n"), dims.index("f")

    # data to numpy.ndarray
    x = _to_numpy(x)
    assert x.ndim == 3
    mask = _to_numpy(mask)
    edge_index_spatial = _to_numpy(edge_index)
    edge_weight_spatial = _to_numpy(edge_weight)

    if remove_median:
        x_ = x.copy()
        x_[np.logical_not(mask)] = np.nan
        x_median = np.nanmedian(x_, axis=[T_DIM, N_DIM], keepdims=True)
        x -= x_median

    F = x.shape[F_DIM]
    if F == 1:
        multivariate = True

    az_test_args = dict(x=x,
                        mask=mask,
                        pattern=pattern,
                        edge_index_spatial=edge_index_spatial,
                        edge_weight_spatial=edge_weight_spatial,
                        edge_weight_temporal=edge_weight_temporal,
                        lamb=lamb)

    if multivariate:
        # Single test with edge statistic: `sign( (xu * xv).sum() )`.
        return _az_whiteness_test(**az_test_args)
    else:
        # Multiple scalar tests based on `sign( xu[f] * xv[f] )`, i.e., a test
        # for each feature dimension.
        res = []
        for f in range(F):
            x_ = x[..., f:f + 1]
            if mask is None:
                mask_ = None
            else:
                mask_ = mask[..., f:f + 1]
            az_test_args["x"] = x_
            az_test_args["mask"] = mask_
            res.append(_az_whiteness_test(**az_test_args))
            C_multi = np.sum([r.statistic for r in res]) / np.sqrt(len(res))
            pval = _twosided_std_gaussian_pval(C_multi)
        return AZWhitenessMultiTestResult(C_multi, pval, res)


def _az_whiteness_test(x, mask, pattern, edge_index_spatial,
                       edge_weight_spatial, edge_weight_temporal, lamb):
    """Core computation of the AZ-whiteness test.

    All parameters are assumed to be `numpy.ndarray` or `float`.
    """

    # retrieve pattern
    dims = pattern.strip().split(' ')
    T_DIM, N_DIM, F_DIM = dims.index("t"), dims.index("n"), dims.index("f")
    T, N = x.shape[T_DIM], x.shape[N_DIM]

    # --- Spatial edges and weight ---

    # Parse weight
    if edge_weight_spatial is None:
        edge_weight_spatial = 1.0
    if (isinstance(edge_weight_spatial, int)
            or isinstance(edge_weight_spatial, float)):
        edge_weight_spatial = edge_weight_spatial * np.ones(
            edge_index_spatial.shape[1])

    # Check dims
    assert edge_weight_spatial.shape[0] == edge_index_spatial.shape[
        1], "Dimension mismatch between edge_weight and edge_index."
    assert np.all(edge_weight_spatial > 0), \
        "Edge weights are not all positive."
    assert N == edge_index_spatial.max() + 1, \
        "Is the input signal given with pattern (T, N, F)?"

    # Make the graph undirected and without self-loops
    edge_index_spatial, edge_weight_spatial = _to_undirected_no_selfloops(
        edge_index=edge_index_spatial, edge_weight=edge_weight_spatial)

    # Node mask
    if mask is None:
        mask = np.ones_like(x)
    mask = mask.astype(int)
    assert np.all(np.logical_or(mask == 0, mask == 1))
    mask_node = mask.max(axis=F_DIM)
    # Mask data
    x = x * mask
    # Edge mask:
    #  - repeat node mask for every source node
    #  - repeat node mask for every target node
    #  - compare the two
    mask_edge_spatial = np.where(
        np.logical_and(mask_node[:, edge_index_spatial[0]],
                       mask_node[:, edge_index_spatial[1]]))

    # Spatial normalization factor
    # sums over all unmasked edges (it considers already the dynamic graph
    # with all "repeated" edges)
    W_spatial = np.sum(edge_weight_spatial[mask_edge_spatial[1]]**2)

    # --- Temporal edges and weight ---

    # Parse temporal edge weight
    if T == 1:
        num_temporal_edge_masked = 0
        edge_weight_temporal = 1
    else:
        assert T_DIM == 0
        # num of temporal edges
        num_temporal_edge_masked = (mask[1:] * mask[:-1]).sum()
        # default temporal weight
        if edge_weight_temporal == "auto" or edge_weight_temporal is None:
            edge_weight_temporal = np.sqrt(W_spatial /
                                           num_temporal_edge_masked)
    assert isinstance(edge_weight_temporal, int) or isinstance(
        edge_weight_temporal, float)
    assert edge_weight_temporal > 0

    # Temporal normalization factor
    W_temporal = (edge_weight_temporal**2) * num_temporal_edge_masked

    # --- Test statistics ---

    # Inner products
    assert T_DIM == 0 and F_DIM == 2
    # (T, E, F) * (T, E, F) -> (T, E, F)
    xxs = x[:, edge_index_spatial[0]] * x[:, edge_index_spatial[1]]
    # (T, E, F) -> (T, E)
    xxs = xxs.sum(axis=F_DIM)
    # (T-1, N, F) * (T-1, N, F) -> (T-1, N, F)
    xxt = x[1:] * x[:-1]
    # (T-1, N, F) -> (T-1, N)
    xxt = xxt.sum(axis=F_DIM)

    # Weighted signs and Ctilde
    # (1, E) * (T, E) -> (T, E)
    w_sgn_xxs = edge_weight_spatial[None, ...] * np.sign(xxs)
    Ctilde_spatial = w_sgn_xxs.sum()
    sgn_xxt = np.sign(xxt)
    Ctilde_temporal = edge_weight_temporal * sgn_xxt.sum()

    # Normalize Ctilde: C
    assert 0 <= lamb <= 1
    Ctilde = lamb * Ctilde_spatial + (1 - lamb) * Ctilde_temporal
    W = (lamb**2) * W_spatial + ((1 - lamb)**2) * W_temporal
    C = Ctilde / np.sqrt(W)

    # p-value
    pval = _twosided_std_gaussian_pval(C)
    return AZWhitenessTestResult(C, pval)
