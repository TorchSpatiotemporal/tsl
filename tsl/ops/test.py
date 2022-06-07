from collections import namedtuple
import scipy.stats
import numpy as np
from torch_geometric.utils import to_undirected, remove_self_loops
from tsl.typing import TensArray, OptTensArray, Optional, Union

def _twosided_std_gaussian_pval(stat):
    """Return the two-sided p-value associated with value `stat` of a standard 
    Gaussian statistic."""
    return 2 * (1 - scipy.stats.norm.cdf(np.abs(stat)))

def _to_numpy(o):
    """Cast the object `o` to `numpy.ndarray` or `None` if it is `None`."""
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
    raise NotImplementedError(f"I don't know how to convert {type(o)} to numpy")

AZWhitenessTestResult = namedtuple('AZWhitenessTestResult', ('statistic', 'pvalue'))
AZWhitenessMultiTestResult = namedtuple('AZWhitenessMultiTestResult', ('statistic', 'pvalue', 'componentwise_tests'))

def az_whiteness_test(x: TensArray, mask: OptTensArray = None, pattern: str = "T N F",
                      multivariate: bool = False, remove_median: bool = False, 
                      **kwargs): 
                      -> Union[AZWhitenessTestResult, AZWhitenessMultiTestResult]
    """
    AZ-whiteness test by Zambon and Alippi 2022 [1].

    Parameters:
     - `remove_median`: whether to manually fulfill --- where possible --- the assumption 
            of null median or not.
     - `multivariate`: whether to run a single test on a multivariate signal or combine 
            multiple scalar tests, one for each of the `F` features; it applies only 
            when `F>1`.
    For `x`, `mask`, `pattern` and parameters in `kwargs` see `_az_whiteness_test()`.

    [1] Zambon, Daniele, and Cesare Alippi. "AZ-whiteness test: a test for uncorrelated noise on spatio-temporal graphs." arXiv preprint arXiv:2204.11135 (2022).
    """

    # retrieve pattern
    dims = pattern.strip().split(' ')
    T_DIM, N_DIM, F_DIM = dims.index("T"), dims.index("N"), dims.index("F")

    if remove_median:
        # be careful that when the estimated median is not
        # accurate it can lead to false alarms; this happens
        # for example when T=1, N<<100, F>>10.
        x = _to_numpy(x)
        mask = _to_numpy(mask)
        x_ = x + 0.0 
        x_[mask] = np.nan
        x_median = np.nanmedian(x_, axis=F_DIM)
        x -= x_median

    F = x.shape[F_DIM]
    if F == 1:
        multivariate = True

    if multivariate:
        # Single test with edge statistic: `sign( (xu * xv).sum() )`.
        return _az_whiteness_test(x=x, mask=mask, pattern=pattern, **kwargs)
    else:
        # Multiple scalar tests based on `sign( xu[f] * xv[f] )`; a test for each feature dimension.
        res = []
        for f in range(F):
            x_ = x[..., f:f + 1]
            if mask is None:
                mask_ = None
            else:
                mask_ = mask[..., f:f + 1]
            res.append(_az_whiteness_test(x=x_, mask=mask_, pattern=pattern, **kwargs))
            C_multi = np.sum([r.statistic for r in res]) / np.sqrt(len(res))
            pval = twosided_std_gaussian_pval(C_multi)
        return AZWhitenessMultiTestResult(C_multi, pval, res)
           
def _az_whiteness_test(x: TensArray, mask: OptTensArray = None, pattern: str = "T N F", 
                       edge_index: TensArray, edge_weight: Optional[TensArray, float] = None,
                       edge_weight_temporal: Optional[float] = None,
                       lamb: float = 0.5):
                       -> Union[AZWhitenessTestResult, AZWhitenessMultiTestResult]
    """Core computation of the AZ-whiteness test.

    All parameters are assumed to be `numpy.ndarray` or `float`, and with pattern "T N F".
    """

    # --- Check datatypes and patterns

    x = _to_numpy(x)
    mask = _to_numpy(mask)
    edge_index_spatial = _to_numpy(edge_index)
    edge_weight_spatial = _to_numpy(edge_weight)

    dims = pattern.strip().split(' ')
    T_DIM, N_DIM, F_DIM = dims.index("T"), dims.index("N"), dims.index("F")
    T, N, F = x.shape[T_DIM], x.shape[N_DIM], x.shape[F_DIM]

    # --- Spatial edges and weight

    # Parse weight
    if edge_weight_spatial is None:
        edge_weight_spatial = 1.0
    if isinstance(edge_weight_spatial, int) or isinstance(edge_weight_spatial, float):
        edge_weight_spatial = edge_weight_spatial * np.ones(edge_index_spatial.shape[1])
    
    # Check dims
    assert edge_weight_spatial.shape[0] == edge_index_spatial.shape[1], "Dimension mismatch between edge_weight and edge_index."
    assert np.all(edge_weight_spatial > 0), "Edge weights are not all positive."
    assert N == edge_index_spatial.max() + 1, "Is the input signal given with pattern (T, N, F)?"
    
    # Make the graph undirected and without self-loops
    edge_index_spatial, edge_weight_spatial = to_undirected(edge_index=edge_index_spatial, 
                                                            edge_attr=edge_weight_spatial,
                                                            num_nodes=N, 
                                                            reduce="add")
    edge_index_spatial, edge_weight_spatial = remove_self_loops(edge_index=edge_index_spatial, 
                                                                edge_attr=edge_weight_spatial)

    # Node mask
    if mask is None:
        mask = np.ones_like(x)
    mask = mask.astype(int)
    assert np.all(np.logical_or(mask == 0, mask ==1))
    mask_node = mask.max(axis=F_DIM)
    # Mask data
    x = x * mask
    # Edge mask:  
    #  - repeat node mask for every source node
    #  - repeat node mask for every target node
    #  - compare the two
    mask_edge_spatial = np.where(np.logical_and(
                             mask_node[:, edge_index_spatial[0]],
                             mask_node[:, edge_index_spatial[1]]))

    # Spatial normalization factor
    #sums over all non masked edges (it considers already the dynamic graph with all "repeated" edges)
    W2_spatial = np.sum(edge_weight_spatial[mask_edge_spatial[1]]**2)

    # --- Temporal edges and weight

    # Parse temporal edge weight
    if T == 1:
        num_temporal_edge_masked = 0
        edge_weight_temporal = 1
    else:
        assert T_DIM == 0
        #num of temporal edges
        num_temporal_edge_masked = (mask[1:] * mask[:-1]).sum()
        #default temporal weight
        if edge_weight_temporal == "auto" or edge_weight_temporal is None:
            edge_weight_temporal = np.sqrt(W2_spatial / num_temporal_edge_masked)
    assert isinstance(edge_weight_temporal, int) or isinstance(edge_weight_temporal, float)
    assert edge_weight_temporal > 0

    # Temporal normalization factor
    W2_temporal = (edge_weight_temporal ** 2) * num_temporal_edge_masked


    # --- Test statistics
 
    # Inner products
    assert T_DIM == 0 and F_DIM == 2
    xxs = x[:, edge_index_spatial[0]] * x[:, edge_index_spatial[1]]  # (T, E, F) * (T, E, F) -> (T, E, F)
    xxs = xxs.sum(axis=F_DIM)       # (T, E, F) -> (T, E)
    xxt = x[1:] * x[:-1]            # (T-1, N, F) * (T-1, N, F) -> (T-1, N, F)
    xxt = xxt.sum(axis=F_DIM)       # (T-1, N, F) -> (T-1, N)

    # Weighted signs and Ctilde
    w_sgn_xxs = edge_weight_spatial[None, ...] * np.sign(xxs)  # (1, E) * (T, E) -> (T, E)
    Ctilde_spatial = w_sgn_xxs.sum()
    sgn_xxt = np.sign(xxt)
    Ctilde_temporal = edge_weight_temporal * sgn_xxt.sum()

    # Normalize Ctilde: C
    assert 0 <= lamb <= 1
    Ctilde = lamb * Ctilde_spatial + (1-lamb) * Ctilde_temporal
    W2 = (lamb**2) * W2_spatial + ((1-lamb)**2) * W2_temporal
    C = Ctilde / np.sqrt(W2)

    # p-value
    pval = twosided_std_gaussian_pval(C)
    return AZWhitenessTestResult(C, pval)
