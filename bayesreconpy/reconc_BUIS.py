import numpy as np
from scipy import stats
from scipy.stats import norm
from bayesreconpy.utils import DEFAULT_PARS, _check_input_BUIS, _check_weights, _resample, _distr_sample, _distr_pmf
from bayesreconpy.hierarchy import _check_hierarchical, _get_HG
from KDEpy import FFTKDE
from typing import List, Dict, Union

def _check_hierfamily_rel(sh_res, distr, debug=False):
    bottom_idxs = sh_res['bottom_idxs']
    upper_idxs = sh_res['upper_idxs']
    A = sh_res['A']

    for bi in range(len(bottom_idxs)):
        distr_bottom = distr[bottom_idxs[bi]]
        rel_upper_i = A[:, bi]
        rel_distr_upper = [distr[i] for i in range(A.shape[0]) if rel_upper_i[i] == 1]

        err_message = "A continuous bottom distribution cannot be child of a discrete one."

        if distr_bottom == "continuous":
            if "discrete" in rel_distr_upper or any(d in DEFAULT_PARS['DISCR_DISTR'] for d in rel_distr_upper):
                if debug:
                    return -1
                else:
                    raise ValueError(err_message)

        if distr_bottom in DEFAULT_PARS['CONT_DISTR']:
            if "discrete" in rel_distr_upper or any(d in DEFAULT_PARS['DISCR_DISTR'] for d in rel_distr_upper):
                if debug:
                    return -1
                else:
                    raise ValueError(err_message)

    if debug:
        return 0



def _emp_pmf(l, density_samples):
    # Compute the empirical PMF from the density samples
    values, counts = np.unique(density_samples, return_counts=True)
    empirical_pmf = dict(zip(values, counts / len(density_samples)))

    # Extract PMF values for the indices specified in `l`
    w = np.array([empirical_pmf.get(i, 0) for i in l])
    return w


def _compute_weights(b, u, in_type_, distr_):
    if in_type_ == "samples":
        if distr_ == "discrete":
            # Discrete samples
            w = _emp_pmf(b, u)
        elif distr_ == "continuous":
            # KDE (Kernel Density Estimation)
            fftkde = FFTKDE(kernel='gaussian', bw='ISJ').fit(u)
            grid_lims = [np.minimum(np.min(u), np.min(b)), np.maximum(np.max(u), np.max(b))]
            grid = np.linspace(grid_lims[0] - 1e-3, grid_lims[1] + 1e-3, 1000)
            res = fftkde.evaluate(grid)
            w = np.interp(b, grid, res, )


        # Ensure no NA values are returned, replace with 0
        w[np.isnan(w)] = 0
    elif in_type_ == "params":
        # This function needs to be defined separately
        w = _distr_pmf(b, u, distr_)

    # Ensure not to return all 0 weights, return ones instead
    if np.sum(w) == 0:
        w = np.ones_like(w)

    return w



def reconc_buis(
    A: np.ndarray,
    base_forecasts: List[Union[Dict[str, float], np.ndarray]],
    in_type: Union[str, List[str]],
    distr: Union[str, List[str]],
    num_samples: int = 20000,
    suppress_warnings: bool = False,
    seed: Union[int, None] = None
) -> Dict[str, np.ndarray]:
    """
    BUIS for Probabilistic Reconciliation of Forecasts via Conditioning

    Uses the Bottom-Up Importance Sampling (BUIS) algorithm to draw samples from the reconciled
    forecast distribution obtained via conditioning.

    Parameters
    ----------
    A : numpy.ndarray
        Aggregation matrix with shape `(n_upper, n_bottom)`. Each column represents a bottom-level
        forecast, and each row represents an upper-level forecast. A value of `1` indicates a
        contribution of a bottom-level forecast to an upper-level forecast.

    base_forecasts : list of dict or numpy.ndarray
        A list containing `n_upper + n_bottom` elements. The first `n_upper` elements represent
        the upper-level base forecasts (in the order of rows in `A`), and the remaining elements
        represent the bottom-level base forecasts (in the order of columns in `A`).

        - If `in_type[i] == "samples"`, `base_forecasts[i]` is a NumPy array containing samples
          from the base forecast distribution.

        - If `in_type[i] == "params"`, `base_forecasts[i]` is a dictionary containing parameters
          for the specified distribution in `distr[i]`:

          * `'gaussian'`: {"mean": float, "sd": float}
          * `'poisson'`: {"lambda": float}
          * `'nbinom'`: {"size": float, "prob": float} or {"size": float, "mu": float}

    in_type : str or list of str
        Specifies the input type for each base forecast. If a string, the same input type is applied
        to all forecasts. If a list, `in_type[i]` specifies the type for the `i`-th forecast:

        - `'samples'`: The forecast is provided as samples.
        - `'params'`: The forecast is provided as parameters.

    distr : str or list of str
        Specifies the distribution type for each base forecast. If a string, the same distribution
        is applied to all forecasts. If a list, `distr[i]` specifies the distribution for the `i`-th forecast:

        - `'continuous'` or `'discrete'` if `in_type[i] == "samples"`.
        - `'gaussian'`, `'poisson'`, or `'nbinom'` if `in_type[i] == "params"`.

    num_samples : int, optional
        Number of samples to draw from the reconciled distribution. Ignored if `in_type == "samples"`,
        in which case the number of reconciled samples matches the number of samples in the base forecasts.
        Default is 20,000.

    suppress_warnings : bool, optional
        Whether to suppress warnings during the importance sampling step. Default is `False`.

    seed : int or None, optional
        Random seed for reproducibility. Default is `None`.

    Returns
    -------
    dict
        A dictionary containing the reconciled forecasts:

        - `'bottom_reconciled_samples'`: numpy.ndarray
            A matrix of shape `(n_bottom, num_samples)` containing the reconciled samples
            for the bottom-level forecasts.
        - `'upper_reconciled_samples'`: numpy.ndarray
            A matrix of shape `(n_upper, num_samples)` containing the reconciled samples
            for the upper-level forecasts.
        - `'reconciled_samples'`: numpy.ndarray
            A matrix of shape `(n, num_samples)` containing the reconciled samples
            for all forecasts.

    Notes
    -----
    - Warnings are triggered during the importance sampling step if:

      * All weights are zero (the corresponding upper forecast is ignored).
      * Effective sample size is less than 200.
      * Effective sample size is less than 1% of the total number of samples.
    - Such warnings indicate potential issues with the base forecasts. Check the inputs in case of warnings.

    Examples
    --------
    Example 1: Gaussian base forecasts (in params form)
        >>> A = np.array([
        ...     [1, 0, 0],
        ...     [0, 1, 1]
        ... ])
        >>> base_forecasts = [
        ...     {"mean": 9.0, "sd": 3.0},  # Upper forecast
        ...     {"mean": 2.0, "sd": 2.0},  # Bottom forecast 1
        ...     {"mean": 4.0, "sd": 2.0}   # Bottom forecast 2
        ... ]
        >>> result = reconc_buis(A, base_forecasts, in_type="params", distr="gaussian", num_samples=10000, seed=42)
        >>> print(result['reconciled_samples'].shape)
        (3, 10000)

    Example 2: Poisson base forecasts (in params form)
        >>> A = np.array([
        ...     [1, 0, 0],
        ...     [0, 1, 1]
        ... ])
        >>> base_forecasts = [
        ...     {"lambda": 9.0},  # Upper forecast
        ...     {"lambda": 2.0},  # Bottom forecast 1
        ...     {"lambda": 4.0}   # Bottom forecast 2
        ... ]
        >>> result = reconc_buis(A, base_forecasts, in_type="params", distr="poisson", num_samples=10000, seed=42)
        >>> print(result['reconciled_samples'].shape)
        (3, 10000)

    """
    if seed is not None:
        np.random.seed(seed)

    n_upper = A.shape[0]
    n_bottom = A.shape[1]
    n_tot = len(base_forecasts)

    # Transform distr and in_type into lists
    if isinstance(distr, str):
        distr = np.tile(distr, n_tot).tolist()
    if isinstance(in_type, str):
        in_type = np.tile(in_type, n_tot).tolist()

    # Ensure that data inputs are valid
    _check_input_BUIS(A, base_forecasts, in_type, distr)

    # Split bottoms and uppers
    upper_base_forecasts = base_forecasts[:n_upper]
    bottom_base_forecasts = base_forecasts[n_upper:]

    split_hierarchy_res = {
        'A': A,
        'upper': upper_base_forecasts,
        'bottom': bottom_base_forecasts,
        'upper_idxs': list(range(n_upper)),
        'bottom_idxs': list(range(n_upper, n_tot))
    }

    # Check on continuous/discrete in relationship to the hierarchy
    _check_hierfamily_rel(split_hierarchy_res, distr)

    # H, G
    is_hier = _check_hierarchical(A)
    if is_hier:
        H = A
        G = None
        upper_base_forecasts_H = upper_base_forecasts
        upper_base_forecasts_G = None
        in_typeH = [in_type[i] for i in split_hierarchy_res['upper_idxs']]
        distr_H = [distr[i] for i in split_hierarchy_res['upper_idxs']]
        in_typeG = None
        distr_G = None
    else:
        get_HG_res = _get_HG(A, upper_base_forecasts, [distr[i] for i in split_hierarchy_res['upper_idxs']],
                            [in_type[i] for i in split_hierarchy_res['upper_idxs']])
        H = get_HG_res['H']
        upper_base_forecasts_H = get_HG_res['Hv']
        G = get_HG_res['G']
        upper_base_forecasts_G = get_HG_res['Gv']
        in_typeH = get_HG_res['Hin_type']
        distr_H = get_HG_res['Hdistr']
        in_typeG = get_HG_res['Gin_type']
        distr_G = get_HG_res['Gdistr']

    # Reconciliation using BUIS
    # 1. Bottom samples
    B = []
    in_type_bottom = [in_type[i] for i in split_hierarchy_res['bottom_idxs']]

    for bi in range(n_bottom):
        if in_type_bottom[bi] == "samples":
            B.append(np.array(bottom_base_forecasts[bi]))
        elif in_type_bottom[bi] == "params":
            B.append(
                _distr_sample(bottom_base_forecasts[bi], np.array(distr)[split_hierarchy_res['bottom_idxs']][bi], num_samples))

    B = np.column_stack(B)  # B is a matrix (num_samples x n_bottom)

    # Bottom-Up IS on the hierarchical part
    for hi in range(H.shape[0]):
        c = H[hi, :]
        b_mask = (c != 0)
        weights = _compute_weights(
            b=np.dot(B, c),
            u=upper_base_forecasts_H[hi],
            in_type_=in_typeH[hi],
            distr_=distr_H[hi]
        )
        check_weights_res = _check_weights(weights)
        if check_weights_res['warning'] and not suppress_warnings:
            warning_msg = check_weights_res['warning_msg']
            upper_fromA_i = [i for i, row in enumerate(A) if np.allclose(row, c)]
            for wmsg in warning_msg:
                wmsg = f"{wmsg} Check the upper forecast at index: {upper_fromA_i}."
                print(f"Warning: {wmsg}")
        if check_weights_res['warning'] and 1 in check_weights_res['warning_code']:
            continue

        B[:, b_mask] = _resample(B[:, b_mask], weights)

    if G is not None:
        # Plain IS on the additional constraints
        weights = np.ones(B.shape[0])
        for gi in range(G.shape[0]):
            c = G[gi, :]
            weights *= _compute_weights(
                b=np.dot(B, c),
                u=upper_base_forecasts_G[gi],
                in_type_=in_typeG[gi],
                distr_=distr_G[gi]
            )
        check_weights_res = _check_weights(weights)
        if check_weights_res['warning'] and not suppress_warnings:
            warning_msg = check_weights_res['warning_msg']
            upper_fromA_i = [i for i in range(n_upper) for gi in range(G.shape[0]) if np.allclose(A[i, :], G[gi, :])]
            for wmsg in warning_msg:
                wmsg = f"{wmsg} Check the upper forecasts at index: {{{', '.join(map(str, upper_fromA_i))}}}."
                print(f"Warning: {wmsg}")
        if not (check_weights_res['warning'] and 1 in check_weights_res['warning_code']):
            B = _resample(B, weights)

    B = B.T
    U = np.dot(A, B)
    Y_reconc = np.vstack([U, B])

    return {
        'bottom_reconciled_samples': B,
        'upper_reconciled_samples': U,
        'reconciled_samples': Y_reconc
    }

