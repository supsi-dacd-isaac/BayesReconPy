import numpy as np
from bayesreconpy.utils import DEFAULT_PARS, _check_input_TD
from bayesreconpy.hierarchy import _lowest_lev, _get_Au
from bayesreconpy.reconc_gaussian import reconc_gaussian
from bayesreconpy.PMF import _pmf_from_samples, _pmf_from_params, _pmf_check_support, _pmf_bottom_up
from bayesreconpy.utils import _MVN_sample
from typing import Union, Optional, Dict, List, Tuple

def _cond_biv_sampling(u, pmf1, pmf2):
    # Initialize switch flag
    pmf1 = np.atleast_1d(pmf1)
    pmf2 = np.atleast_1d(pmf2)
    sw = False
    if len(pmf1) > len(pmf2):
        pmf1, pmf2 = pmf2, pmf1
        sw = True

    b1 = np.full(len(u), np.nan)  # Initialize empty array

    for u_uniq in np.unique(u):  # Loop over unique values of u
        len_supp1 = len(pmf1)
        supp1 = np.arange(len_supp1)  # Support for pmf1
        p1 = pmf1

        supp2 = int(u_uniq) - supp1
        #supp2[supp2 < 0] = np.zero  # Trick to get NaN when accessing pmf2 outside the support
        # Handle out-of-bounds access
        p2 = np.array([pmf2[i] if i < len(pmf2) and i >= 0 else 0 for i in supp2])

        # Normalize probabilities
        p = p1 * p2
        p /= p.sum()

        # Sample from supp1 based on the probability vector p
        u_posit = (u == u_uniq)
        b1[u_posit] = np.random.choice(supp1, size=u_posit.sum(), replace=True, p=p)

    if sw:
        b1 = u - b1  # If we switched pmf1 and pmf2, switch back

    return b1, u - b1


def _TD_sampling(u, bott_pmf, toll=DEFAULT_PARS['TOL'], rtol=DEFAULT_PARS['RTOL'], smoothing=True,
                al_smooth=DEFAULT_PARS['ALPHA_SMOOTHING'], lap_smooth=DEFAULT_PARS['LAP_SMOOTHING']):
    if len(bott_pmf) == 1:
        return np.tile(u, (1, 1))

    l_l_pmf = _pmf_bottom_up(bott_pmf, toll=toll, rtol=rtol, return_all=True,
                            smoothing=smoothing, alpha_smooth=al_smooth, laplace_smooth=lap_smooth)
    l_l_pmf = l_l_pmf[::-1] # flipping the list sa that we skip the upper pmf

    b_old = np.reshape(u, (1, -1))
    for l_pmf in l_l_pmf[1:]:
        L = len(l_pmf)
        b_new = np.empty((L, len(u)))
        for j in range(L // 2):
            b = _cond_biv_sampling(b_old[j, :], l_pmf[2 * j], l_pmf[2 * j + 1])
            b_new[2 * j, :] = b[0]
            b_new[2 * j + 1, :] = b[1]
        if L % 2 == 1:
            b_new[L - 1, :] = b_old[L // 2, :]
        b_old = b_new

    return b_new


def reconc_td_cond(
    A: np.ndarray,
    fc_bottom: Union[np.array, dict],
    fc_upper: dict,
    bottom_in_type: str = "pmf",
    distr: Optional[str] = None,
    num_samples: int = 20000,
    return_type: str = "pmf",
    suppress_warnings: bool = False,
    seed: Optional[int] = None,
    return_num_samples_ok: bool = False
) -> Union[Dict[str, Dict[str, Union[List, np.ndarray]]], Tuple[Dict, int]]:
    """
    Probabilistic forecast reconciliation of mixed hierarchies via top-down conditioning.

    Uses the top-down conditioning algorithm to draw samples from the reconciled forecast distribution.
    Reconciliation is performed in two steps:
    1. Upper base forecasts are reconciled via conditioning using the hierarchical constraints between upper variables.
    2. Bottom distributions are updated via a probabilistic top-down procedure.

    Parameters
    ----------
    A : numpy.ndarray
        Aggregation matrix with shape `(n_upper, n_bottom)`. Each column represents a bottom-level
        forecast, and each row represents an upper-level forecast. A value of `1` in `A[i, j]` indicates
        that bottom-level forecast `j` contributes to upper-level forecast `i`.

    fc_bottom : Union[np.array, dict]
        Base bottom forecasts. The format depends on `bottom_in_type`:

        - If `bottom_in_type == "pmf"`: A dictionary where each value is a PMF object (probability mass function).
        - If `bottom_in_type == "samples"`: A dictionary or array where each value contains samples.
        - If `bottom_in_type == "params"`: A dictionary where each value contains parameters:

            * `'poisson'`: {"lambda": float}
            * `'nbinom'`: {"size": float, "prob": float} or {"size": float, "mu": float}

    fc_upper : dict
        Base upper forecasts, represented as parameters of a multivariate Gaussian distribution:

        - `'mu'`: A vector of length `n_upper` containing the means.
        - `'Sigma'`: A covariance matrix of shape `(n_upper, n_upper)`.

    bottom_in_type : str, optional
        Specifies the type of the base bottom forecasts. Possible values are:

        - `'pmf'`: Bottom base forecasts are provided as PMF objects.
        - `'samples'`: Bottom base forecasts are provided as samples.
        - `'params'`: Bottom base forecasts are provided as estimated parameters.
        Default is `'pmf'`.

    distr : str, optional
        Specifies the distribution type for the bottom base forecasts. Only used if `bottom_in_type == "params"`.
        Possible values:

        - `'poisson'`
        - `'nbinom'`

    num_samples : int, optional
        Number of samples to draw from the reconciled distribution. Ignored if `bottom_in_type == "samples"`.
        In this case, the number of reconciled samples equals the number of samples in the base forecasts.
        Default is `20,000`.

    return_type : str, optional
        Specifies the return type of the reconciled distributions. Possible values:

        - `'pmf'`: Returns reconciled marginal PMF objects.
        - `'samples'`: Returns reconciled multivariate samples.
        - `'all'`: Returns both PMF objects and samples.
        Default is `'pmf'`.

    suppress_warnings : bool, optional
        Whether to suppress warnings about samples lying outside the support of the bottom-up distribution.
        Default is `False`.

    seed : int or None, optional
        Random seed for reproducibility. Default is `None`.

    return_num_samples_ok : bool, optional
        Whether to return the number of samples that were valid after reconciliation. Default is `False`.

    Returns
    -------
    Union[Dict[str, Dict[str, Union[List, np.ndarray]]], Tuple[Dict, int]]
        A dictionary containing the reconciled forecasts:

        - `'bottom_reconciled'`: Contains the reconciled forecasts for the bottom-level variables.

            - If `return_type == "pmf"`: A list of PMF objects.
            - If `return_type == "samples"`: A matrix of shape `(n_bottom, num_samples)`.
            - If `return_type == "all"`: Contains both PMF objects and samples.
        - `'upper_reconciled'`: Contains the reconciled forecasts for the upper-level variables.

            - If `return_type == "pmf"`: A list of PMF objects.
            - If `return_type == "samples"`: A matrix of shape `(n_upper, num_samples)`.
            - If `return_type == "all"`: Contains both PMF objects and samples.

        - If `return_num_samples_ok` is `True`, a tuple is returned.
        - The first element is the above dictionary.
        - The second element is the number of valid samples after reconciliation.

    Notes
    -----
    - A PMF object is a numerical vector where each element corresponds to the probability of integers
      from `0` to the last value in the support.
    - Samples lying outside the support of the bottom-up distribution are discarded, and a warning is issued
      if `suppress_warnings` is `False`.

    Examples
    --------
    Simple hierarchy with PMF inputs
        >>> import numpy as np
        >>> from scipy.stats import poisson
        >>>
        >>> # Simple hierarchy with one upper and two bottom nodes
        >>> A = np.array([[1, 1]])  # Aggregation matrix
        >>>
        >>> # Bottom forecasts as Poisson distributions
        >>> lambda_val = 15
        >>> n_tot = 60
        >>> fc_bottom = {
        ...     0: np.array([poisson.pmf(x, lambda_val) for x in range(n_tot + 1)]),
        ...     1: np.array([poisson.pmf(x, lambda_val) for x in range(n_tot + 1)])
        ... }
        >>>
        >>> # Upper forecast as Gaussian parameters
        >>> fc_upper = {
        ...     "mu": np.array([40]),  # Mean
        ...     "Sigma": np.array([[25]])  # Variance (standard deviation squared)
        ... }
        >>>
        >>> # Perform reconciliation
        >>> result = reconc_td_cond(A, fc_bottom, fc_upper, bottom_in_type="pmf", return_type="all")
        >>>
        >>> # Check the results
        >>> print(result['bottom_reconciled']['pmf'][0])
        >>> print(result['bottom_reconciled']['pmf'][1])
        >>> print(result['upper_reconciled']['pmf'][0])


    References
    ----------
    - Zambon, L., Azzimonti, D., Rubattu, N., Corani, G. (2024).
      *Probabilistic reconciliation of mixed-type hierarchical time series*.
      40th Conference on Uncertainty in Artificial Intelligence.

    See Also
    --------
    reconc_mix_cond : Reconciliation for mixed-type hierarchies.
    reconc_buis : Reconciliation using Bottom-Up Importance Sampling (BUIS).
    """
    if seed is not None:
        np.random.seed(seed)

    # Check inputs
    _check_input_TD(A, fc_bottom, fc_upper, bottom_in_type, distr, return_type)

    # Find the "lowest upper"
    n_u = A.shape[0]
    n_b = A.shape[1]
    lowest_rows = _lowest_lev(A)
    n_u_low = len(lowest_rows)  # number of lowest upper
    n_u_upp = n_u - n_u_low  # number of "upper upper"

    # Get mean and covariance matrix of the MVN upper base forecasts
    mu_u = fc_upper['mu']
    mu_u = np.array([mu_u[key] for key in mu_u]) if isinstance(mu_u, dict) else mu_u

    Sigma_u = np.array(fc_upper['Sigma'])

    # Get upper samples
    if n_u == n_u_low:
        U = _MVN_sample(num_samples, mu_u, Sigma_u)  # (dim: num_samples x n_u_low)
        U = np.round(U)  # round to integer
        U_js = [U[:, i] for i in range(U.shape[1])]
    else:
        # Reconcile the upper
        A_u = _get_Au(A, lowest_rows)
        mu_u_ord = np.concatenate([np.delete(mu_u,lowest_rows), mu_u[lowest_rows]])
        Sigma_u_ord = np.zeros((n_u, n_u))
        Sigma_u_ord[:n_u_upp, :n_u_upp] = np.delete(np.delete(Sigma_u, lowest_rows, axis=0), lowest_rows, axis=1)
        Sigma_u_ord[:n_u_upp, n_u_upp:] = np.delete(Sigma_u, lowest_rows, axis=0)[:, lowest_rows]
        Sigma_u_ord[n_u_upp:, :n_u_upp] = np.delete(Sigma_u, lowest_rows, axis=1)[lowest_rows, :]
        Sigma_u_ord[n_u_upp:, n_u_upp:] = Sigma_u[np.ix_(lowest_rows, lowest_rows)]

        rec_gauss_u = reconc_gaussian(A_u, mu_u_ord, Sigma_u_ord)
        U = _MVN_sample(num_samples, rec_gauss_u['bottom_reconciled_mean'], rec_gauss_u['bottom_reconciled_covariance'])
        U = np.round(U)  # round to integer
        U_js = [U[:, i] for i in range(U.shape[1])]

    # Prepare list of bottom pmf
    if bottom_in_type == "pmf":
        L_pmf = [v for v in fc_bottom.values()]
    elif bottom_in_type == "samples":
        if isinstance(fc_bottom, dict):
            L_pmf = [_pmf_from_samples(fc) for fc in fc_bottom.values()]
        else:
            L_pmf = [_pmf_from_samples(fc) for fc in fc_bottom]
    elif bottom_in_type == "params":
        L_pmf = [_pmf_from_params(fc, distr) for key, fc in fc_bottom.items()]

    # Prepare list of lists of bottom pmf relative to each lowest upper
    L_pmf_js = []
    for j in lowest_rows:
        Aj = A[j, :]
        L_pmf_js.append([L_pmf[i] for i in range(len(Aj)) if Aj[i]])

    # Check that each multiv. sample of U is contained in the supp of the bottom-up distr
    samp_ok = np.array([_pmf_check_support(u_j, L_pmf_j) for u_j, L_pmf_j in zip(U_js, L_pmf_js)])
    samp_ok = np.sum(samp_ok, axis=0) == n_u_low

    U_js = [U_j[samp_ok] for U_j in U_js]
    num_samples_ok = np.sum(samp_ok)

    if num_samples_ok != num_samples and not suppress_warnings:
        print(f"Warning: Only {np.floor(np.sum(samp_ok) / num_samples * 1000) / 10}% of the upper samples "
              "are in the support of the bottom-up distribution; the others are discarded.")

    # Get bottom samples via the prob top-down
    B = np.empty((n_b, num_samples_ok))
    for j in range(n_u_low):

        mask_j = A[lowest_rows[j], :]  # mask for the position of the bottom referring to lowest upper j
        B[np.where(mask_j)[0], :] = _TD_sampling(U_js[j], L_pmf_js[j])

    U = A @ B  # dim: n_upper x num_samples
    B = B.astype(int)
    U = U.astype(int)

    # Prepare output: include the marginal pmfs and/or the samples (depending on "return" inputs)
    result = {'bottom_reconciled': {}, 'upper_reconciled': {}}

    if return_type in ['pmf', 'all']:
        upper_pmf = [_pmf_from_samples(U[i, :]) for i in range(n_u)]
        bottom_pmf = [_pmf_from_samples(B[i, :]) for i in range(n_b)]
        result['bottom_reconciled']['pmf'] = bottom_pmf
        result['upper_reconciled']['pmf'] = upper_pmf

    if return_type in ['samples', 'all']:
        result['bottom_reconciled']['samples'] = B
        result['upper_reconciled']['samples'] = U

    if return_num_samples_ok:
        return result, num_samples_ok
    else:
        return result
