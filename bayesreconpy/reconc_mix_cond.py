import numpy as np
from scipy import stats
from bayesreconpy.utils import _check_input_TD, _check_weights, _resample, _MVN_density
from bayesreconpy.PMF import _pmf_from_samples, _pmf_from_params, _pmf_sample
from typing import Union, Optional, Dict, List


def reconc_mix_cond(
    A: np.ndarray,
    fc_bottom: Union[np.array, dict],
    fc_upper: dict,
    bottom_in_type: str = "pmf",
    distr: Optional[str] = None,
    num_samples: int = 20000,
    return_type: str = "pmf",
    suppress_warnings: bool = False,
    seed: Optional[int] = None
) -> Dict[str, Union[Dict[str, Union[List, np.ndarray]], float]]:
    """
    Probabilistic forecast reconciliation of mixed hierarchies via conditioning.

    Uses importance sampling to draw samples from the reconciled forecast distribution,
    obtained via conditioning, in the case of a mixed hierarchy.

    Parameters
    ----------
    A : numpy.ndarray
        Aggregation matrix with shape `(n_upper, n_bottom)`. Each column represents a bottom-level
        forecast, and each row represents an upper-level forecast. A value of `1` in `A[i, j]`
        indicates that bottom-level forecast `j` contributes to upper-level forecast `i`.

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
        Whether to suppress warnings about importance sampling weights.
        Default is `False`.

    seed : int or None, optional
        Random seed for reproducibility. Default is `None`.

    Returns
    -------
    Dict[str, Union[Dict[str, Union[List, np.ndarray]], float]]
        A dictionary containing the reconciled forecasts:

        - `'bottom_reconciled'`: Contains the reconciled forecasts for the bottom-level variables.

            - If `return_type == "pmf"`: A list of PMF objects.
            - If `return_type == "samples"`: A matrix of shape `(n_bottom, num_samples)`.
            - If `return_type == "all"`: Contains both PMF objects and samples.

        - `'upper_reconciled'`: Contains the reconciled forecasts for the upper-level variables.

            - If `return_type == "pmf"`: A list of PMF objects.
            - If `return_type == "samples"`: A matrix of shape `(n_upper, num_samples)`.
            - If `return_type == "all"`: Contains both PMF objects and samples.
        - `'ESS'`: Effective sample size after importance sampling.

    Notes
    -----
    - A PMF object is a numerical vector where each element corresponds to the probability of integers
      from `0` to the last value in the support.
    - Warnings are triggered during the importance sampling step if:

      * All weights are zero, causing the upper forecast to be ignored.
      * Effective sample size (ESS) is less than `200`.
      * ESS is less than `1%` of the total sample size.

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
        >>> result = reconc_mix_cond(A, fc_bottom, fc_upper, bottom_in_type="pmf", return_type="all")
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
    reconc_td_cond : Probabilistic reconciliation via top-down conditioning.
    reconc_buis : Reconciliation using Bottom-Up Importance Sampling (BUIS).
    """

    if seed is not None:
        np.random.seed(seed)

    # Check inputs
    _check_input_TD(A, fc_bottom, fc_upper, bottom_in_type, distr, return_type)


    n_u = A.shape[0]
    n_b = A.shape[1]

    # Prepare samples from the base bottom distribution
    if bottom_in_type == "pmf":
        B = np.vstack([_pmf_sample(fc, num_samples) for key, fc in fc_bottom.items()])
        B = B.T
    elif bottom_in_type == "samples":
        B = np.vstack(fc_bottom)
        B = B.T
        num_samples = B.shape[0]
    elif bottom_in_type == "params":
        L_pmf = [_pmf_from_params(fc, distr) for key, fc in fc_bottom.items()]
        B = np.vstack([_pmf_sample(pmf, num_samples) for pmf in L_pmf])
        B = B.T

    # Get mean and covariance matrix of the MVN upper base forecasts
    mu_u = fc_upper['mu']
    mu_u = [mu_u[key] for key in mu_u] if isinstance(mu_u, dict) else mu_u
    Sigma_u = np.array(fc_upper['Sigma'])

    # IS using MVN
    U = B @ A.T
    weights = _MVN_density(U, mu_u, Sigma_u)


    check_weights_res = _check_weights(weights)
    if check_weights_res['warning'] and not suppress_warnings:
        warning_msg = check_weights_res['warning_msg']
        print(f"Warning: {warning_msg}")

    if not (check_weights_res['warning'] and (1 in check_weights_res['warning_code'])):
        B = _resample(B, weights, num_samples)

    ESS = np.sum(weights) ** 2 / np.sum(weights ** 2)

    B = B.T
    U = A @ B

    # Prepare output: include the marginal pmfs and/or the samples
    result = {
        'bottom_reconciled': {},
        'upper_reconciled': {},
        'ESS': ESS
    }

    if return_type in ['pmf', 'all']:
        upper_pmf = [_pmf_from_samples(U[i, :]) for i in range(n_u)]
        bottom_pmf = [_pmf_from_samples(B[i, :]) for i in range(n_b)]

        result['bottom_reconciled']['pmf'] = bottom_pmf
        result['upper_reconciled']['pmf'] = upper_pmf

    if return_type in ['samples', 'all']:
        result['bottom_reconciled']['samples'] = B
        result['upper_reconciled']['samples'] = U

    return result
