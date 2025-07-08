import numpy as np
from bayesreconpy.utils import _check_input_BUIS, _distr_sample, _distr_pmf
from bayesreconpy.reconc_buis import reconc_buis
from typing import Optional, Dict


def reconc_mcmc(
    A: np.ndarray,
    base_forecasts: list,
    distr: str,
    num_samples: int = 10000,
    tuning_int: int = 100,
    init_scale: float = 1,
    burn_in: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    MCMC for Probabilistic Reconciliation of forecasts via conditioning.

    Uses a Markov Chain Monte Carlo (MCMC) algorithm to draw samples from the reconciled
    forecast distribution, obtained via conditioning. This implementation uses the
    Metropolis-Hastings algorithm.

    **Note**: This implementation only supports Poisson or Negative Binomial base forecasts.
    Gaussian distributions are not supported for MCMC.

    Parameters
    ----------
    A : numpy.ndarray
        Aggregation matrix of shape `(n_upper, n_bottom)`. Each column represents a bottom-level
        forecast, and each row represents an upper-level forecast. A value of `1` in `A[i, j]`
        indicates that bottom-level forecast `j` contributes to upper-level forecast `i`.

    base_forecasts : list
        A list containing the parameters of the base forecast distributions. The first `n_upper` elements
        correspond to the upper base forecasts (in the order of rows in `A`), and the remaining `n_bottom`
        elements correspond to the bottom base forecasts (in the order of columns in `A`). Each element is a dictionary:

        - For `'poisson'`: {"lambda": float}
        - For `'nbinom'`: {"size": float, "prob": float} or {"size": float, "mu": float}

    distr : str
        Type of predictive distribution. Supported values:

        - `'poisson'`
        - `'nbinom'`

    num_samples : int, optional
        Number of samples to draw using MCMC. Default is `10,000`.

    tuning_int : int, optional
        Number of iterations between scale updates of the proposal. Default is `100`.

    init_scale : float, optional
        Initial scale of the proposal distribution. Default is `1.0`.

    burn_in : int, optional
        Number of initial samples to discard. Default is `1,000`.

    seed : int or None, optional
        Random seed for reproducibility. Default is `None`.

    Returns
    -------
    Dict
        A dictionary containing the reconciled forecasts:

        - `'bottom_reconciled_samples'`: numpy.ndarray
            A matrix of shape `(n_bottom, num_samples)` containing
            reconciled samples for the bottom-level time series.
        - `'upper_reconciled_samples'`: numpy.ndarray
            A matrix of shape `(n_upper, num_samples)` containing
            reconciled samples for the upper-level time series.
        - `'reconciled_samples'`: numpy.ndarray
            A matrix of shape `(n, num_samples)` containing the reconciled
            samples for all time series, where `n = n_upper + n_bottom`.

    Notes
    -----
    - This is a bare-bones implementation of the Metropolis-Hastings algorithm.
    - We recommend using additional tools to assess the convergence of the MCMC chains.
    - The `reconc_buis` function is generally faster for most hierarchies.

    Examples
    --------
    Example: Simple hierarchy with Poisson base forecasts
        >>> import numpy as np
        >>> from bayesreconpy.reconc_buis import reconc_buis
        >>>
        >>> # Create a minimal hierarchy with 1 upper and 2 bottom variables
        >>> A = np.array([[1, 1]])  # Aggregation matrix
        >>>
        >>> # Set the parameters of the Poisson base forecast distributions
        >>> lambda_vals = [9, 2, 4]
        >>> base_forecasts = [{"lambda": lam} for lam in lambda_vals]
        >>>
        >>> # Perform MCMC reconciliation
        >>> mcmc_result = reconc_mcmc(A, base_forecasts, distr="poisson", num_samples=30000, seed=42)
        >>>
        >>> # Access reconciled samples
        >>> samples_mcmc = mcmc_result['reconciled_samples']
        >>>
        >>> # Compare reconciled means with those from BUIS
        >>> buis_result = reconc_buis(A, base_forecasts, in_type="params", distr="poisson", num_samples=100000, seed=42)
        >>> samples_buis = buis_result['reconciled_samples']
        >>>
        >>> print("MCMC Reconciled Means:", np.mean(samples_mcmc, axis=1))
        >>> print("BUIS Reconciled Means:", np.mean(samples_buis, axis=1))

    References
    ----------
    - Corani, G., Azzimonti, D., Rubattu, N. (2024).
      *Probabilistic reconciliation of count time series*.
      International Journal of Forecasting, 40(2), 457-469.

    See Also
    --------
    reconc_buis : Faster reconciliation method for most hierarchies.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure that data inputs are valid
    if distr == "gaussian":
        raise NotImplementedError("MCMC for Gaussian distributions is not implemented")

    n_bottom = A.shape[1]
    n_ts = A.shape[0] + A.shape[1]

    # Transform distr into list
    if not isinstance(distr, list):
        distr = [distr] * n_ts

    # Check input
    _check_input_BUIS(A, base_forecasts, in_type=['params'] * n_ts, distr=distr)

    # the first burn_in samples will be removed
    num_samples += burn_in

    # Set the covariance matrix of the proposal (identity matrix)
    cov_mat_prop = np.eye(n_bottom)

    # Set the counter for tuning
    c_tuning = tuning_int

    # Initialize acceptance counter
    accept_count = 0

    # Create empty matrix for the samples from MCMC
    b = np.zeros((num_samples, n_bottom))

    # Get matrix A and bottom base forecasts
    split_hierarchy_res = {
        'A': A,
        'upper': base_forecasts[:A.shape[0]],
        'bottom': base_forecasts[A.shape[0]:],
        'upper_idxs': list(range(A.shape[0])),
        'bottom_idxs': list(range(A.shape[0], n_ts))
    }
    bottom_base_forecasts = split_hierarchy_res['bottom']
    bottom_distr = [distr[i] for i in split_hierarchy_res['bottom_idxs']]

    # Initialize first sample (draw from base distribution)
    b[0, :] = _initialize_b(bottom_base_forecasts, bottom_distr).ravel()

    # Initialize prop list
    old_prop = {
        'b': b[0, :],
        'scale': init_scale
    }

    # Run the chain
    for i in range(1, num_samples):
        if c_tuning == 0:
            old_prop['acc_rate'] = accept_count / tuning_int  # set acc_rate
            accept_count = 0  # reset acceptance counter
            c_tuning = tuning_int  # reset tuning counter

        prop = _proposal(old_prop, cov_mat_prop)
        b_prop = prop['b']
        alpha = _accept_prob(b_prop, b[i - 1, :], A, distr, base_forecasts)

        if np.random.uniform() < alpha:
            b[i, :] = b_prop
            accept_count += 1
        else:
            b[i, :] = b[i - 1, :]

        old_prop = {
            'b': b[i, :],
            'scale': prop['scale']
        }

        c_tuning -= 1

    b_samples = b[burn_in:, :].T  # output shape: n_bottom x num_samples
    u_samples = A @ b_samples
    y_samples = np.vstack([u_samples, b_samples])

    return {
        'bottom_reconciled_samples': b_samples,
        'upper_reconciled_samples': u_samples,
        'reconciled_samples': y_samples
    }


def _initialize_b(bottom_base_forecasts, bottom_distr):
    b = []
    for i in range(len(bottom_distr)):
        b.append(_distr_sample(bottom_base_forecasts[i], bottom_distr[i], 1))
    return np.array(b)


def _accept_prob(b, b0, A, distr, params):
    alpha = _target_pmf(b, A, distr, params) / _target_pmf(b0, A, distr, params)
    return min(1, alpha)


def _target_pmf(b, A, distr, params):
    n_ts = A.shape[0] + A.shape[1]
    y = np.vstack([A, np.eye(A.shape[1])]) @ b
    pmf = 1
    for j in range(n_ts):
        pmf *= _distr_pmf(y[j], params[j], distr[j])
    return pmf


def _proposal(prev_prop, cov_mat_prop):
    b0 = prev_prop['b']
    old_scale = prev_prop['scale']
    acc_rate = prev_prop.get('acc_rate', None)

    if acc_rate is not None:
        scale = _tune(old_scale, acc_rate)
    else:
        scale = old_scale

    n_x = len(b0)

    if n_x != cov_mat_prop.shape[0]:
        raise ValueError(
            f"Error in proposal: previous state dim ({n_x}) and covariance dim ({cov_mat_prop.shape[0]}) do not match")

    dd = np.random.normal(size=n_x) * np.diag(cov_mat_prop) * scale
    dd = np.round(dd, 0)
    b = b0 + dd

    return {'b': b, 'acc_rate': acc_rate, 'scale': scale}


def _tune(scale, acc_rate):
    if acc_rate < 0.001:
        return scale * 0.1
    elif acc_rate < 0.05:
        return scale * 0.5
    elif acc_rate < 0.2:
        return scale * 0.9
    elif acc_rate > 0.5:
        return scale * 1.1
    elif acc_rate > 0.75:
        return scale * 2
    elif acc_rate > 0.95:
        return scale * 10
    return scale
