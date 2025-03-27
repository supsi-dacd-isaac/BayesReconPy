import numpy as np
from bayesreconpy.reconc_ols import get_S_from_A
from bayesreconpy.shrink_cov import _schafer_strimmer_cov as schafer_strimmer_cov

def estimate_cov_matrix(res, upper=False, n_u=None):
    """
    Estimate the covariance matrix using the Schafer-Strimmer method.

    Parameters:
    - res: array-like, residuals or data for covariance estimation.
    - upper: bool, whether to use only the upper portion of the data.
    - n_u: int, number of upper rows/columns to include (required if upper=True).

    Returns:
    - ss: Covariance matrix (shrinkage-based or regular).
    """
    if upper:
        if n_u is None:
            raise ValueError("n_u must be provided when upper=True.")
        ss = schafer_strimmer_cov(res[:n_u, :n_u])
    else:
        ss = schafer_strimmer_cov(res)['shrink_cov']
    return ss

def reconc_mint(A: np.ndarray,base_forecasts: np.ndarray,res:np.ndarray,samples=False):
    """
        Reconciles base forecasts using the MinT (Minimum Trace) approach with a shrinkage-based covariance estimator.

    The MinT method adjusts base forecasts to ensure coherence with a given aggregation structure,
    minimizing the total variance of the reconciliation errors using the residual covariance matrix.

    Parameters:
    ----------
    A : np.ndarray
        The aggregation constraint matrix (typically of shape [n_agg_levels, n_total_series]),
        defining how the bottom-level time series aggregate into upper levels.

    base_forecasts : np.ndarray
        The base forecasts to be reconciled:
        - of shape [n_total_series, n_time, n_samples] if `samples=True`, or
        - of shape [n_total_series, n_time] if `samples=False`.

    res : np.ndarray
        Residuals (forecast errors) from a previous model, used to estimate the covariance matrix.
        Should be of shape [n_total_series, n_time].

    samples : bool, optional (default=False)
        Indicates whether the base forecasts include multiple samples. If True, reconciliation is
        applied sample-by-sample along the third axis.

    Returns:
    -------
    y_tilde_mean : np.ndarray
        The reconciled forecast means, with the same shape as `base_forecasts`.

    y_tilde_var : np.ndarray or list of np.ndarray
        The reconciled forecast variances:
        - A single matrix of shape [n_total_series, n_total_series] if `samples=False`.
        - A list of such matrices, one per sample, if `samples=True`.
    """
    S = get_S_from_A(A)
    W_h = estimate_cov_matrix(res)
    P = np.linalg.inv(S.T @ np.linalg.inv(W_h) @ S) @ S.T @ np.linalg.inv(W_h)

    if samples:
        y_tilde_mean = np.zeros_like(base_forecasts)
        y_tilde_var = []
        for i in range(base_forecasts.shape[2]):
            y_hat = base_forecasts[:,:,i]
            y_tilde_mean[:,:,i] = S @ P @ y_hat
            y_tilde_var.append(S @ P @ W_h @ P.T @ S.T)
    else:
        y_hat = base_forecasts
        y_tilde_mean = S @ P @ y_hat
        y_tilde_var = S @ P @ W_h @ P.T @ S.T
    return y_tilde_mean, y_tilde_var