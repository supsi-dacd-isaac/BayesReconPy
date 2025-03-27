import numpy as np

def get_S_from_A(A:np.ndarray):
    n_b = A.shape[1]
    S = np.concatenate((A, np.eye(n_b)), axis=0)
    return S

def reconc_ols(A: np.ndarray,base_forecasts: np.ndarray, samples=False):
    """
    Reconciles base forecasts using Ordinary Least Squares (OLS) based on a summing matrix.

    This function applies forecast reconciliation using the OLS method, ensuring that
    the reconciled forecasts are coherent with the aggregation structure defined by the
    matrix `A`.

    Parameters:
    ----------
    A : np.ndarray
        The aggregation constraint matrix (usually of shape [n_agg_levels, n_total_series]).
        It defines how bottom-level time series are aggregated into upper levels.

    base_forecasts : np.ndarray
        The base forecasts before reconciliation. Should be:
        - of shape [n_total_series, n_time, n_samples] if `samples=True`, or
        - of shape [n_total_series, n_time] if `samples=False`.

    samples : bool, optional (default=False)
        Indicates whether the base forecasts contain multiple samples (i.e., are probabilistic).
        If True, reconciliation is applied to each sample individually along the third axis.

    Returns:
    -------
    y_tilde_mean : np.ndarray
        The reconciled forecasts:
        - of shape [n_total_series, n_time, n_samples] if `samples=True`, or
        - of shape [n_total_series, n_time] if `samples=False`.

    """

    S = get_S_from_A(A)
    P = (np.linalg.inv(S.T @ S)) @ S.T
    if samples:
        y_tilde_mean = np.zeros_like(base_forecasts)
        for i in range(base_forecasts.shape[2]):
            y_hat = base_forecasts[:,:,i]
            y_tilde_mean[:,:,i] = S @ P @ y_hat
    else:
        y_hat = base_forecasts
        y_tilde_mean = S @ P @ y_hat
    return y_tilde_mean
