import numpy as np
from scipy.linalg import solve, LinAlgError
from typing import List, Dict, Tuple

def _check_cov(matrix, name, pd_check=False, symm_check=False):
    """Check if a covariance matrix is positive definite and/or symmetric.

    Parameters:
    - matrix: The matrix to check
    - name: The name of the matrix (for error messages)
    - pd_check: Boolean indicating whether to check for positive definiteness
    - symm_check: Boolean indicating whether to check for symmetry
    """
    if symm_check and not np.allclose(matrix, matrix.T):
        raise ValueError(f"Input error: {name} is not symmetric")

    if pd_check:
        try:
            np.linalg.cholesky(matrix)
        except LinAlgError:
            raise ValueError(f"Input error: {name} is not positive definite")


def reconc_gaussian(A: np.ndarray, base_forecasts_mu: List[float], base_forecasts_Sigma: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Analytical reconciliation of Gaussian base forecasts.
    Reconciles forecasts using a closed-form computation for Gaussian base forecasts.

    Parameters
    ----------
    A : numpy.ndarray
        Aggregation matrix with shape `(n_upper, n_bottom)`. Each column represents a bottom-level
        forecast, and each row represents an upper-level forecast. A value of `1` in `A[i, j]` indicates
        that the bottom-level forecast `j` contributes to the upper-level forecast `i`.

    base_forecasts_mu : List[float]
        A 1D list containing the means of the base forecasts. The order is:

        - First, the upper-level means (in the order of rows in `A`).
        - Then, the bottom-level means (in the order of columns in `A`).

    base_forecasts_Sigma : numpy.ndarray
        A 2D covariance matrix representing the uncertainties in the base forecasts.
        The order of rows and columns must match the order of `base_forecasts_mu`.
        Shape: `(n, n)`, where `n = n_upper + n_bottom`.


    Returns
    -------
    Dict[str, numpy.ndarray]
        A dictionary containing:

        - `'bottom_reconciled_mean'` : numpy.ndarray
            A 1D array of the reconciled means for the bottom-level forecasts.
            Shape: `(n_bottom,)`.

        - `'bottom_reconciled_covariance'` : numpy.ndarray
            A 2D covariance matrix of the reconciled bottom-level forecasts.
            Shape: `(n_bottom, n_bottom)`.

    Notes
    -----
    - The function assumes that base forecasts follow a multivariate Gaussian distribution.
    - The covariance matrix `base_forecasts_Sigma` should be symmetric and positive semi-definite.
    - The order of elements in `base_forecasts_mu` and rows/columns in `base_forecasts_Sigma` is critical:
      first the upper-level forecasts (in the order of rows in `A`), followed by the bottom-level forecasts
      (in the order of columns in `A`).
    - The function returns only the reconciled parameters for the bottom-level forecasts. Reconciled
      parameters for upper-level forecasts and the entire hierarchy can be derived using the reconciliation
      matrix `A`.

    Examples
    --------
    Example 1: Minimal hierarchy with Gaussian base forecasts
        >>> A = np.array([
        ...     [1, 0, 0],
        ...     [0, 1, 1]
        ... ])
        >>> base_forecasts_mu = [9.0, 2.0, 4.0]
        >>> base_forecasts_Sigma = np.diag([9.0, 4.0, 4.0])
        >>> result = reconc_gaussian(A, base_forecasts_mu, base_forecasts_Sigma)
        >>> print(result['bottom_reconciled_mean'])
        [2.5, 4.0]
        >>> print(result['bottom_reconciled_covariance'])
        [[2.25, 1.5 ],
         [1.5 , 2.0 ]]

    References
    ----------
    - Corani, G., Azzimonti, D., Augusto, J.P.S.C., Zaffalon, M. (2021).
      *Probabilistic Reconciliation of Hierarchical Forecast via Bayes' Rule*.
      ECML PKDD 2020. Lecture Notes in Computer Science, vol 12459.
      https://doi.org/10.1007/978-3-030-67664-3_13
    - Zambon, L., Agosto, A., Giudici, P., Corani, G. (2024).
      *Properties of the reconciled distributions for Gaussian and count forecasts*.
      International Journal of Forecasting (in press).
      https://doi.org/10.1016/j.ijforecast.2023.12.004
    """


    k = A.shape[0]  # number of upper TS
    m = A.shape[1]  # number of bottom TS
    n = len(base_forecasts_mu)  # total number of TS

    # Ensure that data inputs are valid
    if base_forecasts_Sigma.shape[0] != n or base_forecasts_Sigma.shape[1] != n:
        raise ValueError("Input error: base_forecasts_Sigma shape is incorrect")

    if k + m != n:
        raise ValueError("Input error: the shape of A is not correct")

    _check_cov(base_forecasts_Sigma, "Sigma", pd_check=False, symm_check=True)

    Sigma_u = base_forecasts_Sigma[:k, :k]
    Sigma_b = base_forecasts_Sigma[k:, k:]
    Sigma_ub = base_forecasts_Sigma[:k, k:]
    mu_u = base_forecasts_mu[:k]
    mu_b = base_forecasts_mu[k:]

    # Calculate Q
    Q = Sigma_u - Sigma_ub @ A.T - A @ Sigma_ub.T + A @ Sigma_b @ A.T

    # Check if Q is positive definite
    _check_cov(Q, "Q", pd_check=True, symm_check=False)

    # Calculate invQ
    invQ = np.linalg.inv(Q)

    # Calculate mu_b_tilde and Sigma_b_tilde
    mu_b_tilde = mu_b + (Sigma_ub.T - Sigma_b @ A.T) @ invQ @ (A @ mu_b - mu_u)
    term = Sigma_ub.T - Sigma_b @ A.T
    Sigma_b_tilde = Sigma_b - term @ invQ @ term.T

    return {
        'bottom_reconciled_mean': mu_b_tilde,
        'bottom_reconciled_covariance': Sigma_b_tilde
    }
