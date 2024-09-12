import numpy as np
from scipy.linalg import solve, LinAlgError


def check_cov(matrix, name, pd_check=False, symm_check=False):
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


def reconc_gaussian(A, base_forecasts_mu, base_forecasts_Sigma):
    k = A.shape[0]  # number of upper TS
    m = A.shape[1]  # number of bottom TS
    n = len(base_forecasts_mu)  # total number of TS

    # Ensure that data inputs are valid
    if base_forecasts_Sigma.shape[0] != n or base_forecasts_Sigma.shape[1] != n:
        raise ValueError("Input error: base_forecasts_Sigma shape is incorrect")

    if k + m != n:
        raise ValueError("Input error: the shape of A is not correct")

    check_cov(base_forecasts_Sigma, "Sigma", pd_check=False, symm_check=True)

    Sigma_u = base_forecasts_Sigma[:k, :k]
    Sigma_b = base_forecasts_Sigma[k:, k:]
    Sigma_ub = base_forecasts_Sigma[:k, k:]
    mu_u = base_forecasts_mu[:k]
    mu_b = base_forecasts_mu[k:]

    # Calculate Q
    Q = Sigma_u - Sigma_ub @ A.T - A @ Sigma_ub.T + A @ Sigma_b @ A.T

    # Check if Q is positive definite
    check_cov(Q, "Q", pd_check=True, symm_check=False)

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
