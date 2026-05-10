from typing import Callable
import numpy as np
from typing import List, Union, Dict, Callable, Optional, Tuple
from utils.stats import _schafer_strimmer_cov


def _logpdf_mvn(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    X = np.asarray(X, float)
    mean = np.asarray(mean, float).reshape(-1)
    cov = np.asarray(cov, float)

    d = mean.size
    cov = 0.5 * (cov + cov.T) + eps * np.eye(d)

    L = np.linalg.cholesky(cov)
    XC = X - mean
    Y = np.linalg.solve(L, XC.T)               # (d, N)
    maha = np.sum(Y * Y, axis=0)               # (N,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2*np.pi) + logdet + maha)


def _compute_weights(
    B: np.ndarray,            # (N, n_free)
    U_pred: np.ndarray,       # (N, n_constrained)  <-- coherent constrained predictions for each free particle
    joint_mean: np.ndarray,   # (n_constrained+n_free,)
    joint_cov: np.ndarray, # (n_constrained+n_free, n_constrained+n_free)
    eps_cov: float = 1e-9,
) -> np.ndarray:

    B = np.asarray(B)
    U_pred = np.asarray(U_pred)

    N, n_free = B.shape
    if U_pred.ndim != 2 or U_pred.shape[0] != N:
        raise ValueError("U_pred must be shape (N, n_constrained)")
    n_constrained = U_pred.shape[1]

    d = n_constrained + n_free
    mu = np.asarray(joint_mean, float).reshape(-1)
    Sig = np.asarray(joint_cov, float)

    if mu.size != d:
        raise ValueError(f"joint_mean must have length {d} (=n_constrained+n_free), got {mu.size}")
    if Sig.shape != (d, d):
        raise ValueError(f"joint_cov must be shape {(d, d)}, got {Sig.shape}")

    X_joint = np.hstack([U_pred, B])  # (N, d)

    # marginal over B: take free block
    mu_b = mu[n_constrained:]
    Sig_b = Sig[n_constrained:, n_constrained:]

    log_joint = _logpdf_mvn(X_joint, mu, Sig, eps=eps_cov)
    log_b = _logpdf_mvn(B, mu_b, Sig_b, eps=eps_cov)

    w = np.exp(log_joint - log_b)
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return w

def reconc_nl_is(
        f_u: Callable[[np.ndarray], np.ndarray],
        num_samples=10000,
        n_free = None,
        seed=None,
        joint_mean=None,         
        joint_cov=None,          
        eps_cov: float = 1e-9,
):
    """
    Nonlinear Probabilistic Reconciliation using Importance Sampling (IS).

    Uses importance sampling to reconcile forecasts by conditioning on a nonlinear manifold.
    The method draws samples from the joint distribution of free and constrained variables,
    computes importance weights based on the constraint function, and resamples to obtain
    the reconciled distribution.

    Parameters
    ----------
    f_u : Callable[[np.ndarray], np.ndarray]
        The FTC function mapping free variables to constrained variables.

        - Input: array of shape `(N, n_free)` - N samples of the n_free free variables.
        - Output: array of shape `(n_constrained, N)` - constrained variables for each sample.

        The function defines the relationship: `constrained = f_u(free)`.

    num_samples : int, optional
        Number of samples to draw from the reconciled distribution. Default is 10,000.

    n_free : int
        Number of free (bottom-level) variables. Required parameter.
        The joint mean and covariance have dimension `n_constrained + n_free`.

    seed : int or None, optional
        Random seed for reproducibility. Default is `None`.

    joint_mean : numpy.ndarray
        Mean vector of the joint distribution over [constrained, free] variables.
        Shape: `(n_constrained + n_free,)`.

        The first `n_constrained` elements are the means of the constrained variables,
        and the remaining `n_free` elements are the means of the free variables.

    joint_cov : numpy.ndarray
        Covariance matrix of the joint distribution over [constrained, free] variables.
        Shape: `(n_constrained + n_free, n_constrained + n_free)`.
        Must be symmetric and positive semi-definite.

    eps_cov : float, optional
        Small regularization value added to the diagonal of covariance matrices to ensure
        numerical stability. Default is 1e-9.

    Returns
    -------
    dict
        A dictionary containing the reconciled forecasts:

        - `'free_reconciled_samples'`: numpy.ndarray
            Reconciled samples for the free variables.
            Shape: `(n_free, num_samples)`.

        - `'constrained_reconciled_samples'`: numpy.ndarray
            Reconciled samples for the constrained variables.
            Shape: `(n_constrained, num_samples)`.

        - `'reconciled_samples'`: numpy.ndarray
            Concatenated reconciled samples for all variables [constrained, free].
            Shape: `(n_constrained + n_free, num_samples)`.

    Raises
    ------
    ValueError
        If `joint_mean` or `joint_cov` is `None`. Both parameters are required.

    TypeError
        If `joint_mean` is `None` and accessed before validation.

    Notes
    -----
    - The function computes weights as: w_j ∝ p(U_pred_j, B_j) / p(B_j)
      where U_pred_j = f_u(B_j) are the predicted constrained variables.
    - Weights are normalized and used to resample particles via multinomial resampling.
    - The joint distribution should reflect the prior beliefs about both free and constrained
      variables. Ideally, it captures correlations between hierarchical levels.
    - This method is particularly useful for complex nonlinear relationships where
      the constraint function cannot be inverted analytically.

    Examples
    --------
    Example: Multiple constraints
        >>> n_free = 3
        >>> n_constrained = 2
        >>> n_total = n_free + n_constrained
        >>>
        >>> joint_mean = np.ones(n_total) * 2.0
        >>> joint_cov = np.eye(n_total) * 0.3
        >>>
        >>> def f_u(B):
        ...     # B shape: (N, n_free)
        ...     # Two constraints: sum of first two frees, and third free
        ...     return np.vstack([
        ...         np.sum(B[:, :2], axis=1),    # (N,)
        ...         B[:, 2]                       # (N,)
        ...     ])  # Output: (n_constrained, N)
        >>>
        >>> result = reconc_nl_is(f_u, num_samples=5000, n_free=n_free,
        ...                        joint_mean=joint_mean, joint_cov=joint_cov, seed=42)
        >>> print("Reconciled shape:", result["reconciled_samples"].shape)
        Reconciled shape: (5, 5000)
    """

    if seed is not None:
        np.random.seed(seed)

    N = num_samples
    n_constrained = len(joint_mean) - n_free

    # ============================================================
    # TRUE JOINT MODE: one-shot weights using Gaussian on [U,B]
    # ============================================================
    if joint_mean is None or joint_cov is None:
        raise ValueError("IS requires joint_mean and joint_cov for the full vector [U,B].")

    bot_mean = joint_mean[n_constrained:]
    bot_cov = joint_cov[n_constrained:, n_constrained:]
    B = np.random.multivariate_normal(bot_mean, bot_cov, size=N)
    U_pred = f_u(B)  # expected shape (n_constrained, N)
    if U_pred.shape != (n_constrained, N):
        raise ValueError(f"f(B) must return shape (n_constrained, N)=({n_constrained},{N}), got {U_pred.shape}")

    # compute weights w_j ∝ p(U_pred_j, B_j) / p(B_j)
    w = _compute_weights(
            B=B,
            U_pred=U_pred.T,          # (N, n_constrained)
            joint_mean=joint_mean,    # (n_constrained+n_free,)
            joint_cov=joint_cov,      # (n_constrained+n_free, n_constrained+n_free)
            eps_cov=eps_cov,
    )

    # normalize safely
    w = np.maximum(w, 0.0)
    w = w / (w.sum() if w.sum() > 0 else 1.0)

    # resample ONCE
    idx = np.random.choice(N, size=N, replace=True, p=w)
    B = B[idx]

    U_final = f_u(B)
    return {
            "free_reconciled_samples": B.T,
            "constrained_reconciled_samples": U_final,
            "reconciled_samples": np.vstack([U_final, B.T]),
        }


def _mean_cov_from_params(params: Dict[str, float], distr: str) -> Tuple[np.ndarray, np.ndarray]:
    if distr == "gaussian":
        mu = np.array([params["mean"]])
        cov = np.array([[params["sd"] ** 2]])
        return mu, cov
    else:
        raise NotImplementedError(f"Distribution '{distr}' not implemented yet.")


def _aggregate_mean_cov(means: List[np.ndarray], covs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.concatenate(means)
    cov = np.block([
        [covs[i] if i == j else np.zeros((covs[i].shape[0], covs[j].shape[0]))
         for j in range(len(covs))]
        for i in range(len(covs))
    ])
    return mu, cov


def sample_multivariate_gaussian(mean: np.ndarray, cov: np.ndarray, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n_samples)


def unscented_transform(mu_x, Sigma_x, f, R, alpha=1e-3, beta=2, kappa=1):
    n = mu_x.shape[0]
    lam = 1
    gamma = 3

    Wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    Wc = np.copy(Wm)
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha ** 2 + beta)
    S = np.linalg.cholesky(Sigma_x + 1e-9 * np.eye(n))
    sigma_pts = np.zeros((2 * n + 1, n))
    sigma_pts[0] = mu_x
    for i in range(n):
        sigma_pts[i + 1] = mu_x + gamma * S[:, i]
        sigma_pts[n + i + 1] = mu_x - gamma * S[:, i]

    z_sigma = np.array([f(pt) for pt in sigma_pts])
    m = z_sigma.shape[1] if z_sigma.ndim > 1 else 1
    z_sigma = z_sigma.reshape(2 * n + 1, m)

    u_pred = np.sum(Wm[:, None] * z_sigma, axis=0)

    S_y = R.copy()
    P_xy = np.zeros((n, m))
    for i in range(2 * n + 1):
        dz = z_sigma[i] - u_pred
        dx = sigma_pts[i] - mu_x
        S_y += Wc[i] * np.outer(dz, dz)
        P_xy += Wc[i] * np.outer(dx, dz)

    K = P_xy @ np.linalg.inv(S_y)

    def condition_on(constrained_base_forecasts: np.ndarray):
        mu_post = mu_x + K @ (constrained_base_forecasts - u_pred)
        Sigma_post = Sigma_x - K @ S_y @ K.T
        Sigma_post = (Sigma_post + Sigma_post.T) / 2  # Symmetrize
        Sigma_post += 1e-6 * np.eye(Sigma_post.shape[0])  # Regularize
        return mu_post, Sigma_post

    return condition_on


def reconc_nl_ukf(
    free_base_forecasts: List[Dict[str, Union[float, np.ndarray]]],
    in_type: List[str],
    distr: List[str],
    f_u: Callable[[np.ndarray], np.ndarray],
    constrained_base_forecasts: np.ndarray,
    R: np.ndarray,
    num_samples: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:

    """
        Nonlinear Probabilistic Reconciliation using Unscented Kalman Filter (UKF) Conditioning.

    Uses the Unscented Kalman Filter to condition the free  forecast distributions
    on the constrained base forecasts via a user-defined nonlinear constraint function.
    The method performs probabilistic reconciliation by projecting the joint distribution onto the
    nonlinear manifold defined by f_u(free) = constrained_base_forecasts.

    Parameters
    ----------
    free_base_forecasts : list of dict
        A list of `n_free` dictionaries, one for each free-level forecast.

        - If `in_type[i] == "samples"`, `free_base_forecasts[i]` must contain:
          * `'samples'`: numpy.ndarray of shape (n_samples,) containing samples from the forecast.
          * `'residuals'`: numpy.ndarray of shape (n_samples,) containing residuals (samples - mean).

        - If `in_type[i] == "params"`, `free_base_forecasts[i]` must be a dictionary containing
          parameters for the specified distribution in `distr[i]`:
          * `'gaussian'`: {"mean": float, "sd": float}

    in_type : list of str
        Specifies the input type for each free forecast. Must have length `n_free`.
        Each element is one of:

        - `'samples'`: The forecast is provided as samples.
        - `'params'`: The forecast is provided as parameters.

        **Important**: All elements must be the same (either all 'samples' or all 'params').
        Mixed input types are not supported and will raise `NotImplementedError`.

    distr : list of str
        Specifies the distribution type for each free forecast. Must have length `n_free`.

        - If `in_type[i] == "samples"`: `distr[i]` should be descriptive (e.g., 'gaussian', 'continuous').
        - If `in_type[i] == "params"`: `distr[i]` must be one of:
          * `'gaussian'`: Gaussian distribution

    f_u : Callable[[np.ndarray], np.ndarray]
        The FTC function. Maps free variables to constrained
        variables via the relationship: `constrained = f_u(free)`.

        The function should:
        - Accept an array of shape `(n_free,)` representing the free variables.
        - Return an array of shape `(n_constrained,)` representing the constrained variables.
        - Be differentiable (required by the unscented transform).

    constrained_base_forecasts : numpy.ndarray
        Base forecast for the constrained (upper-level) variables.
        Shape: `(n_constrained,)`.
        These are the target values that the reconciled free forecasts should aggregate to
        (according to the constraint function f_u).

    R : numpy.ndarray
        Measurement noise covariance matrix representing uncertainty in the constrained forecasts.
        Shape: `(n_constrained, n_constrained)`.
        Must be symmetric and positive semi-definite.

    num_samples : int, optional
        Number of samples to draw from the reconciled distribution. Default is 10,000.

    seed : int or None, optional
        Random seed for reproducibility. Default is `None`.

    Returns
    -------
    dict
        A dictionary containing the reconciled forecasts:

        - `'free_reconciled_samples'`: numpy.ndarray
            Reconciled samples for the free (bottom-level) variables.
            Shape: `(n_free, num_samples)`.
            Each column is a sample from the reconciled distribution conditioned on the
            constrained base forecasts.

    Raises
    ------
    NotImplementedError
        If `in_type` contains mixed 'params' and 'samples' entries.

    ValueError
        If required keys are missing from parameter dictionaries or if sample arrays have
        incompatible shapes.

    KeyError
        If a required parameter key is missing (e.g., 'sd' for Gaussian distribution).

    Notes
    -----
    - The unscented Kalman filter uses sigma points to approximate the mean and covariance
      of the nonlinearly transformed distributions, avoiding the need for linearization.
    - All free forecasts must use the same input type. For mixed input types, first convert
      or reconstruct one type before calling this function.
    - The constraint function `f_u` should be well-defined for the range of the free forecasts.
    - The measurement noise covariance `R` should reflect the uncertainty in the constrained
      base forecasts. Smaller values increase the weight of the constrained forecasts.

    Examples
    --------

    Example : Multiplicative constraint with samples
        >>> rng = np.random.default_rng(42)
        >>> n_samples = 500
        >>> s0 = rng.normal(2.0, 0.2, n_samples)
        >>> s1 = rng.normal(3.0, 0.2, n_samples)
        >>>
        >>> free_base_forecasts = [
        ...     {"samples": s0, "residuals": s0 - s0.mean()},
        ...     {"samples": s1, "residuals": s1 - s1.mean()},
        ... ]
        >>> in_type = ["samples"] * 2
        >>> distr = ["gaussian"] * 2
        >>>
        >>> def multiplicative_constraint(free):
        ...     # free[0] * free[1] should equal constrained value
        ...     return free[0] * free[1]
        >>>
        >>> constrained_base_forecasts = np.array([6.0])
        >>> R = 0.1 * np.eye(1)
        >>>
        >>> result = reconc_nl_ukf(free_base_forecasts, in_type, distr,
        ...                         multiplicative_constraint, constrained_base_forecasts, R,
        ...                         num_samples=5000, seed=42)
        >>> print("Reconciled shape:", result["free_reconciled_samples"].shape)
        Reconciled shape: (2, 5000)
    """
    if all(t == "samples" for t in in_type):
        try:
            sample_mat = np.stack([bf["samples"] for bf in free_base_forecasts], axis=1)
            residual_mat = np.stack([bf["residuals"] for bf in free_base_forecasts], axis=1)
        except KeyError as e:
            raise ValueError(f"Missing key in sample-based input: {e}")

        mu_b = np.mean(sample_mat, axis=0)
        Sigma_b = _schafer_strimmer_cov(residual_mat)["shrink_cov"]

    elif all(t == "params" for t in in_type):
        free_means = []
        free_covs = []
        for i, bf in enumerate(free_base_forecasts):
            mu, cov = _mean_cov_from_params(bf, distr[i])
            free_means.append(mu)
            free_covs.append(cov)
        mu_b, Sigma_b = _aggregate_mean_cov(free_means, free_covs)

    else:
        raise NotImplementedError("Mixed 'params' and 'samples' input types are not supported.")

    ukf = unscented_transform(mu_b, Sigma_b, f_u, R)
    mu_post, Sigma_post = ukf(constrained_base_forecasts)

    B = sample_multivariate_gaussian(mu_post, Sigma_post, n_samples=num_samples, seed=seed).T
    #U = np.stack([f(B[:, i]) for i in range(B.shape[1])], axis=1)
    #Y = np.vstack([U, B])

    return {
        "free_reconciled_samples": B,
        #"constrained_reconciled_samples": U,
        #"reconciled_samples": Y
    }

