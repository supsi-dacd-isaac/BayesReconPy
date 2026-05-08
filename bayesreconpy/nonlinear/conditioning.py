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
    Probabilistic reconciliation using Unscented Kalman conditioning on a user-defined nonlinear manifold.

    # EXAMPLE 1: Using 'params'
    free_base_forecasts = [
        {"mean": 1.0, "sd": 0.1},
        {"mean": 2.0, "sd": 0.1},
        {"mean": 3.0, "sd": 0.1},
    ]
    in_type = ["params"] * 3
    distr = ["gaussian"] * 3


    def f(free):
        middle1 = free[0] + free[1]
        middle2 = free[2]
        top = middle1 * middle2
        return np.array([top, middle1, middle2])


    constrained_base_forecasts = np.array([8.0, 3.0, 3.0])
    R = 0.01 * np.eye(3)

    reconciled = reconc_nl_ukf(free_base_forecasts, in_type, distr, f, constrained_base_forecasts, R, num_samples=1000, seed=42)
    print("Params input:", reconciled["free_reconciled_samples"].shape)

    # EXAMPLE 2: Using 'samples'
    rng = np.random.default_rng(42)
    s0 = rng.normal(1.0, 0.1, 500)
    s1 = rng.normal(2.0, 0.1, 500)
    s2 = rng.normal(3.0, 0.1, 500)
    r0 = s0 - s0.mean()
    r1 = s1 - s1.mean()
    r2 = s2 - s2.mean()

    free_base_forecasts = [
        {"samples": s0, "residuals": r0},
        {"samples": s1, "residuals": r1},
        {"samples": s2, "residuals": r2},
    ]
    in_type = ["samples"] * 3

    reconciled = reconc_nl_ukf(free_base_forecasts, in_type, distr, f, constrained_base_forecasts, R, num_samples=1000, seed=42)
    print("Samples input:", reconciled["reconciled_samples"].shape)
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

