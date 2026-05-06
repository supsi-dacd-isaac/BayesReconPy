import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Dict
from jnlr.reconcile import make_solver


def reconc_nl_ols(
    samples: np.ndarray,                         # shape (n_samples, d)
    f: Callable[[jnp.ndarray], jnp.ndarray],     # constraint function f(Z)
    W: Optional[np.ndarray] = None,
    n_iter: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Project joint samples Z ∈ ℝ^d onto the nonlinear constraint manifold f(Z) = 0.

    Parameters
    ----------
    samples : np.ndarray
        Joint forecast samples of shape (n_samples, d)

    f : Callable[[jnp.ndarray], jnp.ndarray]
        Constraint function defining the manifold

    n_constraints : int
        Dimensionality of the constraint (k = dim(f(Z)))

    W : np.ndarray or None
        Weight matrix for the objective (default: identity)

    n_iter : int
        Number of Newton iterations

    seed : int or None
        Random seed (for reproducibility)

    Returns
    -------
    dict
        {
            "reconciled_samples": np.ndarray of shape (d, n_samples),
            "residuals": np.ndarray of shape (n_constraints, n_samples)
        }

    EXAMPLE:
    # Generate 3D Gaussian samples: [top, middle, b0, b1, b2]
    rng = np.random.default_rng(42)
    n_samples = 5000
    b0 = rng.normal(1.0, 0.1, n_samples)
    b1 = rng.normal(2.0, 0.1, n_samples)
    b2 = rng.normal(3.0, 0.1, n_samples)

    middle = b0 + b1
    top = middle * b2
    samples = np.stack([top, middle, b0, b1, b2], axis=1)  # shape (n_samples, 5)

    # Define constraint function f(Z) = [top - middle * b2, middle - (b0 + b1)]
    def f(z):
        return jnp.array([
            z[0] - z[1] * z[4],     # top - middle * b2
            z[1] - (z[2] + z[3])    # middle - (b0 + b1)
        ])

    result = reconc_nl_ols(samples, f, n_constraints=2, n_iter=15)
    Z_proj = result["reconciled_samples"]
    res = result["residuals"]

    print("Projected sample shape:", Z_proj.shape)
    print("Constraint residuals (max abs):", np.max(np.abs(res), axis=1))
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, d = samples.shape
    if W is None:
        W = np.eye(d)

    W_jax = jnp.array(W)
    solver = make_solver(f, W_jax, n_iterations=n_iter,beta=1.0, damping=0.0)

    samples_jax = jnp.array(samples)
    proj = solver(samples_jax)  # shape (n_samples, d)

    proj_np = np.asarray(proj).T  # shape (d, n_samples)
    residuals_np = np.asarray(jax.vmap(f)(proj)).T  # shape (k, n_samples)

    return {
        "reconciled_samples": proj_np,
        "residuals": residuals_np
    }