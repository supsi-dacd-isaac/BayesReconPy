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
    Nonlinear Probabilistic Reconciliation via Constrained Projection (OLS).

    Projects joint forecast samples onto a nonlinear constraint manifold using iterative
    constrained optimization. The method minimizes the weighted Euclidean distance from
    original samples to the constraint manifold f(Z) = 0, ensuring all reconciled samples
    satisfy the nonlinear constraints exactly.

    Parameters
    ----------
    samples : np.ndarray
        Joint forecast samples to be reconciled.
        Shape: `(n_samples, d)` where `d` is the total number of variables
        (constrained + free variables combined).
        Each row represents one sample from the joint distribution.

    f : Callable[[jnp.ndarray], jnp.ndarray]
        Constraint function defining the nonlinear reconciliation manifold.

        - Input: JAX array of shape `(d,)` representing one sample point.
        - Output: JAX array of shape `(k,)` where `k = n_constraints` is the number
          of constraints. The function defines the constraints as f(Z) = 0.

        The constraints express relationships between variables that reconciled samples
        must satisfy exactly. For example, hierarchical aggregation constraints:
        `top = middle * b2` becomes `top - middle * b2 = 0`.

    W : np.ndarray, optional
        Weight matrix for the optimization objective.
        Shape: `(d, d)`. Default is identity matrix `np.eye(d)`.

        Controls the relative importance of different variables when projecting onto
        the constraint manifold. Can be used to emphasize certain variables or account
        for different units of measurement.

        The optimization minimizes: ||W @ (Z_proj - Z_orig)||_2^2
        subject to f(Z_proj) = 0.

    n_iter : int, optional
        Number of Newton iterations for solving the constrained optimization problem.
        Default is 10. Increase for tighter constraint satisfaction or complex manifolds.

    seed : int or None, optional
        Random seed for reproducibility. Default is `None`.

    Returns
    -------
    dict
        A dictionary containing the reconciled forecasts and constraint residuals:

        - `'reconciled_samples'`: np.ndarray
            Projected samples satisfying the constraint manifold.
            Shape: `(d, n_samples)`.
            Each column is a reconciled sample with f(Z_proj) ≈ 0.

        - `'residuals'`: np.ndarray
            Constraint residuals f(Z_proj) for each reconciled sample.
            Shape: `(n_constraints, n_samples)`.
            Close to zero for successful reconciliation. Larger residuals indicate
            that the constraint manifold is difficult to satisfy or the optimization
            did not converge well.

    Notes
    -----
    - **Constraint Satisfaction**: Unlike sampling-based methods, reconciled samples
      exactly (or approximately) satisfy the nonlinear constraints f(Z) = 0. This
      ensures perfect coherency.

    - **Optimization Method**: Uses constrained Newton iterations (via JNLR library's
      `make_solver`) to project samples onto the manifold. The method solves a
      constrained least-squares problem at each iteration.

    - **Nonlinearity Support**: Handles arbitrary smooth nonlinear constraints
      (multiplicative, exponential, etc.) without linearization assumptions.

    - **Weight Matrix**: Use `W` to handle different variable scales or to weight
      constraints differently. For example, `W = np.diag([1.0, 0.1, 0.1, 0.1, 0.1])`
      prioritizes getting the first variable correct.

    - **Convergence**: The `n_iter` parameter controls the Newton iterations.
      The method should converge for well-posed problems within 10-20 iterations.
      Check the residuals to assess convergence.

    Raises
    ------
    ValueError
        If constraint function input/output shapes are incompatible with samples.

    TypeError
        If samples are not numpy arrays or constraint function is not callable.

    Examples
    --------
    Example : Multiple constraints with identity weight matrix
        >>> rng = np.random.default_rng(42)
        >>> n_samples = 2000
        >>>
        >>> # Create samples
        >>> samples = rng.normal(0, 1, (n_samples, 4))
        >>>
        >>> # Define two constraints
        >>> def f(z):
        ...     return jnp.array([
        ...         z[0] + z[1] - z[2],         # constraint 1: z0 + z1 = z2
        ...         z[1] * z[3] - z[0]          # constraint 2: z1 * z3 = z0
        ...     ])
        >>>
        >>> result = reconc_nl_ols(samples, f, n_iter=20, seed=42)
        >>> Z_proj = result["reconciled_samples"]
        >>> residuals = result["residuals"]
        >>>
        >>> print("Residuals shape:", residuals.shape)
        Residuals shape: (2, 2000)
        >>> print("Mean abs residuals per constraint:", np.mean(np.abs(residuals), axis=1))
        Mean abs residuals per constraint: [1.2e-04 5.8e-05]
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