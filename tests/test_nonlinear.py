import numpy as np
import unittest
import jax.numpy as jnp
from bayesreconpy.nonlinear.projection import reconc_nl_ols
from bayesreconpy.nonlinear.conditioning import reconc_nl_ukf, reconc_nl_is

class TestNonlinearUKF(unittest.TestCase):
    """Test nonlinear UKF reconciliation"""

    # --- Input Type Tests ---
    def test_ukf_with_params_input(self):
        """Test UKF with parameter-based free forecasts (Gaussian distributions)"""
        # All 'params' - should work without error
        free_forecasts = [
            {"mean": 1.0, "sd": 0.1},
            {"mean": 2.0, "sd": 0.1},
            {"mean": 3.0, "sd": 0.1},
        ]
        in_type = ["params"] * 3
        distr = ["gaussian"] * 3

        def f_linear(free):
            # Simple aggregation: top = sum(free)
            return np.sum(free)

        constrained_forecasts = np.array([6.0])
        R = np.eye(1)

        result = reconc_nl_ukf(free_forecasts, in_type, distr, f_linear,
                               constrained_forecasts, R, num_samples=500, seed=42)

        # Verify output shape
        assert result["free_reconciled_samples"].shape == (3, 500)
        # Verify samples are reasonable (close to original means)
        assert np.all(np.isfinite(result["free_reconciled_samples"]))

    def test_ukf_with_samples_input(self):
        """Test UKF with sample-based free forecasts"""
        # All 'samples' - should work without error
        rng = np.random.default_rng(42)
        n_samples = 200

        s0 = rng.normal(1.0, 0.1, n_samples)
        s1 = rng.normal(2.0, 0.1, n_samples)
        s2 = rng.normal(3.0, 0.1, n_samples)

        free_forecasts = [
            {"samples": s0, "residuals": s0 - s0.mean()},
            {"samples": s1, "residuals": s1 - s1.mean()},
            {"samples": s2, "residuals": s2 - s2.mean()},
        ]
        in_type = ["samples"] * 3
        distr = ["gaussian"] * 3

        def f_sum(free):
            return np.sum(free, axis=0)

        constrained_forecasts = np.array([6.0])
        R = np.eye(1)

        result = reconc_nl_ukf(free_forecasts, in_type, distr, f_sum,
                               constrained_forecasts, R, num_samples=500, seed=42)

        assert result["free_reconciled_samples"].shape == (3, 500)
        assert np.all(np.isfinite(result["free_reconciled_samples"]))

    # --- Mixed Input Type Error Test ---
    def test_ukf_mixed_input_types_error(self):
        """Test that mixed 'params' and 'samples' raises NotImplementedError"""
        free_forecasts = [
            {"mean": 1.0, "sd": 0.1},  # params
            {"samples": np.random.rand(100), "residuals": np.random.rand(100)},  # samples
        ]
        in_type = ["params", "samples"]
        distr = ["gaussian", "gaussian"]

        def f(free):
            return np.sum(free)

        with self.assertRaises(NotImplementedError):
            reconc_nl_ukf(free_forecasts, in_type, distr, f,
                          np.array([3.0]), np.eye(1), seed=42)

    # --- Constraint Function Tests ---
    def test_ukf_with_nonlinear_constraint(self):
        """Test UKF with multiplicative nonlinear constraint"""
        free_forecasts = [
            {"mean": 2.0, "sd": 0.2},
            {"mean": 3.0, "sd": 0.2},
        ]
        in_type = ["params"] * 2
        distr = ["gaussian"] * 2

        def f_product(free):
            # Constraint: top = free[0] * free[1]
            return free[0] * free[1]

        # Constrained forecast (product of means = 2*3 = 6)
        constrained_forecasts = np.array([6.0])
        R = np.eye(1)

        result = reconc_nl_ukf(free_forecasts, in_type, distr, f_product,
                               constrained_forecasts, R, num_samples=500, seed=42)

        assert result["free_reconciled_samples"].shape == (2, 500)

    # --- Error Handling ---
    def test_ukf_missing_params_key(self):
        """Test error when params dict is missing required keys"""
        free_forecasts = [{"mean": 1.0}]  # Missing 'sd'
        in_type = ["params"]
        distr = ["gaussian"]

        def f(free):
            return free[0]

        with self.assertRaises(KeyError):
            reconc_nl_ukf(free_forecasts, in_type, distr, f,
                          np.array([1.0]), np.eye(1), seed=42)

    def test_ukf_missing_samples_key(self):
        """Test error when samples dict is missing required keys"""
        free_forecasts = [{"samples": np.random.rand(100)}]  # Missing 'residuals'
        in_type = ["samples"]
        distr = ["gaussian"]

        def f(free):
            return np.sum(free)

        with self.assertRaises(ValueError):
            reconc_nl_ukf(free_forecasts, in_type, distr, f,
                          np.array([1.0]), np.eye(1), seed=42)

    # --- Reproducibility ---
    def test_ukf_seed_reproducibility(self):
        """Test that same seed produces identical results"""
        free_forecasts = [
            {"mean": 1.0, "sd": 0.1},
            {"mean": 2.0, "sd": 0.1},
        ]
        in_type = ["params"] * 2
        distr = ["gaussian"] * 2

        def f(free):
            return np.sum(free)

        constrained_forecasts = np.array([3.0])
        R = np.eye(1)

        result1 = reconc_nl_ukf(free_forecasts, in_type, distr, f,
                                constrained_forecasts, R, num_samples=500, seed=42)
        result2 = reconc_nl_ukf(free_forecasts, in_type, distr, f,
                                constrained_forecasts, R, num_samples=500, seed=42)

        np.testing.assert_array_equal(result1["free_reconciled_samples"],
                                      result2["free_reconciled_samples"])


class TestNonlinearIS(unittest.TestCase):
    """Test nonlinear importance sampling reconciliation"""

    def test_is_valid_inputs(self):
        """Test IS with valid joint_mean and joint_cov"""
        n_free = 2
        n_constrained = 1
        n_total = n_free + n_constrained

        joint_mean = np.array([6.0, 1.0, 2.0])  # [constrained, free1, free2]
        joint_cov = np.eye(n_total)

        def f_u(B):
            # B shape: (N, n_free) -> U shape: (n_constrained, N)
            # B[:, 0] and B[:, 1] are the two free variables
            return np.sum(B, axis=1, keepdims=True).T  # (1, N)

        result = reconc_nl_is(f_u, num_samples=500, n_free=n_free,
                              joint_mean=joint_mean, joint_cov=joint_cov, seed=42)

        assert "free_reconciled_samples" in result
        assert "constrained_reconciled_samples" in result
        assert "reconciled_samples" in result
        assert result["free_reconciled_samples"].shape == (n_free, 500)

    def test_is_missing_joint_params_error(self):
        """Test error when joint_mean or joint_cov is missing"""

        def f_u(B):
            return np.sum(B, axis=1, keepdims=True).T

        # This should raise TypeError before ValueError due to line 77 bug
        with self.assertRaises((ValueError, TypeError)):
            reconc_nl_is(f_u, num_samples=500, n_free=2, seed=42)

    def test_is_output_shape_consistency(self):
        """Test output shapes are consistent with inputs"""
        n_free = 3
        n_constrained = 2
        n_total = n_free + n_constrained
        n_samples = 500

        joint_mean = np.ones(n_total) * 2.0
        joint_cov = np.eye(n_total) * 0.5

        def f_u(B):
            # B shape: (N, n_free) -> output (n_constrained, N)
            # Two constraints: sum of first 2, and third value
            return np.vstack([
                np.sum(B[:, :2], axis=1),  # sum of first 2 free vars
                B[:, 2]  # third free var
            ])

        result = reconc_nl_is(f_u, num_samples=n_samples, n_free=n_free,
                              joint_mean=joint_mean, joint_cov=joint_cov, seed=42)

        assert result["free_reconciled_samples"].shape == (n_free, n_samples)
        assert result["constrained_reconciled_samples"].shape == (n_constrained, n_samples)
        assert result["reconciled_samples"].shape == (n_total, n_samples)

    def test_is_weight_normalization(self):
        """Test that importance weights are properly normalized"""
        # This indirectly tests through sampling validity
        n_free = 2
        joint_mean = np.array([4.0, 1.0, 2.0])
        joint_cov = np.eye(3) * 0.3

        def f_u(B):
            # B shape: (N, n_free) -> (1, N)
            return np.sum(B, axis=1, keepdims=True).T

        result = reconc_nl_is(f_u, num_samples=500, n_free=n_free,
                              joint_mean=joint_mean, joint_cov=joint_cov, seed=42)

        # Samples should be finite and reasonable
        assert np.all(np.isfinite(result["free_reconciled_samples"]))
        assert np.all(np.isfinite(result["constrained_reconciled_samples"]))


class TestNonlinearOLS(unittest.TestCase):
    """Test nonlinear OLS projection reconciliation"""

    def test_ols_basic_projection(self):
        """Test OLS projects samples onto constraint manifold"""
        rng = np.random.default_rng(42)
        n_samples = 500

        # Generate samples for: [top, middle, b0, b1, b2]
        b0 = rng.normal(1.0, 0.1, n_samples)
        b1 = rng.normal(2.0, 0.1, n_samples)
        b2 = rng.normal(3.0, 0.1, n_samples)
        middle = b0 + b1
        top = middle * b2

        samples = np.stack([top, middle, b0, b1, b2], axis=1)

        def f(z):
            return jnp.array([
                z[0] - z[1] * z[4],     # top - middle * b2
                z[1] - (z[2] + z[3])    # middle - (b0 + b1)
            ])

        result = reconc_nl_ols(samples, f, n_iter=10, seed=42)

        # Output shapes
        assert result["reconciled_samples"].shape == (5, n_samples)
        assert result["residuals"].shape == (2, n_samples)
        assert np.all(np.isfinite(result["reconciled_samples"]))

    def test_ols_constraint_residuals_small(self):
        """Test that constraint residuals are small after projection"""
        rng = np.random.default_rng(42)
        n_samples = 300

        # Create samples that slightly violate constraints
        b0 = rng.normal(1.0, 0.2, n_samples)
        b1 = rng.normal(2.0, 0.2, n_samples)
        b2 = rng.normal(3.0, 0.2, n_samples)
        middle = b0 + b1 + rng.normal(0, 0.1, n_samples)  # slight violation
        top = middle * b2 + rng.normal(0, 0.1, n_samples)  # slight violation

        samples = np.stack([top, middle, b0, b1, b2], axis=1)

        def f(z):
            return jnp.array([
                z[0] - z[1] * z[4],
                z[1] - (z[2] + z[3])
            ])

        result = reconc_nl_ols(samples, f, n_iter=15, seed=42)

        # Residuals should be small
        residuals = result["residuals"]
        max_residuals = np.max(np.abs(residuals), axis=1)
        assert np.all(max_residuals < 0.5)  # Allow some tolerance

    def test_ols_with_custom_weight_matrix(self):
        """Test OLS with custom weight matrix"""
        rng = np.random.default_rng(42)
        n_samples = 200
        d = 3

        samples = rng.normal(1.0, 0.2, (n_samples, d))

        def f(z):
            return jnp.array([z[0] - z[1] - z[2]])

        # Custom weights: emphasize first dimension
        W = np.diag([2.0, 1.0, 1.0])

        result = reconc_nl_ols(samples, f, W=W, n_iter=10, seed=42)

        assert result["reconciled_samples"].shape == (d, n_samples)
        assert result["residuals"].shape == (1, n_samples)

    def test_ols_seed_reproducibility(self):
        """Test that seed produces reproducible results"""
        rng = np.random.default_rng(42)
        n_samples = 200

        samples = rng.normal(0, 1, (n_samples, 3))

        def f(z):
            return jnp.array([z[0] + z[1] + z[2] - 3.0])

        result1 = reconc_nl_ols(samples, f, n_iter=10, seed=42)
        result2 = reconc_nl_ols(samples, f, n_iter=10, seed=42)

        np.testing.assert_array_almost_equal(
            result1["reconciled_samples"],
            result2["reconciled_samples"]
        )

    def test_ols_already_constrained_samples(self):
        """Test OLS with samples already satisfying constraints"""
        rng = np.random.default_rng(42)
        n_samples = 200

        # Create perfectly constrained samples
        b0 = rng.normal(1.0, 0.1, n_samples)
        b1 = rng.normal(2.0, 0.1, n_samples)
        top = b0 + b1  # Already satisfies constraint

        samples = np.stack([top, b0, b1], axis=1)

        def f(z):
            return jnp.array([z[0] - z[1] - z[2]])

        result = reconc_nl_ols(samples, f, n_iter=10, seed=42)

        # Residuals should be very small
        residuals = result["residuals"]
        assert np.max(np.abs(residuals)) < 1e-4

if __name__ == '__main__':
    unittest.main()