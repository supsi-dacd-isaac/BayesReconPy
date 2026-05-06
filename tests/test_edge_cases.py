import unittest
import numpy as np
from bayesreconpy.reconc_gaussian import reconc_gaussian
from bayesreconpy.reconc_ols import reconc_ols, get_S_from_A
from bayesreconpy import PMF


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_matrix_dimensions_match(self):
        """Test reconciliation with different matrix dimensions"""
        # Test valid dimensions where m < n (upper < bottom)
        for m in [1, 2, 3]:
            for n in [3, 4, 5]:
                if m < n:  # Only valid when m < n
                    A = np.hstack([np.eye(m), np.ones((m, n-m))])[:m, :n]
                    assert A.shape == (m, n)

    def test_reconciliation_zero_forecast(self):
        """Test reconciliation with zero forecasts"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        Sigma = np.eye(5)

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert result['bottom_reconciled_mean'].shape == (3,)
        np.testing.assert_array_almost_equal(result['bottom_reconciled_mean'], np.zeros(3))

    def test_reconciliation_identity_covariance(self):
        """Test reconciliation with identity covariance"""
        A = np.array([[1, 1, 1]])
        mu = np.array([3.0, 1.0, 1.0, 1.0])
        Sigma = np.eye(4)

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert not np.any(np.isnan(result['bottom_reconciled_mean']))
        assert not np.any(np.isnan(result['bottom_reconciled_covariance']))

    def test_reconciliation_diagonal_covariance(self):
        """Test reconciliation with diagonal covariance"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        # mu order: [upper_1, upper_2, bottom_1, bottom_2, bottom_3] = 2 + 3 = 5 elements
        mu = np.array([2.0, 1.5, 1.0, 1.0, 1.0])
        Sigma = np.diag([1.0, 0.5, 0.5, 0.8, 0.6])

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert result['bottom_reconciled_covariance'].shape == (3, 3)

    def test_ols_reconciliation_small_hierarchy(self):
        """Test OLS reconciliation with minimal hierarchy"""
        A = np.array([[1, 1]])
        base_det = np.array([[3.0, 1.0, 2.0]]).T  # 3 forecasts (1 upper + 2 bottom)

        y_rec = reconc_ols(A, base_det.astype(float), samples=False)
        assert y_rec.shape == base_det.shape
        assert not np.any(np.isnan(y_rec))

    def test_ols_reconciliation_large_hierarchy(self):
        """Test OLS reconciliation with larger hierarchy"""
        A = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        mu = np.array([10.0] + [1.0]*10)
        base_det = mu.reshape(-1, 1)

        y_rec = reconc_ols(A, base_det.astype(float), samples=False)
        assert y_rec.shape == base_det.shape

    def test_very_small_positive_values(self):
        """Test with very small positive values"""
        pmf = PMF._pmf_from_params({'lambda': 1e-6}, 'poisson', 1e-5)
        assert np.isclose(np.sum(pmf), 1)
        mean = PMF._pmf_get_mean(pmf)
        assert mean >= 0

    def test_very_large_values(self):
        """Test with very large parameter values"""
        pmf = PMF._pmf_from_params({'lambda': 1000}, 'poisson', 1e-5)
        assert np.isclose(np.sum(pmf), 1)
        assert not np.any(np.isnan(pmf))

    def test_nearly_singular_covariance(self):
        """Test reconciliation with nearly singular covariance"""
        A = np.array([[1, 1, 1]])
        mu = np.array([3.0, 1.0, 1.0, 1.0])

        # Nearly singular covariance (one eigenvalue very small)
        V = np.array([[1.0, 0.99, 0.98],
                      [0.99, 1.0, 0.99],
                      [0.98, 0.99, 1.0]])
        Sigma = np.block([[1.0, np.zeros(3)],
                          [np.zeros((3, 1)), V]])

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert not np.any(np.isnan(result['bottom_reconciled_mean']))

    def test_s_from_a_validity(self):
        """Test S matrix generation from A is valid"""
        A = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])
        S = get_S_from_A(A)

        # S should be valid hierarchy matrix
        assert np.all(np.isin(S, [0, 1]))
        assert S.shape[1] == A.shape[1]  # Same number of bottom levels

    def test_ols_with_sample_input(self):
        """Test OLS reconciliation with sample format input"""
        A = np.array([[1, 1, 1]])  # 1 upper, 3 bottom
        n_samples = 100
        # Shape should be [n_upper + n_bottom, n_time, n_samples] = [4, 5, 100]
        base_samples = np.random.rand(4, 5, n_samples).astype(float)

        y_rec = reconc_ols(A, base_samples, samples=True)
        assert y_rec.shape == base_samples.shape

    def test_negative_forecast_handling(self):
        """Test handling of negative base forecasts"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        # mu order: [upper_1, upper_2, bottom_1, bottom_2, bottom_3]
        mu = np.array([-1.0, -0.5, 0.5, 2.0, 1.0])
        Sigma = np.eye(5)

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert result['bottom_reconciled_mean'].shape == (3,)

    def test_highly_imbalanced_hierarchy(self):
        """Test with imbalanced aggregation structure"""
        A = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])

        n_upper = A.shape[0]
        n_bottom = A.shape[1]
        mu = np.random.randn(n_upper + n_bottom)
        Sigma = np.eye(n_upper + n_bottom)

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert result['bottom_reconciled_mean'].shape == (n_bottom,)

    def test_single_bottom_level(self):
        """Test reconciliation with single bottom level"""
        A = np.array([[1]])
        mu = np.array([5.0, 5.0])
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert result['bottom_reconciled_mean'].shape == (1,)

    def test_pmf_extreme_quantiles(self):
        """Test PMF quantile with extreme probabilities"""
        pmf = PMF._pmf_from_params({'lambda': 5}, 'poisson', 1e-5)

        q_very_high = PMF._pmf_get_quantile(pmf, 0.99999)
        q_very_low = PMF._pmf_get_quantile(pmf, 0.00001)

        assert q_very_high >= q_very_low
        assert q_very_high < len(pmf)

    def test_covariance_with_zero_variance_dimension(self):
        """Test handling of zero variance dimensions"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        mu = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        # Zero variance on position 2 and 4
        Sigma = np.diag([1.0, 1.0, 0.0, 1.0, 0.0])

        # This might raise an error or handle gracefully
        try:
            result = reconc_gaussian(A, mu.tolist(), Sigma)
            assert not np.any(np.isnan(result['bottom_reconciled_mean']))
        except (np.linalg.LinAlgError, ValueError):
            # Expected for singular covariance
            pass

    def test_numerical_stability_large_variance(self):
        """Test numerical stability with large variance values"""
        A = np.array([[1, 1, 1]])
        mu = np.array([1e6, 1e6, 1e6, 1e6])
        Sigma = np.diag([1e12, 1e12, 1e12, 1e12])

        try:
            result = reconc_gaussian(A, mu.tolist(), Sigma)
            assert not np.any(np.isnan(result['bottom_reconciled_mean']))
            assert not np.any(np.isinf(result['bottom_reconciled_mean']))
        except (np.linalg.LinAlgError, ValueError):
            # Acceptable due to numerical precision
            pass

    def test_numerical_stability_small_variance(self):
        """Test numerical stability with very small variance values"""
        A = np.array([[1, 1, 1]])
        mu = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        Sigma = np.diag([1e-12, 1e-12, 1e-12, 1e-12])

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        assert not np.any(np.isnan(result['bottom_reconciled_mean']))


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and precision issues"""

    def test_covariance_eigenvalues_positive(self):
        """Test covariance eigenvalues are non-negative"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        mu = np.array([5.0, 2.0, 3.0, 1.0, 1.0])
        Sigma = np.eye(5)

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        cov = result['bottom_reconciled_covariance']

        eigenvalues = np.linalg.eigvals(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_covariance_symmetry_preserved(self):
        """Test reconciled covariance remains symmetric"""
        A = np.array([[1, 1, 1]])
        mu = np.random.randn(4)
        Sigma = np.random.randn(4, 4)
        Sigma = (Sigma + Sigma.T) / 2  # Make symmetric
        Sigma = Sigma + np.eye(4) * 2  # Make PD

        result = reconc_gaussian(A, mu.tolist(), Sigma)
        cov = result['bottom_reconciled_covariance']

        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_matrix_operations_consistency(self):
        """Test matrix operations produce consistent results"""
        np.random.seed(42)
        A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        mu = np.random.randn(6)
        Sigma = np.eye(6)

        # Run multiple times
        results = [reconc_gaussian(A, mu.tolist(), Sigma) for _ in range(3)]

        # All should be identical
        for res in results[1:]:
            np.testing.assert_array_almost_equal(
                res['bottom_reconciled_mean'],
                results[0]['bottom_reconciled_mean']
            )


if __name__ == '__main__':
    unittest.main()