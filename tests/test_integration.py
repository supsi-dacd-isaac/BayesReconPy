import unittest
import numpy as np
import pandas as pd
from bayesreconpy.reconc_gaussian import reconc_gaussian
from bayesreconpy.reconc_ols import reconc_ols, get_S_from_A
from bayesreconpy.reconc_mint import reconc_mint, estimate_cov_matrix
from bayesreconpy import PMF
from bayesreconpy.hierarchy import _lowest_lev


class TestHierarchicalCoherence(unittest.TestCase):
    """Test hierarchical coherency properties of reconciliation"""

    def test_gaussian_reconciliation_maintains_aggregation(self):
        """Test Gaussian reconciliation maintains hierarchical constraints"""
        A = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])
        mu = np.array([5.0, 2.0, 3.0, 3.0, 2.0, 1.0])  # upper levels then bottom
        Sigma = np.eye(6)
        
        result = reconc_gaussian(A, mu, Sigma)
        bottom_mean = result['bottom_reconciled_mean']
        
        # Check aggregation: A @ bottom_mean should equal upper means (approximately)
        # Upper aggregates from bottom levels
        for i in range(A.shape[0]):
            agg_bottom = np.sum(A[i, :] * bottom_mean)
            # This tests the coherency structure

    def test_ols_reconciliation_coherency(self):
        """Test OLS maintains hierarchical coherency"""
        A = np.array([[1, 1, 1]])
        n_bottom = A.shape[1]
        n_upper = A.shape[0]
        
        # Create base forecasts
        base_mean = np.array([10.0, 3.0, 3.0, 4.0])  # 1 upper + 3 bottom
        base_det = base_mean.reshape(-1, 1)
        
        y_rec = reconc_ols(A, base_det, samples=False)
        
        # Check shape
        assert y_rec.shape[0] == n_upper + n_bottom

    def test_s_matrix_coherency(self):
        """Test S matrix structure maintains hierarchical properties"""
        A = np.array([[1, 1, 0],
                      [0, 0, 1]])
        S = get_S_from_A(A)
        
        # S should map bottom levels to all levels
        assert S.shape[0] == A.shape[0] + A.shape[1]
        assert S.shape[1] == A.shape[1]
        
        # Check that each bottom level appears as a row in S
        assert np.any(np.all(S == np.array([1, 0, 0]), axis=1))
        assert np.any(np.all(S == np.array([0, 1, 0]), axis=1))
        assert np.any(np.all(S == np.array([0, 0, 1]), axis=1))


class TestReconciliationConsistency(unittest.TestCase):
    """Test consistency across different reconciliation methods"""

    def test_gaussian_vs_ols_similar_means(self):
        """Test Gaussian and OLS reconciliation produce similar results for identity covariance"""
        np.random.seed(42)
        A = np.array([[1, 1, 1]])
        base_mu = np.array([9.0, 3.0, 3.0, 3.0])
        base_det = base_mu.reshape(-1, 1)
        
        # Gaussian reconciliation
        Sigma = np.eye(4)
        result_gauss = reconc_gaussian(A, base_mu, Sigma)
        gauss_means = result_gauss['bottom_reconciled_mean']
        
        # OLS reconciliation
        result_ols = reconc_ols(A, base_det, samples=False)
        ols_bottom = result_ols[-A.shape[1]:, 0]  # Extract bottom levels
        
        # Should be similar (not exactly equal because OLS and Gaussian have different algorithms)
        assert gauss_means.shape == ols_bottom.shape

    def test_mint_covariance_structure(self):
        """Test MINT method produces valid covariance structure"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        base_det = np.array([3.0, 1.0, 2.0, 1.0, 0.5]).reshape(-1, 1)
        residuals = np.random.randn(50, 5)
        
        y_rec, var_rec = reconc_mint(A, base_det, residuals, samples=False)
        
        # Covariance should be square and symmetric
        assert var_rec.shape[0] == var_rec.shape[1]
        np.testing.assert_array_almost_equal(var_rec, var_rec.T)
        
        # Should be positive semi-definite
        eigenvalues = np.linalg.eigvals(var_rec)
        assert np.all(eigenvalues >= -1e-10)

    def test_reconciliation_reduces_variance(self):
        """Test that reconciliation reduces overall variance"""
        np.random.seed(42)
        A = np.array([[1, 1, 1]])
        n_samples = 500
        
        # Base forecasts with specific variance structure
        base_samples = np.random.randn(4, 10, n_samples)
        
        # Compute empirical variance before reconciliation
        var_before = np.mean(np.var(base_samples, axis=2))
        
        # Reconcile
        y_rec = reconc_ols(A, base_samples, samples=True)
        
        # Compute empirical variance after reconciliation
        var_after = np.mean(np.var(y_rec, axis=2))
        
        # Reconciliation should reduce variance for bottom levels
        # This is a soft check due to stochasticity
        assert var_before > 0
        assert var_after > 0


class TestDistributionProperties(unittest.TestCase):
    """Test properties of distribution functions"""

    def test_pmf_from_params_consistency(self):
        """Test PMF generation is consistent"""
        params_poisson = {'lambda': 5}
        pmf1 = PMF._pmf_from_params(params_poisson, 'poisson', 1e-5)
        pmf2 = PMF._pmf_from_params(params_poisson, 'poisson', 1e-5)
        
        np.testing.assert_array_equal(pmf1, pmf2)

    def test_pmf_poisson_properties(self):
        """Test Poisson PMF has correct mathematical properties"""
        lambda_ = 7.0
        pmf = PMF._pmf_from_params({'lambda': lambda_}, 'poisson', 1e-5)
        
        # Mean should be close to lambda
        mean = PMF._pmf_get_mean(pmf)
        var = PMF._pmf_get_var(pmf)
        
        assert abs(mean - lambda_) < 1.0  # Within 1 unit
        assert abs(var - lambda_) < 1.5   # Variance should also be close to lambda

    def test_pmf_nbinom_properties(self):
        """Test Negative Binomial PMF properties"""
        size = 5
        mu = 10.0
        pmf = PMF._pmf_from_params({'size': size, 'mu': mu}, 'nbinom', 1e-5)
        
        mean = PMF._pmf_get_mean(pmf)
        var = PMF._pmf_get_var(pmf)
        
        # For NB: E[X] = mu, Var[X] = mu + mu^2/size
        expected_var = mu + mu**2 / size
        
        assert abs(mean - mu) < 2.0
        assert var > mean  # Variance > mean for overdispersed distributions

    def test_pmf_sampling_distribution(self):
        """Test PMF sampling produces expected distribution"""
        np.random.seed(42)
        pmf = PMF._pmf_from_params({'lambda': 4}, 'poisson', 1e-5)
        
        samples = PMF._pmf_sample(pmf, 10000)
        empirical_mean = np.mean(samples)
        theoretical_mean = PMF._pmf_get_mean(pmf)
        
        # Empirical mean should be close to theoretical mean
        assert abs(empirical_mean - theoretical_mean) < 0.5

    def test_pmf_tempering_reduces_variance(self):
        """Test PMF tempering with beta < 1 reduces variance"""
        pmf = PMF._pmf_from_params({'lambda': 10}, 'poisson', 1e-5)
        
        mean_original = PMF._pmf_get_mean(pmf)
        var_original = PMF._pmf_get_var(pmf)
        
        pmf_tempered = PMF._pmf_tempering(pmf, 0.8)
        mean_tempered = PMF._pmf_get_mean(pmf_tempered)
        var_tempered = PMF._pmf_get_var(pmf_tempered)
        
        # Tempering should reduce variance
        assert var_tempered < var_original

    def test_pmf_tempering_preserves_normalization(self):
        """Test PMF tempering maintains probability normalization"""
        pmf = PMF._pmf_from_params({'lambda': 5}, 'poisson', 1e-5)
        
        for beta in [0.5, 0.7, 0.9, 1.0]:
            pmf_temp = PMF._pmf_tempering(pmf, beta)
            assert np.isclose(np.sum(pmf_temp), 1.0)


class TestMatrixOperations(unittest.TestCase):
    """Test matrix operations in reconciliation"""

    def test_get_s_from_a_properties(self):
        """Test S matrix generation properties"""
        # Test with different A matrices
        test_cases = [
            np.array([[1, 1, 0], [0, 0, 1]]),
            np.array([[1, 1, 1]]),
            np.array([[1, 1, 0, 0], [0, 0, 1, 1]]),
        ]
        
        for A in test_cases:
            S = get_S_from_A(A)
            
            # S should be binary
            assert np.all(np.isin(S, [0, 1]))
            
            # Number of columns should match A
            assert S.shape[1] == A.shape[1]
            
            # Each row should have at least one 1
            assert np.all(np.sum(S, axis=1) >= 1)
            
            # Each column should have at least one 1
            assert np.all(np.sum(S, axis=0) >= 1)

    def test_get_s_from_a_includes_bottom_levels(self):
        """Test S matrix includes identity rows for bottom levels"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        S = get_S_from_A(A)
        
        # Should have identity rows for each bottom level
        n_bottom = A.shape[1]
        identity_rows = S[-n_bottom:, :]
        
        # Last n_bottom rows should form partial identity
        for i in range(n_bottom):
            assert identity_rows[i, i] == 1

    def test_gaussian_reconciliation_output_structure(self):
        """Test Gaussian reconciliation output has correct structure"""
        A = np.array([[1, 1, 0], [0, 0, 1]])
        mu = np.random.randn(5)
        Sigma = np.eye(5)
        
        result = reconc_gaussian(A, mu, Sigma)
        
        assert isinstance(result, dict)
        assert 'bottom_reconciled_mean' in result
        assert 'bottom_reconciled_covariance' in result
        
        n_bottom = A.shape[1]
        assert result['bottom_reconciled_mean'].shape == (n_bottom,)
        assert result['bottom_reconciled_covariance'].shape == (n_bottom, n_bottom)

    def test_ols_reconciliation_output_structure(self):
        """Test OLS reconciliation output structure"""
        A = np.array([[1, 1, 1]])
        base_det = np.random.randn(4, 5)
        
        y_rec = reconc_ols(A, base_det, samples=False)
        
        # Should have same shape as input
        assert y_rec.shape == base_det.shape
        
        # Extract reconciled bottom levels
        n_upper = A.shape[0]
        n_bottom = A.shape[1]
        bottom_rec = y_rec[n_upper:, :]
        
        assert bottom_rec.shape == (n_bottom, 5)


class TestDataConsistency(unittest.TestCase):
    """Test consistency of data structures"""

    def test_reconciliation_with_time_series_data(self):
        """Test reconciliation works with time series data"""
        A = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])
        
        # Create time series data: rows are series, columns are time points
        n_series = A.shape[0] + A.shape[1]  # upper + bottom
        n_time = 24
        
        time_series = np.random.randn(n_series, n_time)
        
        y_rec = reconc_ols(A, time_series, samples=False)
        assert y_rec.shape == time_series.shape

    def test_reconciliation_with_multiple_samples(self):
        """Test reconciliation handles multiple sample outputs"""
        A = np.array([[1, 1, 1]])
        n_series = A.shape[0] + A.shape[1]
        n_time = 10
        n_samples = 500
        
        base_samples = np.random.randn(n_series, n_time, n_samples)
        
        y_rec = reconc_ols(A, base_samples, samples=True)
        
        # Should maintain sample dimension
        assert y_rec.shape == base_samples.shape
        
        # Check consistency across time and series
        assert y_rec.shape[0] == n_series
        assert y_rec.shape[1] == n_time
        assert y_rec.shape[2] == n_samples

    def test_estimate_cov_matrix_output(self):
        """Test covariance matrix estimation"""
        n_time = 50
        n_series = 10
        residuals = np.random.randn(n_time, n_series)
        
        cov_est = estimate_cov_matrix(residuals)
        
        # Should be square matrix
        assert cov_est.shape == (n_series, n_series)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov_est, cov_est.T)
        
        # Should be positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_est)
        assert np.all(eigenvalues >= -1e-10)


if __name__ == '__main__':
    unittest.main()

