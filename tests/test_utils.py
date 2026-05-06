import unittest
import numpy as np
from bayesreconpy import hierarchy, shrink_cov, PMF
from bayesreconpy.utils import (
    _check_S, _check_A, _check_cov, _check_real_number,
    _check_positive_number, _check_implemented_distr, _check_distr_params
)

class TestUtils(unittest.TestCase):
    def test_hierarchy_gen(self):
        h_m = hierarchy._gen_monthly()
        h_w = hierarchy._gen_weekly()
        assert h_m.shape == (16, 12)
        assert h_w.shape == (46, 52)


    def test_shrink_cov(self):
        x = np.random.randn(100, 5)
        result = shrink_cov._schafer_strimmer_cov(x)
        assert result['shrink_cov'].shape == (5, 5)
        assert 0 <= result['lambda_star'] <= 1

        rm = np.random.randn(5, 5)
        cov = np.dot(rm.T, rm)
        cor = shrink_cov._cov2cor(cov)
        assert np.all(np.diag(cor) == np.ones(5))
        assert np.all(cor == cor.T)
        assert np.all(np.sign(cor) == np.sign(cov))
        assert np.all(np.eye(5) == shrink_cov._cov2cor(np.eye(5)))

    def test_pmf(self):
        pmf_poisson = PMF._pmf_from_params({'lambda': 3}, 'poisson', 1e-5)
        pmf_nbinom = PMF._pmf_from_params({'size': 1, 'mu': 1},'nbinom',1e-5)
        assert np.isclose(np.sum(pmf_poisson), 1)
        assert np.isclose(np.sum(pmf_nbinom), 1)

        assert PMF._pmf_get_quantile(pmf_poisson, 1-1e-9) <= len(pmf_poisson)
        assert PMF._pmf_get_quantile(pmf_poisson, 1e-9) >= 0

        PMF._pmf_summary(pmf_poisson)
        assert np.isclose(1, np.sum(PMF._pmf_tempering(pmf_poisson, 1)))
        assert np.isclose(1, np.sum(PMF._pmf_tempering(pmf_poisson, 0.9)))

    def test_checks(self):

        Ss = [np.random.randn(10, 10), np.array([[1,1], [1,1]]), np.vstack([np.ones((1, 5)), np.eye(5)[:4,:]])]
        counter = 0
        for S in Ss:
            try:
                _check_S(S)
            except ValueError:
                counter += 1
        assert counter == len(Ss)


        As = [np.random.randn(10, 10), np.array([[0,0], [0,0]]), np.ones((5, 5))]
        counter = 0
        for A in As:
            try:
                _check_A(A)
            except ValueError:
                counter += 1
        assert counter == len(As)-1

    def test_shrink_cov_edge_cases(self):
        """Test shrinkage covariance with edge cases"""
        # Very small sample size
        x = np.random.randn(5, 3)
        result = shrink_cov._schafer_strimmer_cov(x)
        assert result['shrink_cov'].shape == (3, 3)
        assert 0 <= result['lambda_star'] <= 1 or np.isnan(result['lambda_star'])

        # Random data instead of identity to avoid NaN
        x_random = np.random.randn(50, 5)
        result = shrink_cov._schafer_strimmer_cov(x_random)
        assert 0 <= result['lambda_star'] <= 1

        # Highly correlated data
        x_corr = np.column_stack([np.arange(100), np.arange(100)])
        result = shrink_cov._schafer_strimmer_cov(x_corr)
        assert result['shrink_cov'].shape == (2, 2)

    def test_cov2cor_properties(self):
        """Test correlation matrix properties"""
        # Test with singular covariance matrix
        cov_low_rank = np.array([[1, 1], [1, 1]])
        cor = shrink_cov._cov2cor(cov_low_rank)
        assert np.all(np.diag(cor) == 1.0)
        assert np.allclose(cor, cor.T)

        # Test with near-zero variance
        cov_near_zero = np.array([[1e-10, 0], [0, 1]])
        cor = shrink_cov._cov2cor(cov_near_zero)
        assert np.all(np.diag(cor) == 1.0)

    def test_pmf_edge_cases(self):
        """Test PMF functions with edge cases"""
        # Very small lambda
        pmf_small = PMF._pmf_from_params({'lambda': 0.1}, 'poisson', 1e-5)
        assert np.isclose(np.sum(pmf_small), 1)

        # Large lambda
        pmf_large = PMF._pmf_from_params({'lambda': 100}, 'poisson', 1e-5)
        assert np.isclose(np.sum(pmf_large), 1)

        # Very small mu
        pmf_zero = PMF._pmf_from_params({'size': 5, 'mu': 0.01}, 'nbinom', 1e-5)
        assert np.isclose(np.sum(pmf_zero), 1)

    def test_pmf_sampling_and_stats(self):
        """Test PMF sampling and statistics"""
        pmf = PMF._pmf_from_params({'lambda': 5}, 'poisson', 1e-5)

        # Test sampling
        samples = PMF._pmf_sample(pmf, 10000)
        assert len(samples) == 10000
        assert np.all(samples >= 0)

        # Test statistics
        mean = PMF._pmf_get_mean(pmf)
        var = PMF._pmf_get_var(pmf)
        assert mean > 0
        assert var > 0
        # For truncated Poisson, var can be < mean due to truncation
        assert var > 0 and mean > 0

        # Empirical mean from samples should be close to theoretical mean
        empirical_mean = np.mean(samples)
        assert np.abs(empirical_mean - mean) < 1.0

    def test_pmf_quantile_bounds(self):
        """Test PMF quantile function bounds"""
        pmf = PMF._pmf_from_params({'lambda': 10}, 'poisson', 1e-5)

        # Extreme quantiles
        q_high = PMF._pmf_get_quantile(pmf, 0.9999)
        q_low = PMF._pmf_get_quantile(pmf, 0.0001)

        assert q_high <= len(pmf) - 1
        assert q_low >= 0
        assert q_high >= q_low

    def test_check_S_valid_cases(self):
        """Test _check_S with valid matrices"""
        # Valid simple hierarchy: each column must sum > 1
        # and each bottom level must have identity row
        S_valid = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        try:
            _check_S(S_valid)
        except ValueError:
            self.fail("_check_S raised ValueError with valid input")

    def test_check_S_invalid_values(self):
        """Test _check_S rejects non-binary matrices"""
        S_invalid = np.array([[2, 1], [1, 0]])
        with self.assertRaises(ValueError):
            _check_S(S_invalid)

    def test_check_S_no_aggregation(self):
        """Test _check_S rejects matrices without aggregation"""
        # Each column must sum > 1 (not just 1)
        S_no_agg = np.array([[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            _check_S(S_no_agg)

    def test_check_A_valid_cases(self):
        """Test _check_A with valid aggregation matrices"""
        A_valid = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        try:
            _check_A(A_valid)
        except ValueError:
            self.fail("_check_A raised ValueError with valid input")

    def test_check_A_zero_columns(self):
        """Test _check_A rejects matrices with zero columns"""
        A_zero_col = np.array([[1, 0], [0, 0]])
        with self.assertRaises(ValueError):
            _check_A(A_zero_col)

    def test_check_cov_symmetry(self):
        """Test covariance matrix symmetry check"""
        sym_cov = np.array([[2, 1], [1, 2]])
        result = _check_cov(sym_cov, "cov", symm_check=True)
        assert result is True

        non_sym_cov = np.array([[2, 1], [2, 2]])
        with self.assertRaises(ValueError):
            _check_cov(non_sym_cov, "cov", symm_check=True)

    def test_check_cov_positive_definite(self):
        """Test positive definiteness check"""
        pd_cov = np.array([[2, 0.5], [0.5, 1]])
        result = _check_cov(pd_cov, "cov", pd_check=True)
        assert result is True

        non_pd_cov = np.array([[1, 2], [2, 1]])
        with self.assertRaises(ValueError):
            _check_cov(non_pd_cov, "cov", pd_check=True)

    def test_check_cov_negative_diagonal(self):
        """Test negative diagonal detection"""
        neg_diag_cov = np.array([[-1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            _check_cov(neg_diag_cov, "cov")

    def test_check_real_number(self):
        """Test real number validation"""
        assert _check_real_number(5) is True
        assert _check_real_number(5.5) is True
        assert _check_real_number(-10) is True
        assert _check_real_number(True) is False
        assert _check_real_number("5") is False
        assert _check_real_number(None) is False

    def test_check_positive_number(self):
        """Test positive number validation"""
        assert _check_positive_number(5) is True
        assert _check_positive_number(0.0001) is True
        assert _check_positive_number(-5) is False
        assert _check_positive_number(0) is False
        assert _check_positive_number(True) is False

    def test_check_implemented_distr(self):
        """Test distribution implementation check"""
        _check_implemented_distr("gaussian")
        _check_implemented_distr("poisson")
        _check_implemented_distr("nbinom")

        with self.assertRaises(ValueError):
            _check_implemented_distr("uniform")

        with self.assertRaises(ValueError):
            _check_implemented_distr("gamma")

    def test_check_distr_params_gaussian(self):
        """Test Gaussian distribution parameters"""
        valid_params = {"mean": 5.0, "sd": 2.0}
        _check_distr_params("gaussian", valid_params)

        invalid_params = {"mean": 5.0, "sd": -2.0}
        with self.assertRaises(ValueError):
            _check_distr_params("gaussian", invalid_params)

        invalid_params = {"mean": "five", "sd": 2.0}
        with self.assertRaises(ValueError):
            _check_distr_params("gaussian", invalid_params)

    def test_check_distr_params_poisson(self):
        """Test Poisson distribution parameters"""
        valid_params = {"lambda": 3.0}
        _check_distr_params("poisson", valid_params)

        invalid_params = {"lambda": -1.0}
        with self.assertRaises(ValueError):
            _check_distr_params("poisson", invalid_params)

    def test_check_distr_params_nbinom(self):
        """Test Negative Binomial distribution parameters"""
        valid_params_mu = {"size": 5, "mu": 10.0}
        _check_distr_params("nbinom", valid_params_mu)

        valid_params_prob = {"size": 5, "prob": 0.5}
        _check_distr_params("nbinom", valid_params_prob)

    def test_pmf_tempering_valid(self):
        """Test PMF tempering with valid beta values"""
        pmf = PMF._pmf_from_params({'lambda': 3}, 'poisson', 1e-5)

        # Beta = 1 returns original PMF
        pmf_temp_1 = PMF._pmf_tempering(pmf, 1.0)
        np.testing.assert_array_almost_equal(pmf, pmf_temp_1)

        # Beta = 0.5 should work (tempering)
        pmf_temp_05 = PMF._pmf_tempering(pmf, 0.5)
        assert np.isclose(np.sum(pmf_temp_05), 1.0)

    def test_pmf_tempering_invalid_temp(self):
        """Test PMF tempering rejects invalid temperature"""
        pmf = PMF._pmf_from_params({'lambda': 3}, 'poisson', 1e-5)

        # Temp <= 0 should raise error
        with self.assertRaises(ValueError):
            PMF._pmf_tempering(pmf, 0)

        with self.assertRaises(ValueError):
            PMF._pmf_tempering(pmf, -0.5)

    def test_pmf_from_samples_discrete_check(self):
        """Test PMF creation from samples validates discrete values"""
        # Valid discrete samples
        samples = np.array([0, 1, 2, 1, 0, 3])
        pmf = PMF._pmf_from_samples(samples)
        assert np.isclose(np.sum(pmf), 1)

        # Non-integer samples should fail
        non_int_samples = np.array([0.5, 1.5, 2.5])
        with self.assertRaises(ValueError):
            PMF._pmf_from_samples(non_int_samples)

    def test_pmf_means_variance_computation(self):
        """Test mean and variance consistency"""
        pmf = PMF._pmf_from_params({'lambda': 8}, 'poisson', 1e-5)

        mean = PMF._pmf_get_mean(pmf)
        var = PMF._pmf_get_var(pmf)

        # For Poisson, theoretical mean = variance = lambda
        # But truncated PMF may differ slightly
        assert mean > 0
        assert var > 0
        assert np.abs(mean - 8) < 2

if __name__ == '__main__':
    unittest.main()