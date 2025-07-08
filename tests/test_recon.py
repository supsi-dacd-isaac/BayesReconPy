import unittest
import numpy as np
import pandas as pd
from bayesreconpy.reconc_buis import reconc_buis
from bayesreconpy.reconc_gaussian import reconc_gaussian
from bayesreconpy.utils import _gen_gaussian as gen_gaussian_samples
from bayesreconpy.utils import _gen_poisson as gen_poisson_samples
from bayesreconpy.hierarchy import _check_hierarchical, _lowest_lev, _temporal_aggregation, _get_reconc_matrices
from bayesreconpy.reconc_mcmc import reconc_mcmc
from bayesreconpy.PMF import _pmf_get_var as PMF_get_var
from bayesreconpy.PMF import _pmf_get_mean as PMF_get_mean
from bayesreconpy.PMF import _pmf_from_samples as PMF_from_samples
from bayesreconpy.reconc_mix_cond import reconc_mix_cond
from bayesreconpy.reconc_td_cond import reconc_td_cond
from bayesreconpy.reconc_ols import reconc_ols, get_S_from_A
from bayesreconpy.reconc_mint import reconc_mint, estimate_cov_matrix

class TestScenarios(unittest.TestCase):
    def test_hierarchy(self, size=2):
        if size==2:
            A = [[1, 0], [1, 1]]
        elif size==3:
            A = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]

        assert _check_hierarchical(A) == True

    def test_temporal_aggregation(self):
        # inputa data, DataFrame, the time series data with columns as months and index as years.
        A = pd.DataFrame(np.random.randn(10, 12), columns=range(1, 13), index=range(2010, 2020))
        # list of int, aggregation levels in terms of months per period.
        agg = _temporal_aggregation(A, [1, 2, 3, 4, 6, 12])
        agg = _temporal_aggregation(A, [2, 3, 4, 6, 12])
        agg = _temporal_aggregation(A, None)

    def test_get_recon_matrices(self):
        agg_levels = [1, 2, 3, 4, 6, 12]
        h = 48
        matrices_dict = _get_reconc_matrices(agg_levels, h)

        aggs = np.sum([h // a if a != 1 else  0 for a in agg_levels ])
        assert matrices_dict['A'].shape[0] == aggs

    def test_weekly_in_type_params_distr_gaussian(self):
        A = pd.read_csv("tests/Weekly-Gaussian_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Weekly-Gaussian_basef.csv", header=None).values
        base_forecasts = [{"mean": row[0], "sd": row[1]} for row in base_forecasts_in]

        res_buis = reconc_buis(A, base_forecasts, in_type="params", distr="gaussian", num_samples=int(1e5), seed=42)
        res_gauss = reconc_gaussian(A, base_forecasts_in[:, 0], np.diag(base_forecasts_in[:, 1] ** 2))

        n_upper = A.shape[0]
        n_bottom = A.shape[1]
        m = np.mean(np.mean(res_buis["reconciled_samples"], axis=1)[n_upper:n_upper + n_bottom] - res_gauss[
            "bottom_reconciled_mean"])
        assert abs(m) < 5e-2

    def test_monthly_in_type_params_distr_poisson(self):
        A = pd.read_csv("tests/Monthly-Poisson_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Monthly-Poisson_basef.csv", header=None).values
        base_forecasts = [{"lambda": row[0]} for row in base_forecasts_in]

        res_buis = reconc_buis(A, base_forecasts, in_type="params", distr="poisson", num_samples=100, seed=42)
        assert res_buis is not None

    def test_monthly_in_type_params_distr_nbinom(self):
        A = pd.read_csv("tests/Monthly-NegBin_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Monthly-NegBin_basef.csv", header=None).values
        base_forecasts = [{"size": row[1], "mu": row[0]} for row in base_forecasts_in]

        res_buis = reconc_buis(A, base_forecasts, in_type="params", distr="nbinom", num_samples=100, seed=42)
        assert res_buis is not None

    def test_monthly_in_type_samples_distr_continuous(self):
        A = pd.read_csv("tests/Monthly-Gaussian_A.csv", header=None).values
        base_forecasts = gen_gaussian_samples("tests/Monthly-Gaussian_basef.csv", seed=42)

        res_buis_samples = reconc_buis(A, base_forecasts, in_type="samples", distr="continuous", seed=42)

        base_forecasts_in = pd.read_csv("tests/Monthly-Gaussian_basef.csv", header=None).values
        base_forecasts = [{"mean": row[0], "sd": row[1]} for row in base_forecasts_in]

        res_buis = reconc_buis(A, base_forecasts, in_type="params", distr="gaussian", num_samples=int(1e6), seed=42)

        m = np.mean(
            np.mean(res_buis["reconciled_samples"], axis=0) - np.mean(res_buis_samples["reconciled_samples"],
                                                                      axis=0))
        assert abs(m) < 5e-2

    def test_monthly_in_type_samples_distr_discrete(self):
        A = pd.read_csv("tests/Monthly-Poisson_A.csv", header=None).values
        base_forecasts = gen_poisson_samples("tests/Monthly-Poisson_basef.csv", seed=42)

        res_buis_samples = reconc_buis(A, base_forecasts, in_type="samples", distr="discrete", seed=42)

        base_forecasts_in = pd.read_csv("tests/Monthly-Poisson_basef.csv", header=None).values
        base_forecasts = [{"lambda": row[0]} for row in base_forecasts_in]

        res_buis = reconc_buis(A, base_forecasts, in_type="params", distr="poisson", num_samples=int(1e6), seed=42)

        m = np.mean(
            np.mean(res_buis["reconciled_samples"], axis=0) - np.mean(res_buis_samples["reconciled_samples"],
                                                                      axis=0))
        assert abs(m) < 5e-2

    def test_mcmc_monthly_in_type_params_distr_poisson(self):
        # Load matrices and base forecasts
        A = pd.read_csv("tests/Monthly-Poisson_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Monthly-Poisson_basef.csv", header=None).values
        base_forecasts = [{"lambda": row[0]} for row in base_forecasts_in]

        # Perform reconciliations
        res_buis = reconc_buis(A, base_forecasts, in_type="params", distr="poisson", num_samples=int(1e4), seed=42)
        res_mcmc = reconc_mcmc(A, base_forecasts=base_forecasts, distr="poisson", num_samples=int(1e4), seed=42)

        # Compute the relative difference between the two methods
        m = (np.mean(res_buis["reconciled_samples"], axis=0) - np.mean(res_mcmc["reconciled_samples"],
                                                                       axis=0)) / np.mean(
            res_buis["reconciled_samples"], axis=0)

        # Assert that the maximum absolute difference is within the acceptable range
        assert np.max(np.abs(m)) < 0.5


    def test_reconc_mix_cond_simple_example(self):
        # Define matrix A
        A = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ])

        # Define means and vars for the forecasts
        means = np.array([90, 62, 63, 64, 31, 32, 31, 33, 31, 32] + [15] * 12)
        vars_ = np.array([20, 8, 8, 8, 4, 4, 4, 4, 4, 4] + [2] * 12) ** 2

        # Upper forecasts
        fc_upper = {
            "mu": means[:10],
            "Sigma": np.diag(vars_[:10])
        }

        # Bottom samples
        fc_bottom = []
        for i in range(A.shape[1]):
            samples = np.random.normal(loc=means[i + 10], scale=np.sqrt(vars_[i + 10]), size=20000).astype(int)
            samples[samples < 0] = 0  # set negatives to zero
            fc_bottom.append(samples)

        # Run reconc_mix_cond
        res_MixCond = reconc_mix_cond(A, fc_bottom, fc_upper, bottom_in_type="samples", seed=42)

        bott_rec_means = np.array([PMF_get_mean(pmf) for pmf in res_MixCond["bottom_reconciled"]["pmf"]])
        bott_rec_vars = np.array([PMF_get_var(pmf) for pmf in res_MixCond["bottom_reconciled"]["pmf"]])

        # Create PMF from samples
        fc_bottom_pmf = [PMF_from_samples(samples) for samples in fc_bottom]
        fc_bottom_dict = {i: arr for i, arr in enumerate(fc_bottom_pmf)}


        # Reconcile from bottom PMF
        res_MixCond_pmf = reconc_mix_cond(A, fc_bottom_dict, fc_upper, seed=42, num_samples=10000)

        bott_rec_means_pmf = np.array([PMF_get_mean(pmf) for pmf in res_MixCond_pmf["bottom_reconciled"]["pmf"]])
        bott_rec_vars_pmf = np.array([PMF_get_var(pmf) for pmf in res_MixCond_pmf["bottom_reconciled"]["pmf"]])

        assert np.allclose(bott_rec_means, bott_rec_means_pmf, rtol=5e-2)
        assert np.allclose(bott_rec_vars, bott_rec_vars_pmf, rtol=8e-2)


    def test_reconc_td_cond_simple_example(self):
        # Define matrix A
        A = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        ])

        # Define means and vars for the forecasts
        means = np.array([90, 31, 32, 31, 33, 31, 32, 62, 63, 64] + [15] * 12)
        vars_ = np.array([20, 4, 4, 4, 4, 4, 4, 8, 8, 8] + [2] * 12) ** 2

        # Upper forecasts
        fc_upper = {
            "mu": means[:10],
            "Sigma": np.diag(vars_[:10])
        }

        # Bottom samples
        fc_bottom = []
        for i in range(A.shape[1]):
            samples = np.random.normal(loc=means[i + 10], scale=np.sqrt(vars_[i + 10]), size=200).astype(int)
            samples[samples < 0] = 0  # Set negatives to zero
            fc_bottom.append(samples)

        # Reconciliation with TDcond
        res_TDcond = reconc_td_cond(A, fc_bottom, fc_upper, bottom_in_type="samples", num_samples=20000,
                                   return_type="pmf", seed=42)
        res_TDcond2 = reconc_td_cond(A, fc_bottom, fc_upper, bottom_in_type="samples", num_samples=20000,
                                    return_type="samples", seed=42)
        res_TDcond3 = reconc_td_cond(A, fc_bottom, fc_upper, bottom_in_type="samples", num_samples=20000,
                                    return_type="all", seed=42)

        # Check if all return_type return identical results
        pmf_1 = res_TDcond["bottom_reconciled"]["pmf"]
        pmf_samples = res_TDcond2["bottom_reconciled"]["samples"]
        pmf_3 = res_TDcond3["bottom_reconciled"]["pmf"]
        pmf_3_samples = res_TDcond3["bottom_reconciled"]["samples"]
        assert np.all([np.all(pm1==pm2) for pm1, pm2 in zip(pmf_1, pmf_3)])
        assert np.array_equal(pmf_samples, pmf_3_samples)

        # Analytical reconciliation (Gaussian assumptions)
        fc_bott_gauss = {
            "mu": means[10:],
            "Sigma": np.diag(vars_[10:])
        }

        inv_B = np.diag(1 / np.diag(fc_bott_gauss["Sigma"]))
        inv_U = np.diag(1 / np.diag(fc_upper["Sigma"]))
        At_inv_U_A = A.T @ inv_U @ A
        Au = A[_lowest_lev(A),:]
        inv_A_B_At = np.linalg.inv(Au @ fc_bott_gauss["Sigma"] @ Au.T)

        # Reconciled precision, covariance, and mean
        bott_reconc_Prec = inv_B + At_inv_U_A - Au.T @ inv_A_B_At @ Au
        bott_reconc_cov = np.linalg.inv(bott_reconc_Prec)
        bott_reconc_mean = fc_bott_gauss["mu"] + bott_reconc_cov @ A.T @ inv_U @ (
                    fc_upper["mu"] - A @ fc_bott_gauss["mu"])

        # Compute the difference between empirical and analytical
        m_diff = np.array([PMF_get_mean(pmf) for pmf in res_TDcond["bottom_reconciled"]["pmf"]]) - bott_reconc_mean

        assert np.all(np.abs(m_diff / bott_reconc_mean) < 5e-2)

    def create_mock_data(self, n_bottom=5, n_time=10, n_samples=50):
        """
        Creates mock data consistent with full hierarchical structure.
        """
        # Example: A maps 5 bottom series to 2 upper ones
        A = np.array([
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1]
        ])
        S = get_S_from_A(A)  # Shape [7, 5]
        n_total = S.shape[0]  # total = upper + bottom = 2 + 5 = 7

        # Create base forecasts at full level
        base_det = np.random.rand(n_total, n_time)
        base_samples = np.random.rand(n_total, n_time, n_samples)

        # Residuals for all series
        res = np.random.randn(n_time, n_total)

        return A, base_det, base_samples, res


    def test_reconc_ols_deterministic(self):
        A, base_det, _, _ = self.create_mock_data()
        y_rec = reconc_ols(A, base_det, samples=False)

        S = get_S_from_A(A)
        assert y_rec.shape == base_det.shape
        # Coherency check: aggregated series = S @ reconciled bottom series
        np.testing.assert_allclose(S @ np.linalg.pinv(S) @ y_rec, y_rec, rtol=1e-5)


    def test_reconc_ols_samples(self):
        A, _, base_samples, _ = self.create_mock_data()
        y_rec = reconc_ols(A, base_samples, samples=True)

        assert y_rec.shape == base_samples.shape
        # Optional: check that individual samples still match aggregation
        S = get_S_from_A(A)
        for i in range(base_samples.shape[2]):
            np.testing.assert_allclose(S @ np.linalg.pinv(S) @ y_rec[:, :, i], y_rec[:, :, i], rtol=1e-5)

    def test_reconc_mint_deterministic(self):
        A, base_det, _, res = self.create_mock_data()
        y_rec, var_rec = reconc_mint(A, base_det, res, samples=False)

        S = get_S_from_A(A)
        self.assertEqual(y_rec.shape, base_det.shape)
        self.assertEqual(var_rec.shape, (S.shape[0], S.shape[0]))
        np.testing.assert_allclose(var_rec, var_rec.T, rtol=1e-5)

    def test_reconc_mint_samples(self):
        A, _, base_samples, res = self.create_mock_data()
        y_rec, var_rec = reconc_mint(A, base_samples, res, samples=True)

        S = get_S_from_A(A)
        self.assertEqual(y_rec.shape, base_samples.shape)
        self.assertIsInstance(var_rec, list)
        for v in var_rec:
            self.assertEqual(v.shape, (S.shape[0], S.shape[0]))

if __name__ == '__main__':
    unittest.main()

