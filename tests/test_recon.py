import unittest
import numpy as np
import pandas as pd
from bayesreconpy.reconc_BUIS import reconc_BUIS
from bayesreconpy.reconc_gaussian import reconc_gaussian
from bayesreconpy.utils import gen_gaussian as gen_gaussian_samples
from bayesreconpy.utils import gen_poisson as gen_poisson_samples
from bayesreconpy.hierarchy import check_hierarchical, lowest_lev
from bayesreconpy.reconc_MCMC import reconc_MCMC
from bayesreconpy.PMF import pmf_get_var as PMF_get_var
from bayesreconpy.PMF import pmf_get_mean as PMF_get_mean
from bayesreconpy.PMF import pmf_from_samples as PMF_from_samples
from bayesreconpy.reconc_MixCond import reconc_MixCond
from bayesreconpy.reconc_TDcond import reconc_TDcond

class TestScenarios(unittest.TestCase):
    def test_hierarchy(self, size=2):
        if size==2:
            A = [[1, 0], [1, 1]]
        elif size==3:
            A = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]

        assert check_hierarchical(A) == True


    def test_monthly_in_type_params_distr_gaussian(self):
        A = pd.read_csv("tests/Monthly-Gaussian_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Monthly-Gaussian_basef.csv", header=None).values
        base_forecasts = [{"mean": row[0], "sd": row[1]} for row in base_forecasts_in]

        res_buis = reconc_BUIS(A, base_forecasts, in_type="params", distr="gaussian", num_samples=100000, seed=42)
        res_gauss = reconc_gaussian(A, base_forecasts_in[:, 0], np.diag(base_forecasts_in[:, 1] ** 2))

        n_upper = A.shape[0]
        n_bottom = A.shape[1]
        m = np.mean(np.mean(res_buis["reconciled_samples"], axis=0)[n_upper:n_upper + n_bottom] - res_gauss[
            "bottom_reconciled_mean"])
        assert abs(m) < 8e-3

    def test_weekly_in_type_params_distr_gaussian(self):
        A = pd.read_csv("tests/Weekly-Gaussian_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Weekly-Gaussian_basef.csv", header=None).values
        base_forecasts = [{"mean": row[0], "sd": row[1]} for row in base_forecasts_in]

        res_buis = reconc_BUIS(A, base_forecasts, in_type="params", distr="gaussian", num_samples=100000, seed=42)
        res_gauss = reconc_gaussian(A, base_forecasts_in[:, 0], np.diag(base_forecasts_in[:, 1] ** 2))

        n_upper = A.shape[0]
        n_bottom = A.shape[1]
        m = np.mean(np.mean(res_buis["reconciled_samples"], axis=0)[n_upper:n_upper + n_bottom] - res_gauss[
            "bottom_reconciled_mean"])
        assert abs(m) < 2e-2

    def test_monthly_in_type_params_distr_poisson(self):
        A = pd.read_csv("tests/Monthly-Poisson_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Monthly-Poisson_basef.csv", header=None).values
        base_forecasts = [{"lambda": row[0]} for row in base_forecasts_in]

        res_buis = reconc_BUIS(A, base_forecasts, in_type="params", distr="poisson", num_samples=100000, seed=42)
        assert res_buis is not None

    def test_monthly_in_type_params_distr_nbinom(self):
        A = pd.read_csv("tests/Monthly-NegBin_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Monthly-NegBin_basef.csv", header=None).values
        base_forecasts = [{"size": row[1], "mu": row[0]} for row in base_forecasts_in]

        res_buis = reconc_BUIS(A, base_forecasts, in_type="params", distr="nbinom", num_samples=10000, seed=42)
        assert res_buis is not None

    def test_monthly_in_type_samples_distr_continuous(self):
        A = pd.read_csv("tests/Monthly-Gaussian_A.csv", header=None).values
        base_forecasts = gen_gaussian_samples("tests/Monthly-Gaussian_basef.csv", seed=42)

        res_buis_samples = reconc_BUIS(A, base_forecasts, in_type="samples", distr="continuous", seed=42)

        base_forecasts_in = pd.read_csv("tests/Monthly-Gaussian_basef.csv", header=None).values
        base_forecasts = [{"mean": row[0], "sd": row[1]} for row in base_forecasts_in]

        res_buis = reconc_BUIS(A, base_forecasts, in_type="params", distr="gaussian", num_samples=100000, seed=42)

        m = np.mean(
            np.mean(res_buis["reconciled_samples"], axis=0) - np.mean(res_buis_samples["reconciled_samples"],
                                                                      axis=0))
        assert abs(m) < 1e-2

    def test_monthly_in_type_samples_distr_discrete(self):
        A = pd.read_csv("tests/Monthly-Poisson_A.csv", header=None).values
        base_forecasts = gen_poisson_samples("tests/Monthly-Poisson_basef.csv", seed=42)

        res_buis_samples = reconc_BUIS(A, base_forecasts, in_type="samples", distr="discrete", seed=42)

        base_forecasts_in = pd.read_csv("tests/Monthly-Poisson_basef.csv", header=None).values
        base_forecasts = [{"lambda": row[0]} for row in base_forecasts_in]

        res_buis = reconc_BUIS(A, base_forecasts, in_type="params", distr="poisson", num_samples=100000, seed=42)

        m = np.mean(
            np.mean(res_buis["reconciled_samples"], axis=0) - np.mean(res_buis_samples["reconciled_samples"],
                                                                      axis=0))
        assert abs(m) < 1.5e-2

    def test_mcmc_monthly_in_type_params_distr_poisson(self):
        # Load matrices and base forecasts
        A = pd.read_csv("tests/Monthly-Poisson_A.csv", header=None).values
        base_forecasts_in = pd.read_csv("tests/Monthly-Poisson_basef.csv", header=None).values
        base_forecasts = [{"lambda": row[0]} for row in base_forecasts_in]

        # Perform reconciliations
        res_buis = reconc_BUIS(A, base_forecasts, in_type="params", distr="poisson", num_samples=100000, seed=42)
        res_mcmc = reconc_MCMC(A, base_forecasts=base_forecasts, distr="poisson", num_samples=100000, seed=42)

        # Compute the relative difference between the two methods
        m = (np.mean(res_buis["reconciled_samples"], axis=0) - np.mean(res_mcmc["reconciled_samples"],
                                                                       axis=0)) / np.mean(
            res_buis["reconciled_samples"], axis=0)

        # Assert that the maximum absolute difference is within the acceptable range
        assert np.max(np.abs(m)) < 0.02


    def test_reconc_MixCond_simple_example(self):
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

        # Run reconc_MixCond
        res_MixCond = reconc_MixCond(A, fc_bottom, fc_upper, bottom_in_type="samples", seed=42)

        bott_rec_means = np.array([PMF_get_mean(pmf) for pmf in res_MixCond["bottom_reconciled"]["pmf"]])
        bott_rec_vars = np.array([PMF_get_var(pmf) for pmf in res_MixCond["bottom_reconciled"]["pmf"]])

        # Create PMF from samples
        fc_bottom_pmf = [PMF_from_samples(samples) for samples in fc_bottom]

        # Reconcile from bottom PMF
        res_MixCond_pmf = reconc_MixCond(A, fc_bottom_pmf, fc_upper, seed=42, num_samples=1000000)

        bott_rec_means_pmf = np.array([PMF_get_mean(pmf) for pmf in res_MixCond_pmf["bottom_reconciled"]["pmf"]])
        bott_rec_vars_pmf = np.array([PMF_get_var(pmf) for pmf in res_MixCond_pmf["bottom_reconciled"]["pmf"]])

        assert np.allclose(bott_rec_means, bott_rec_means_pmf, rtol=3e-2)
        assert np.allclose(bott_rec_vars, bott_rec_vars_pmf, rtol=5e-2)


    def test_reconc_TDcond_simple_example(self):
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
            samples = np.random.normal(loc=means[i + 10], scale=np.sqrt(vars_[i + 10]), size=20000).astype(int)
            samples[samples < 0] = 0  # Set negatives to zero
            fc_bottom.append(samples)

        # Reconciliation with TDcond
        res_TDcond = reconc_TDcond(A, fc_bottom, fc_upper, bottom_in_type="samples", num_samples=20000,
                                   return_type="pmf", seed=42)
        res_TDcond2 = reconc_TDcond(A, fc_bottom, fc_upper, bottom_in_type="samples", num_samples=20000,
                                    return_type="samples", seed=42)
        res_TDcond3 = reconc_TDcond(A, fc_bottom, fc_upper, bottom_in_type="samples", num_samples=20000,
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
        Au = A[lowest_lev(A),:]
        inv_A_B_At = np.linalg.inv(Au @ fc_bott_gauss["Sigma"] @ Au.T)

        # Reconciled precision, covariance, and mean
        bott_reconc_Prec = inv_B + At_inv_U_A - Au.T @ inv_A_B_At @ Au
        bott_reconc_cov = np.linalg.inv(bott_reconc_Prec)
        bott_reconc_mean = fc_bott_gauss["mu"] + bott_reconc_cov @ A.T @ inv_U @ (
                    fc_upper["mu"] - A @ fc_bott_gauss["mu"])

        # Compute the difference between empirical and analytical
        m_diff = np.array([PMF_get_mean(pmf) for pmf in res_TDcond["bottom_reconciled"]["pmf"]]) - bott_reconc_mean

        assert np.all(np.abs(m_diff / bott_reconc_mean) < 8e-3)


if __name__ == '__main__':
    unittest.main()

