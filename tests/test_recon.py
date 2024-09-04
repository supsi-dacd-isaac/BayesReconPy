import unittest
import pandas as pd
from BayesReconPy.hierarchy import check_hierarchical
import numpy as np
from scipy.stats import binom, nbinom
import pytest

class TestScenarios(unittest.TestCase):
    def test_hierarchy(self, size=2):
        if size==2:
            A = [[1, 0], [1, 1]]
        elif size==3:
            A = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]

        assert check_hierarchical(A) == True

    def test_yyy(self):
        def test_PMF():
            # Helper functions
            def convolve_pmf(pmf1, pmf2):
                """Convolve two PMFs."""
                return np.convolve(pmf1, pmf2, mode='full')

            def get_mean(pmf):
                """Compute the mean of a PMF."""
                return np.sum(np.arange(len(pmf)) * pmf)

            def get_var(pmf, mean):
                """Compute the variance of a PMF."""
                return np.sum(((np.arange(len(pmf)) - mean) ** 2) * pmf)

            def get_quantile(pmf, p):
                """Compute the quantile of a PMF."""
                cdf = np.cumsum(pmf)
                return np.min(np.where(cdf >= p))

            # Test PMF convolution
            def test_pmf_conv():
                # Generate size and probability for each binomial
                s1, p1 = 20, 0.6
                s2, p2 = 30, 0.7

                # Compute the PMF for the two binomials
                x1 = np.arange(0, s1 + 1)
                x2 = np.arange(0, s2 + 1)
                pmf1 = binom.pmf(x1, s1, p1)
                pmf2 = binom.pmf(x2, s2, p2)

                # True mean of the convolution
                expected_conv_mean = s1 * p1 + s2 * p2
                expected_conv_var = s1 * p1 * (1 - p1) + s2 * p2 * (1 - p2)

                # Compute the PMF of the convolution
                pmf_conv = convolve_pmf(pmf1, pmf2)

                # Compute mean and variance of the convolution
                conv_mean = get_mean(pmf_conv)
                conv_var = get_var(pmf_conv, conv_mean)

                # Check if the convolution mean and variance are close to the true values
                assert abs(conv_mean - expected_conv_mean) < 1e-6
                assert abs(conv_var - expected_conv_var) < 8e-5

            # Test PMF bottom-up
            def test_pmf_bottom_up():
                # Test with 10 bottom
                n_bottom = 10

                # Create sizes and probabilities for negative binomial bottom distributions
                sizes = np.concatenate([np.arange(11, 16), np.arange(19, 15)])
                probs = np.concatenate([np.full(5, 0.4), np.full(5, 0.7)])

                # Compute true bottom-up parameters (mean/variance) and bottom PMFs
                true_bu_mean = 0
                true_bu_var = 0
                bott_pmfs = []

                for size, prob in zip(sizes, probs):
                    true_bu_mean += size * (1 - prob) / prob
                    true_bu_var += size * (1 - prob) / (prob ** 2)
                    pmf = nbinom.pmf(np.arange(0, size + 1), size=size, prob=prob)
                    bott_pmfs.append(pmf)

                # Combine PMFs using convolution
                bottom_up_pmf = bott_pmfs[0]
                for pmf in bott_pmfs[1:]:
                    bottom_up_pmf = convolve_pmf(bottom_up_pmf, pmf)

                # Compute mean and variance of the bottom-up PMF
                bottom_up_mean = get_mean(bottom_up_pmf)
                bottom_up_var = get_var(bottom_up_pmf, bottom_up_mean)

                # Check if true mean and variance are close to bottom-up PMF mean and variance
                assert abs(bottom_up_mean - true_bu_mean) / true_bu_mean < 2e-6
                assert abs(bottom_up_var - true_bu_var) / true_bu_var < 6e-5

            # Test PMF quantile
            def test_pmf_quantile():
                n_samples = int(1e5)
                size = 10
                prob = 0.6
                p = 0.01

                # Generate samples and compute PMF
                samples = nbinom.rvs(size=size, p=prob, size=n_samples)
                pmf, _ = np.histogram(samples, bins=np.arange(size + 2), density=True)

                # Compute the quantile and compare with the theoretical quantile
                q = get_quantile(pmf, p)
                qq = nbinom.ppf(p, size=size, prob=prob)

                assert np.isclose(q, qq)

        def test_get_hier_rows():
            def gen_monthly():
                """Generate a sample monthly hierarchy matrix."""
                # Example: 12 months x 3 categories per month
                return np.random.randint(0, 10, (12, 3))

            def gen_weekly():
                """Generate a sample weekly hierarchy matrix."""
                # Example: 52 weeks x 3 categories per week
                return np.random.randint(0, 10, (52, 3))

            def get_hier_rows(hier_matrix):
                """Compute the best hierarchical rows selection."""
                # Example optimization logic (to be replaced with the actual logic)
                num_rows = hier_matrix.shape[0]
                best_ind = np.random.choice([True, False], num_rows)
                return best_ind

            # Tests

            def test_get_hier_rows_monthly():
                month_hier = gen_monthly()
                ind_best = get_hier_rows(month_hier)

                best_obj_fun = max(np.sum(month_hier[ind_best, :], axis=0)) - np.sum(ind_best)

                assert best_obj_fun == -7

            def test_get_hier_rows_weekly():
                week_hier = gen_weekly()
                ind_best = get_hier_rows(week_hier)

                best_obj_fun = max(np.sum(week_hier[ind_best, :], axis=0)) - np.sum(ind_best)

                assert best_obj_fun == -37








if __name__ == '__main__':
    unittest.main()
