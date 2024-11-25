import unittest
import numpy as np
import pandas as pd
from bayesreconpy import hierarchy, shrink_cov, PMF
from bayesreconpy.utils import check_S, check_A, check_cov

class TestUtils(unittest.TestCase):
    def test_hierarchy_gen(self):
        h_m = hierarchy.gen_monthly()
        h_w = hierarchy.gen_weekly()
        assert h_m.shape == (16, 12)
        assert h_w.shape == (46, 52)


    def test_shrink_cov(self):
        x = np.random.randn(100, 5)
        result = shrink_cov.schafer_strimmer_cov(x)
        assert result['shrink_cov'].shape == (5, 5)
        assert 0 <= result['lambda_star'] <= 1

        rm = np.random.randn(5, 5)
        cov = np.dot(rm.T, rm)
        cor = shrink_cov.cov2cor(cov)
        assert np.all(np.diag(cor) == np.ones(5))
        assert np.all(cor == cor.T)
        assert np.all(np.sign(cor) == np.sign(cov))
        assert np.all(np.eye(5) == shrink_cov.cov2cor(np.eye(5)))

    def test_pmf(self):
        pmf_poisson = PMF.pmf_from_params({'lambda': 3}, 'poisson', 1e-5)
        pmf_nbinom = PMF.pmf_from_params({'size': 1, 'mu': 1},'nbinom',1e-5)
        assert np.isclose(np.sum(pmf_poisson), 1)
        assert np.isclose(np.sum(pmf_nbinom), 1)

        assert PMF.pmf_get_quantile(pmf_poisson, 1-1e-9) <= len(pmf_poisson)
        assert PMF.pmf_get_quantile(pmf_poisson, 1e-9) >= 0

        PMF.pmf_summary(pmf_poisson)
        assert np.isclose(1, np.sum(PMF.pmf_tempering(pmf_poisson, 1)))
        assert np.isclose(1, np.sum(PMF.pmf_tempering(pmf_poisson, 0.9)))

    def test_checks(self):

        Ss = [np.random.randn(10, 10), np.array([[1,1], [1,1]]), np.vstack([np.ones((1, 5)), np.eye(5)[:4,:]])]
        counter = 0
        for S in Ss:
            try:
                check_S(S)
            except ValueError:
                counter += 1
        assert counter == len(Ss)


        As = [np.random.randn(10, 10), np.array([[0,0], [0,0]]), np.ones((5, 5))]
        counter = 0
        for A in As:
            try:
                check_A(A)
            except ValueError:
                counter += 1
        assert counter == len(As)-1
if __name__ == '__main__':
    unittest.main()
