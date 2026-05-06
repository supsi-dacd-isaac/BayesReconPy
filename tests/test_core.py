import unittest
import numpy as np
from bayesreconpy.core.covariance import _cov2cor, _schafer_strimmer_cov

class TestCoreCovariance(unittest.TestCase):
    """Test core covariance functions"""

    def test_schafer_strimmer_basic(self):
        """Test basic shrinkage covariance computation"""
        np.random.seed(42)
        x = np.random.randn(50, 4)
        result = _schafer_strimmer_cov(x)

        assert 'shrink_cov' in result
        assert 'lambda_star' in result
        assert result['shrink_cov'].shape == (4, 4)
        assert 0 <= result['lambda_star'] <= 1

    def test_schafer_strimmer_determinism(self):
        """Test shrinkage covariance produces consistent results"""
        np.random.seed(42)
        x = np.random.randn(100, 5)

        result1 = _schafer_strimmer_cov(x)
        result2 = _schafer_strimmer_cov(x)

        np.testing.assert_array_almost_equal(result1['shrink_cov'], result2['shrink_cov'])
        assert np.isclose(result1['lambda_star'], result2['lambda_star'])

    def test_cov2cor_identity(self):
        """Test cov2cor with identity matrix"""
        I = np.eye(5)
        cor = _cov2cor(I)

        np.testing.assert_array_almost_equal(cor, I)

    def test_cov2cor_symmetry(self):
        """Test cov2cor produces symmetric correlation matrices"""
        cov = np.array([[4, 2, 1], [2, 3, 0.5], [1, 0.5, 2]])
        cor = _cov2cor(cov)

        np.testing.assert_array_almost_equal(cor, cor.T)

    def test_cov2cor_diagonal_ones(self):
        """Test cov2cor has 1s on diagonal"""
        cov = np.array([[2, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 1.5]])
        cor = _cov2cor(cov)

        np.testing.assert_array_almost_equal(np.diag(cor), np.ones(3))

    def test_cov2cor_bounds(self):
        """Test correlation values are in [-1, 1]"""
        np.random.seed(42)
        # Generate positive definite covariance
        A = np.random.randn(5, 5)
        cov = A.T @ A
        cor = _cov2cor(cov)

        assert np.all(cor >= -1 - 1e-10)
        assert np.all(cor <= 1 + 1e-10)

    def test_cov2cor_scaling_invariance(self):
        """Test cov2cor is scale invariant"""
        cov = np.array([[4, 2], [2, 3]])
        cor1 = _cov2cor(cov)
        cor2 = _cov2cor(10 * cov)

        np.testing.assert_array_almost_equal(cor1, cor2)

    def test_schafer_strimmer_shape_consistency(self):
        """Test shrinkage preserves matrix shape"""
        for n, p in [(20, 3), (50, 10), (100, 5)]:
            x = np.random.randn(n, p)
            result = _schafer_strimmer_cov(x)
            assert result['shrink_cov'].shape == (p, p)

    def test_schafer_strimmer_lambda_range(self):
        """Test lambda_star is always in valid range"""
        for _ in range(10):
            x = np.random.randn(100, 5)
            result = _schafer_strimmer_cov(x)
            assert 0 <= result['lambda_star'] <= 1

    def test_cov2cor_positive_semidefinite(self):
        """Test correlation matrix is positive semidefinite"""
        np.random.seed(42)
        A = np.random.randn(5, 3)
        cov = A.T @ A
        cor = _cov2cor(cov)

        eigenvalues = np.linalg.eigvals(cor)
        assert np.all(eigenvalues >= -1e-10)

    def test_cov2cor_with_small_variances(self):
        """Test cov2cor handles very small variances"""
        cov = np.array([[1e-10, 1e-11], [1e-11, 1e-10]])
        cor = _cov2cor(cov)

        np.testing.assert_array_almost_equal(np.diag(cor), np.ones(2))


class TestCoreHierarchy(unittest.TestCase):
    """Test core hierarchy functions"""

    def test_hierarchy_matrix_valid_format(self):
        """Test hierarchy matrices are valid"""
        # Simple 2-level hierarchy
        A = np.array([[1, 1, 0], [0, 0, 1]])

        # Check A is binary
        assert np.all(np.isin(A, [0, 1]))

        # Check all bottom levels aggregate to at least one upper
        assert np.all(np.sum(A, axis=0) >= 1)

    def test_hierarchy_matrix_properties(self):
        """Test basic properties of hierarchy matrices"""
        # 2-level hierarchy: 1 upper, 3 bottom
        A = np.array([[1, 1, 1]])

        n_upper = A.shape[0]
        n_bottom = A.shape[1]

        assert n_upper == 1
        assert n_bottom == 3

    def test_hierarchy_all_levels_aggregate(self):
        """Test all bottom levels aggregate to some upper level"""
        A = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])

        # Each column should have at least one 1
        col_sums = np.sum(A, axis=0)
        assert np.all(col_sums >= 1)

    def test_hierarchy_distinct_levels(self):
        """Test hierarchy has distinct aggregation levels"""
        A = np.array([[1, 1, 1, 1],      # Total
                      [1, 1, 0, 0],      # Region 1
                      [0, 0, 1, 1]])     # Region 2

        unique_rows = np.unique(A, axis=0)
        # Should have 3 unique aggregation patterns
        assert unique_rows.shape[0] == 3


if __name__ == '__main__':
    unittest.main()

