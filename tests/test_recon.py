import unittest

from BayesReconPy.hierarchy import check_hierarchical


class TestScenarios(unittest.TestCase):
    def test_hierarchy(self, size=2):
        if size==2:
            A = [[1, 0], [1, 1]]
        elif size==3:
            A = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]

        assert check_hierarchical(A) == True

    def test_get_hier_rows_monthly(self):
        pass

if __name__ == '__main__':
    unittest.main()
