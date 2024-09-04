import numpy as np
import pytest

# Helper functions

def get_reconc_matrices(aggrs, h):
    """Compute reconciliation matrices."""
    sort_aggrs = sorted(aggrs, reverse=True)
    expected_rowSumsS = np.repeat(sort_aggrs, h // np.array(sort_aggrs))
    expectedLenRowSumsA = sum(h // np.array(sort_aggrs[:-1]))
    expected_rowSumsA = expected_rowSumsS[:expectedLenRowSumsA]

    # Example reconciliation matrices (should replace with actual implementation)
    A = np.eye(len(expected_rowSumsA))  # Placeholder
    S = np.eye(len(expected_rowSumsS))  # Placeholder

    return {'A': A, 'S': S}

def row_sums(matrix):
    """Compute the sum of each row in a matrix."""
    return np.sum(matrix, axis=1)

def get_Au(A):
    """Compute the matrix Au."""
    # Example implementation (should replace with actual implementation)
    return np.sum(A, axis=0)

def lowest_lev(A):
    """Compute the lowest level of A."""
    # Example implementation (should replace with actual implementation)
    return np.where(np.all(A == 1, axis=0))[0] + 1

# Tests

def test_get_reconc_matrices():
    aggrs = [1, 3, 6, 12]
    h = 12

    sort_aggrs = sorted(aggrs, reverse=True)
    expected_rowSumsS = np.repeat(sort_aggrs, h // np.array(sort_aggrs))
    expectedLenRowSumsA = sum(h // np.array(sort_aggrs[:-1]))
    expected_rowSumsA = expected_rowSumsS[:expectedLenRowSumsA]

    out = get_reconc_matrices(aggrs, h)

    diff = max(abs(expected_rowSumsA - row_sums(out['A']))) + max(abs(expected_rowSumsS - row_sums(out['S'])))
    assert diff == 0

def test_get_Au_and_lowest_lev():
    A = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]
    ])

    expected_Au = np.array([1, 1, 1])
    expected_lowest_lev = np.array([2, 3, 4])

    assert np.array_equal(get_Au(A), expected_Au)
    assert np.array_equal(lowest_lev(A), expected_lowest_lev)

def test_get_Au_different_row_order():
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

    A1 = np.array([
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

    assert np.array_equal(get_Au(A), get_Au(A1))

# To run the tests, use pytest from the command line
# pytest <this_script_name>.py
