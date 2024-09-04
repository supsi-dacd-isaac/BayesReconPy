import numpy as np
import pytest

# Helper functions

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

# To run the tests, use pytest from the command line
# pytest <this_script_name>.py
