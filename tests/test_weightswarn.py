import numpy as np
import pytest
from scipy.stats import poisson, norm


# Placeholder functions for .compute_weights and .check_weights
def compute_weights(b, u, in_type, distr_type):
    # This is a placeholder implementation
    weights = u * b  # Example operation to generate weights
    return weights / np.sum(weights)  # Normalize weights


def check_weights(w, n_eff_min, p_n_eff=0.01):
    # Compute effective sample size
    n_eff = 1 / np.sum(w ** 2)

    # Initialize result dictionary
    result = {"warning": False, "warning_code": [], "n_eff": n_eff}

    if n_eff < n_eff_min:
        result["warning"] = True
        result["warning_code"].append(1)

    if n_eff / len(w) < p_n_eff:
        result["warning"] = True
        result["warning_code"].append(2)

    if n_eff < n_eff_min and n_eff / len(w) < p_n_eff:
        result["warning_code"].append(3)

    return result


def test_effective_sample_size():
    S = np.array([[1, 1, 1], [0, 0, 1]])

    # -----------
    n = 200
    b1 = poisson.rvs(mu=3, size=n)
    b2 = poisson.rvs(mu=4, size=n)
    u = norm.rvs(loc=30, scale=1, size=n)
    B = np.column_stack((b1, b2))
    c = S[0].reshape(-1, 1)
    b = B @ c

    w = compute_weights(b, u, "samples", "continuous")

    check_w = check_weights(w, n_eff_min=200)
    assert check_w["warning"] == True
    assert check_w["warning_code"] == [1]
    assert check_w["n_eff"] == n

    # -----------
    n = 199
    b1 = poisson.rvs(mu=3, size=n)
    b2 = poisson.rvs(mu=4, size=n)
    u = norm.rvs(loc=30, scale=1, size=n)
    B = np.column_stack((b1, b2))
    c = S[0].reshape(-1, 1)
    b = B @ c

    w = compute_weights(b, u, "samples", "continuous")

    check_w = check_weights(w, n_eff_min=200)
    assert check_w["warning"] == True
    assert check_w["warning_code"] == [1]
    assert check_w["n_eff"] == n

    # -----------
    n = 2000
    b1 = poisson.rvs(mu=3, size=n)
    b2 = poisson.rvs(mu=4, size=n)
    u = norm.rvs(loc=18, scale=1, size=n)
    B = np.column_stack((b1, b2))
    c = S[0].reshape(-1, 1)
    b = B @ c

    w = compute_weights(b, u, "samples", "continuous")

    check_w = check_weights(w, n_eff_min=200, p_n_eff=0.01)
    assert check_w["warning"] == True
    assert check_w["warning_code"] == [2, 3]
    assert check_w["n_eff"] < 200

# To run the tests, use pytest from the command line
# pytest <this_script_name>.py
