import numpy as np
import pytest
from scipy.linalg import cholesky
from scipy.stats import norm


# Simplified Schäfer-Strimmer shrinkage estimator function
def schaferStrimmer_cov(X):
    n, p = X.shape
    sample_cov = np.cov(X, rowvar=False)
    mean_diag = np.mean(np.diag(sample_cov))
    F = np.eye(p) * mean_diag
    lambda_star = max(0, min(1, (
                np.sum((sample_cov - F) ** 2) / np.sum((sample_cov - np.mean(np.diag(sample_cov))) ** 2))))
    shrinkage_cov = lambda_star * F + (1 - lambda_star) * sample_cov
    return {'lambda_star': lambda_star, 'cov': shrinkage_cov}


def test_shrinkage_estimator():
    # Parameters
    nSamples = 500
    pTrue = 2

    # True moments
    trueMean = np.array([0, 0])
    trueSigma = np.array([[3, 2], [2, 2]])
    chol_trueSigma = cholesky(trueSigma, lower=True)

    # Run 100 shrinkage estimators
    lambdas = []
    for _ in range(100):
        rr = np.tile(trueMean, (nSamples, 1)).T + chol_trueSigma @ np.random.normal(size=(pTrue, nSamples))

        # Estimate mean and covariance from samples
        mean_est = np.mean(rr, axis=1)
        Sigma_est = np.cov(rr)

        rr_centered = rr - np.tile(mean_est, (nSamples, 1)).T

        x = rr.T

        result = schaferStrimmer_cov(x)
        lambdas.append(result['lambda_star'])

    # The average over 100 runs must be within a certain range
    mean_lambdas = np.mean(lambdas)

    # Check if mean_lambdas is within the expected range
    assert mean_lambdas <= 0.005191634
    assert mean_lambdas >= 0.004843328

# To run the tests, use pytest from the command line
# pytest <this_script_name>.py
