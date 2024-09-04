import numpy as np
from scipy.stats import binom, nbinom
import pytest

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
