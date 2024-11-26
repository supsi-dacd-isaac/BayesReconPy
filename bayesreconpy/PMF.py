import numpy as np
from scipy.stats import poisson, nbinom
from bayesreconpy.utils import DEFAULT_PARS


# Compute the empirical PMF from a vector of samples
def _pmf_from_samples(v):
    # Check that samples are discrete
    if not np.array_equal(v, v.astype(int)):
        raise ValueError("Input error: samples are not all discrete")
    else:
        v = v.astype(int)
    # Compute the PMF by tabulating the occurrences
    pmf = np.bincount(v) / len(v)  # Support starts from 0

    # Check for any negative samples
    if not np.isclose(np.sum(pmf), 1.0):
        raise ValueError("Input error: some samples are negative")

    return pmf


# Compute the PMF from a parametric distribution
def _pmf_from_params(params, distr, rtol=DEFAULT_PARS['RTOL']):
    # Validate the distribution type
    if distr not in ['poisson', 'nbinom']:
        raise ValueError(f"Input error: distr must be one of ['poisson', 'nbinom']")

    # Initialize PMF
    pmf = None

    # Compute the PMF based on the distribution type
    if distr == 'poisson':
        lambda_ = params['lambda']
        M = poisson.ppf(1 - rtol, lambda_).astype(int)
        pmf = poisson.pmf(np.arange(M + 1), lambda_)

    elif distr == 'nbinom':
        size = params['size']
        prob = params.get('prob')
        mu = params.get('mu')

        if prob is not None:
            M = nbinom.ppf(1 - rtol, n=size, p=prob).astype(int)
            pmf = nbinom.pmf(np.arange(M + 1), n=size, p=prob)

        elif mu is not None:
            M = nbinom.ppf(1 - rtol, n=size, p=size / (size + mu)).astype(int)
            pmf = nbinom.pmf(np.arange(M + 1), n=size, p=size / (size + mu))

    # Normalize the PMF to ensure it sums to 1
    pmf = pmf / np.sum(pmf)

    return pmf


def _pmf_sample(pmf, N_samples):
    pmf = np.atleast_1d(pmf)
    # Ensure that pmf is a normalized probability distribution
    if not np.isclose(np.sum(pmf), 1.0):
        raise ValueError("PMF must be normalized; the sum of probabilities should be 1.")

    # Create an array of possible outcomes
    outcomes = np.arange(len(pmf))

    # Sample from the PMF
    samples = np.random.choice(outcomes, size=N_samples, p=pmf)

    return samples


def _pmf_get_mean(pmf):
    pmf = np.atleast_1d(pmf)
    # Ensure that pmf is a normalized probability distribution
    if not np.isclose(np.sum(pmf), 1.0):
        raise ValueError("PMF must be normalized; the sum of probabilities should be 1.")

    # Create an array of possible outcomes
    supp = np.arange(len(pmf))

    # Calculate the mean
    mean = np.dot(pmf, supp)

    return mean


def _pmf_get_var(pmf):
    pmf = np.atleast_1d(pmf)
    # Ensure that pmf is a normalized probability distribution
    if not np.isclose(np.sum(pmf), 1.0):
        raise ValueError("PMF must be normalized; the sum of probabilities should be 1.")

    # Create an array of possible outcomes
    supp = np.arange(len(pmf))

    # Calculate the mean
    mean = np.dot(pmf, supp)

    # Calculate the second moment
    second_moment = np.dot(pmf, supp ** 2)

    # Calculate the variance
    variance = second_moment - mean ** 2

    return variance


def _pmf_get_quantile(pmf, p):
    # Check that the probability p is within the valid range
    if p <= 0 or p >= 1:
        raise ValueError("Input error: probability p must be between 0 and 1")

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(pmf)

    # Find the smallest index where CDF is greater than or equal to p
    x = np.min(np.where(cdf >= p)[0])

    return x


def _pmf_summary(pmf, Ltoll=DEFAULT_PARS['TOL'], rtol=DEFAULT_PARS['RTOL']):
    min_pmf = np.min(np.where(pmf > Ltoll)) - 1
    max_pmf = np.max(np.where(pmf > rtol)) - 1
    summary = {
        "Min.": min_pmf,
        "1st Qu.": _pmf_get_quantile(pmf, 0.25),
        "Median": _pmf_get_quantile(pmf, 0.5),
        "Mean": _pmf_get_mean(pmf),
        "3rd Qu.": _pmf_get_quantile(pmf, 0.75),
        "Max": max_pmf
    }
    return summary


def _pmf_smoothing(pmf, alpha=DEFAULT_PARS['ALPHA'], laplace=False):
    if alpha is None:
        alpha = np.min(pmf[pmf != 0])

    if np.any(pmf == 0):
        if laplace:
            pmf = pmf + alpha
        else:
            pmf[pmf == 0] = alpha

    return pmf / np.sum(pmf)


def _pmf_conv(pmf1, pmf2, toll=DEFAULT_PARS['TOL'], rtol=DEFAULT_PARS['RTOL']):
    # Convolution
    pmf = np.convolve(pmf1, pmf2, mode='full')

    # Trim values below rtol
    last_pos = np.max(np.where(pmf > rtol)) + 1
    pmf = pmf[:last_pos]

    # Set values below toll to zero
    pmf[pmf < toll] = 0

    # Set to zero values to the left of the minimal support
    m1 = np.min(np.where(pmf1 > 0))

    m2 = np.min(np.where(np.atleast_1d(pmf2) > 0))

    m = m1 + m2 - 1
    if m > 1:
        pmf[:m - 1] = 0

    return pmf / np.sum(pmf)


def _pmf_bottom_up(l_pmf, toll=DEFAULT_PARS['TOL'], rtol=DEFAULT_PARS['RTOL'], return_all=False, smoothing=True, alpha_smooth=DEFAULT_PARS['ALPHA_SMOOTHING'],
                  laplace_smooth=False):
    if smoothing:
        l_pmf = [_pmf_smoothing(pmf, alpha=alpha_smooth, laplace=laplace_smooth) for pmf in l_pmf]

    if len(l_pmf) == 1:
        return l_pmf if return_all else l_pmf[0]

    l_l_v = [l_pmf]
    while len(l_pmf) > 1:
        new_v = []
        for j in range(len(l_pmf) // 2):
            new_v.append(_pmf_conv(l_pmf[2 * j], l_pmf[2 * j + 1], toll=toll, rtol=rtol))
        if len(l_pmf) % 2 == 1:
            new_v.append(l_pmf[-1])

        l_pmf = new_v
        l_l_v.append(l_pmf)

    return l_l_v if return_all else l_pmf[0]


def _pmf_check_support(v_u, l_pmf, toll=DEFAULT_PARS['TOL'], rtol=DEFAULT_PARS['RTOL'], smoothing=True, alpha_smooth=DEFAULT_PARS['ALPHA_SMOOTHING'], laplace_smooth=False):
    # Compute the bottom-up PMF
    pmf_u = _pmf_bottom_up(l_pmf, toll=toll, rtol=rtol, return_all=False, smoothing=smoothing,
                          alpha_smooth=alpha_smooth, laplace_smooth=laplace_smooth)

    # Support of the PMF
    supp_u = np.where(pmf_u > 0)[0]
    # Check if elements of v_u are in the support of pmf_u
    mask = np.isin(v_u, supp_u)
    return mask


def _pmf_tempering(pmf, temp):
    if temp <= 0:
        raise ValueError("temp must be positive")
    if temp == 1:
        return pmf

    temp_pmf = np.power(pmf, 1 / temp)
    return temp_pmf / np.sum(temp_pmf)
