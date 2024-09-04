import numpy as np
import pytest
from scipy.stats import norm, nbinom, poisson, multivariate_normal

def distr_sample(params, distr, n):
    if distr == "gaussian":
        return np.random.normal(loc=params['mean'], scale=params['sd'], size=int(n))
    elif distr == "nbinom":
        if 'prob' in params and 'mu' not in params:
            return nbinom.rvs(n=params['size'], p=params['prob'], size=int(n))
        elif 'mu' in params and 'prob' not in params:
            p = params['size'] / (params['size'] + params['mu'])
            return nbinom.rvs(n=params['size'], p=p, size=int(n))
        else:
            raise ValueError("Invalid parameters for nbinom")
    elif distr == "poisson":
        return poisson.rvs(mu=params['lambda'], size=int(n))
    else:
        raise ValueError("Unsupported distribution")

def MVN_sample(n, mu, Sigma):
    return np.random.multivariate_normal(mean=mu, cov=Sigma, size=int(n))

def MVN_density(x, mu, Sigma, max_size_x=None):
    if max_size_x is None:
        return multivariate_normal.pdf(x, mean=mu, cov=Sigma)
    else:
        res = []
        for i in range(0, len(x), max_size_x):
            chunk = x[i:i + max_size_x]
            res.append(multivariate_normal.pdf(chunk, mean=mu, cov=Sigma))
        return np.concatenate(res)

def test_sampling_univariate_normal():
    params = {'mean': 42, 'sd': 1}
    distr = "gaussian"
    n = 1e4
    samples = distr_sample(params, distr, n)

    sam_mean = np.mean(samples)
    sam_sd = np.std(samples)

    m = abs(sam_mean - 42) / 42
    s = abs(sam_sd - 1)

    assert m < 2e-3
    assert s < 4e-2

def test_sampling_univariate_nbinom():
    params = {'size': 12, 'prob': 0.8}
    distr = "nbinom"
    n = 1e4
    samples = distr_sample(params, distr, n)

    sam_mean = np.mean(samples)
    true_mean = params['size'] * (1 - params['prob']) / params['prob']

    m = abs(sam_mean - true_mean) / true_mean
    assert m < 3e-2

    params = {'size': 12, 'mu': true_mean}
    samples = distr_sample(params, distr, n)

    sam_mean = np.mean(samples)
    m = abs(sam_mean - params['mu']) / params['mu']
    assert m < 3e-2

    with pytest.raises(ValueError):
        params = {'size': 12, 'mu': true_mean, 'prob': 0.8}
        distr_sample(params, distr, n)

    with pytest.raises(ValueError):
        params = {'mu': true_mean, 'prob': 0.8}
        distr_sample(params, distr, n)

def test_sampling_univariate_poisson():
    params = {'lambda': 10}
    distr = "poisson"
    n = 1e4
    samples = distr_sample(params, distr, n)

    sam_mean = np.mean(samples)
    m = abs(sam_mean - 10) / 10

    assert m < 3e-2

def test_sampling_multivariate_normal():
    mu = [10, 10]
    Sigma = np.array([[1, 0.7], [0.7, 1]])
    n = 1e4
    samples = MVN_sample(n, mu, Sigma)

    sam_mean = np.mean(samples, axis=0)
    m = abs(sam_mean - 10) / 10

    assert np.all(m < 8e-3)

def test_MVN_density():
    L = np.zeros((3, 3))
    L[np.tril_indices(3)] = [0.9, 0.8, 0.5, 0.9, 0.2, 0.6]
    Sigma = L @ L.T
    mu = [0, 1, -1]

    xx = np.array([
        [0, 2, 1],
        [2, 3, 4],
        [0.5, 0.5, 0.5],
        [0, 1, -1]
    ])

    res = MVN_density(xx, mu, Sigma)
    true_val = np.array([8.742644e-04, 1.375497e-11, 3.739985e-03, 1.306453e-01])

    np.testing.assert_allclose(res, true_val, rtol=3e-3)

    xx = np.random.rand(3 * int(1e4)).reshape(-1, 3)
    res_chunks = MVN_density(xx, mu, Sigma)
    res_all = MVN_density(xx, mu, Sigma, max_size_x=int(1e4))

    np.testing.assert_allclose(res_chunks, res_all)

# To run the tests, use pytest from the command line
# pytest <this_script_name>.py
