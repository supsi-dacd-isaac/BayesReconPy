from xml.etree.ElementTree import fromstring

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, nbinom, multivariate_normal

################################################################################
# IMPLEMENTED DISTRIBUTIONS

DISTR_TYPES = ["continuous", "discrete"]
DISCR_DISTR = ["poisson", "nbinom"]
CONT_DISTR = ["gaussian"]

################################################################################
# PARAMETERS FOR PMF CONVOLUTION AND SMOOTHING

TOL = 1e-15
RTOL = 1e-9
ALPHA_SMOOTHING = 1e-9
ALPHA = 1e-9
LAP_SMOOTHING = False

DEFAULT_PARS = {
    "TOL": TOL,
    "RTOL": RTOL,
    "ALPHA_SMOOTHING": ALPHA_SMOOTHING,
    "ALPHA": ALPHA,
    "LAP_SMOOTHING": LAP_SMOOTHING,
    "DISTR_TYPES": DISTR_TYPES,
    "DISCR_DISTR": DISCR_DISTR,
    "CONT_DISTR": CONT_DISTR
}

################################################################################
# CHECK INPUT

def _check_S(S):
    # Check if S contains only 0s and 1s
    if not np.array_equal(np.sort(np.unique(S)), np.array([0, 1])):
        raise ValueError("Input error in S: S must be a matrix containing only 0s and 1s.")

    # Check if all columns have sums greater than 1
    if not np.all(np.sum(S, axis=0) > 1):
        raise ValueError("Input error in S: all bottom level forecasts must aggregate into an upper.")

    # Check for repeated rows in S
    if S.shape[0] != np.unique(S, axis=0).shape[0]:
        print("Warning: S has some repeated rows.")

    # Check that each bottom level has a corresponding row with one 1 and the rest 0s
    if np.unique(S[np.sum(S, axis=1) == 1, :], axis=0).shape[0] < S.shape[1]:
        raise ValueError(
            "Input error in S: there is at least one bottom that does not have a row with one 1 and the rest 0s.")

# Example usage:
# S = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
# check_S(S)
# S = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1]])
# check_S(S)

################################################################################
# Function to check aggregation matrix A
def _check_A(A):
    # Check if A contains only 0s and 1s
    if not np.all(np.isin(A, [0, 1])):
        raise ValueError("Input error in A: A must be a matrix containing only 0s and 1s.")

    # Check if all columns have at least one 1
    if np.any(np.sum(A, axis=0) == 0):
        raise ValueError("Input error in A: some columns do not have any 1. "
                         "All bottom level forecasts must aggregate into an upper.")

    # Check for repeated rows in A
    if A.shape[0] != np.unique(A, axis=0).shape[0]:
        print("Warning: A has some repeated rows.")


################################################################################
# Check if it is a covariance matrix (i.e., symmetric and positive definite)
def _check_cov(cov_matrix, Sigma_str, pd_check=False, symm_check=False):
    # Check if the matrix is square
    if not isinstance(cov_matrix, np.ndarray) or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(f"{Sigma_str} is not square")

    # Check if the matrix is positive semi-definite
    if pd_check:
        eigen_values = np.linalg.eigvalsh(cov_matrix)
        if np.any(eigen_values <= 0):
            raise ValueError(f"{Sigma_str} is not positive semi-definite")

    # Check if the matrix is symmetric
    if symm_check:
        if not np.allclose(cov_matrix, cov_matrix.T):
            raise ValueError(f"{Sigma_str} is not symmetric")

    # Check if the diagonal elements are non-negative
    if np.any(np.diag(cov_matrix) < 0):
        raise ValueError(f"{Sigma_str}: some elements on the diagonal are negative")

    # If all checks pass, return True
    return True


################################################################################
# Checks if the input is a real number
def _check_real_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


################################################################################
# Checks if the input is a positive number
def _check_positive_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0


################################################################################
# Check that the distribution is implemented

def _check_implemented_distr(distr):
    if distr not in DISCR_DISTR + CONT_DISTR:
        raise ValueError(
            f"Input error: the distribution must be one of {{{', '.join(DISCR_DISTR + CONT_DISTR)}}}"
        )


################################################################################
# Check the parameters of the distribution

def _check_distr_params(distr, params):
    _check_implemented_distr(distr)

    if not isinstance(params, dict):
        raise ValueError("Input error: the parameters of the distribution must be given as a dictionary.")

    if distr == "gaussian":
        mean = params.get("mean")
        sd = params.get("sd")
        if not _check_real_number(mean):
            raise ValueError("Input error: mean of Gaussian must be a real number")
        if not _check_positive_number(sd):
            raise ValueError("Input error: sd of Gaussian must be a positive number")

    elif distr == "poisson":
        lambda_ = params.get("lambda")
        if not _check_positive_number(lambda_):
            raise ValueError("Input error: lambda of Poisson must be a positive number")

    elif distr == "nbinom":
        size = params.get("size")
        prob = params.get("prob")
        mu = params.get("mu")

        if size is None:
            raise ValueError("Input error: size parameter for the nbinom distribution must be specified")
        if not _check_positive_number(size):
            raise ValueError("Input error: size of nbinom must be a positive number")

        if prob is not None and mu is not None:
            raise ValueError("Input error: prob and mu for the nbinom distribution are both specified")
        elif prob is None and mu is None:
            raise ValueError("Input error: either prob or mu must be specified")
        else:
            if prob is not None:
                if not _check_positive_number(prob) or prob > 1:
                    raise ValueError("Input error: prob of nbinom must be positive and <= 1")
            elif mu is not None:
                if not _check_positive_number(mu):
                    raise ValueError("Input error: mu of nbinom must be positive")


################################################################################
# Check that the samples are discrete
def _check_discrete_samples(samples):
    if not np.array_equal(samples, np.floor(samples)):
        raise ValueError("Input error: samples are not all discrete")


################################################################################
# Check input for BUIS (and for MH)
def _check_input_BUIS(A, base_forecasts, in_type, distr):
    _check_A(A)

    n_tot_A = A.shape[1] + A.shape[0]

    # Check in_type
    if not isinstance(in_type, list):
        raise ValueError("Input error: in_type must be a list")
    if not n_tot_A == len(in_type):
        raise ValueError("Input error: ncol(A) + nrow(A) != length(in_type)")
    for i in range(n_tot_A):
        if in_type[i] not in ["samples", "params"]:
            raise ValueError(f"Input error: in_type[{i}] must be either 'samples' or 'params'")

    # Check distr and base_forecasts
    if not isinstance(distr, list):
        raise ValueError("Input error: distr must be a list")
    if not n_tot_A == len(distr):
        raise ValueError("Input error: ncol(A) + nrow(A) != length(distr)")
    if not isinstance(base_forecasts, list):
        raise ValueError("Input error: base_forecasts must be a list")
    if not n_tot_A == len(base_forecasts):
        raise ValueError("Input error: ncol(A) + nrow(A) != length(base_forecasts)")

    for i in range(n_tot_A):
        if in_type[i] == "params":
            _check_distr_params(distr[i], base_forecasts[i])
        elif in_type[i] == "samples":
            if distr[i] not in DISTR_TYPES:
                raise ValueError(f"Input error: the distribution must be one of {{{', '.join(DISTR_TYPES)}}}")
            if distr[i] == "discrete":
                _check_discrete_samples(base_forecasts[i])


################################################################################
# Check input for TDcond
def _check_input_TD(A, fc_bottom, fc_upper, bottom_in_type, distr, return_type):
    _check_A(A)

    n_b = A.shape[1]  # number of bottom TS
    n_u = A.shape[0]  # number of upper TS

    if bottom_in_type in ["pmf", "params"]:
        if not isinstance(fc_bottom, dict):
            raise ValueError("Input error: fc_bottom must be a dictionary with the names of the bottom TS as keys")
        if not n_b == len(fc_bottom):
            raise ValueError("Input error: ncol(A) != length(fc_bottom)")


    if bottom_in_type not in ["pmf", "samples", "params"]:
        raise ValueError("Input error: bottom_in_type must be either 'pmf', 'samples', or 'params'")
    if return_type not in ["pmf", "samples", "all"]:
        raise ValueError("Input error: return_type must be either 'pmf', 'samples', or 'all'")
    if len(fc_bottom) != n_b:
        raise ValueError("Input error: length of fc_bottom does not match with A")

    # If Sigma is a number, transform it into a matrix
    if isinstance(fc_upper['Sigma'], (int, float)):
        fc_upper['Sigma'] = np.array([[fc_upper['Sigma']]])

    # Check the dimensions of mu and Sigma
    if len(fc_upper['mu']) != n_u or fc_upper['Sigma'].shape != (n_u, n_u):
        raise ValueError("Input error: the dimensions of the upper parameters do not match with A")

    # Check that Sigma is a covariance matrix (symmetric positive semi-definite)
    _check_cov(fc_upper['Sigma'], "Upper covariance matrix", symm_check=True)

    # If bottom_in_type is not "params" but distr is specified, throw a warning
    if bottom_in_type in ["pmf", "samples"] and distr is not None:
        print(f"Warning: Since bottom_in_type = '{bottom_in_type}', the input distr is ignored")

    # If bottom_in_type is params, distr must be one of the implemented discrete distr.
    # Also, check the parameters
    if bottom_in_type == "params":
        if distr is None:
            raise ValueError("Input error: if bottom_in_type = 'params', distr must be specified")
        if distr not in DISCR_DISTR:
            raise ValueError(f"Input error: distr must be one of {{{', '.join(DISCR_DISTR)}}}")
        for i in range(n_b):
            _check_distr_params(distr, fc_bottom[i])


################################################################################
# Check importance sampling weights
def _check_weights(w, n_eff_min=200, p_n_eff=0.01):
    warning = False
    warning_code = []
    warning_msg = []

    n = len(w)
    n_eff = n

    # 1. w == 0
    if np.all(w == 0):
        warning = True
        warning_code.append(1)
        warning_msg.append(
            "Importance Sampling: all the weights are zeros. This is probably caused by a strong incoherence between bottom and upper base forecasts."
        )
    else:
        # Effective sample size
        w = w / np.sum(w)
        n_eff = 1 / np.sum(w ** 2)

        # 2. n_eff < threshold
        if n_eff < n_eff_min:
            warning = True
            warning_code.append(2)
            warning_msg.append(
                f"Importance Sampling: effective_sample_size= {round(n_eff, 2)} (< {n_eff_min})."
            )

        # 3. n_eff < p*n, e.g., p = 0.05
        if n_eff < p_n_eff * n:
            warning = True
            warning_code.append(3)
            warning_msg.append(
                f"Importance Sampling: effective_sample_size= {round(n_eff, 2)} (< {round(p_n_eff * 100, 2)}%)."
            )

    return {
        "warning": warning,
        "warning_code": warning_code,
        "warning_msg": warning_msg,
        "n_eff": n_eff
    }


################################################################################
# SAMPLE

# Sample from one of the implemented distributions
def _distr_sample(params, distr, n):
    _check_distr_params(distr, params)

    if distr == "gaussian":
        mean = params['mean']
        sd = params['sd']
        samples = np.random.normal(loc=mean, scale=sd, size=n)

    elif distr == "poisson":
        lambda_ = params['lambda']
        samples = np.random.poisson(lam=lambda_, size=n)

    elif distr == "nbinom":
        size = params['size']
        prob = params.get('prob')
        mu = params.get('mu')

        if prob is not None:
            samples = np.random.negative_binomial(n=size, p=prob, size=n)
        elif mu is not None:
            samples = np.random.negative_binomial(n=size, p=size / (size + mu), size=n)

    else:
        raise ValueError(f"Unsupported distribution: {distr}")

    return samples


################################################################################
# Sample from a multivariate Gaussian distribution with specified mean and cov. matrix
def _MVN_sample(n_samples, mu, Sigma):
    return np.random.multivariate_normal(mu, Sigma, n_samples)


def _MVN_density(x, mu, Sigma, max_size_x=5000, suppress_warnings=True):
    mvr = multivariate_normal(mean=mu,cov=Sigma)
    pdf = mvr.pdf(x)
    return pdf


################################################################################
# Resample from weighted sample
def _resample(S_, weights, num_samples=None):

    if num_samples is None:
        num_samples = len(weights)

    if S_.shape[0] != len(weights):
        raise ValueError("Error in resample: nrow(S_) != length(weights)")
    #weights are log-likelihood, but we need a normalized vector for using np.random.choice
    weights = weights / weights.sum()
    tmp_idx = np.random.choice(np.arange(S_.shape[0]), size=num_samples, replace=True, p=weights)
    return S_[tmp_idx, :]


################################################################################
# Miscellaneous

# Compute the PMF of the distribution specified by distr and params at the points x
def _distr_pmf(x, params, distr):
    _check_distr_params(distr, params)

    if distr == "gaussian":
        mean = params['mean']
        sd = params['sd']
        pmf = norm.pdf(x, loc=mean, scale=sd)

    elif distr == "poisson":
        lambda_ = params['lambda']
        pmf = poisson.pmf(x, mu=lambda_)

    elif distr == "nbinom":
        size = params['size']
        prob = params.get('prob')
        mu = params.get('mu')

        if prob is not None:
            pmf = nbinom.pmf(x, n=size, p=prob)
        elif mu is not None:
            pmf = nbinom.pmf(x, n=size, p=size / (size + mu))

    else:
        raise ValueError(f"Unsupported distribution: {distr}")

    return pmf


################################################################################
# Print the shape of a matrix or array
def _shape(m):
    print(f"({m.shape[0]}, {m.shape[1]})")


################################################################################
# Functions for tests

# Generate Gaussian samples based on parameters from a CSV file
def _gen_gaussian(params_file, seed=None):
    if seed is not None:
        np.random.seed(seed)

    params = pd.read_csv(params_file, header=None)
    out = []

    for _, row in params.iterrows():
        mean, sd = row[0], row[1]
        samples = np.random.normal(loc=mean, scale=sd, size=int(1e6))
        out.append(samples)

    return out


# Generate Poisson samples based on parameters from a CSV file
def _gen_poisson(params_file, seed=None):
    if seed is not None:
        np.random.seed(seed)

    params = pd.read_csv(params_file, header=None)
    out = []

    for _, row in params.iterrows():
        lambda_ = row[0]
        samples = np.random.poisson(lam=lambda_, size=int(1e6))
        out.append(samples)

    return out

################################################################################
def _samples_from_pmf(pmf, n_samples):
    # compute the cdf of the bottom forecast
    cdf = np.cumsum(pmf)
    # use inverse sample trick to sample from the cdf
    samp = np.random.uniform(0, 1, n_samples)
    samp = np.array(np.searchsorted(cdf, samp))
    return samp