import numpy as np
from scipy import stats
from BayesReconPy.utils import check_input_BUIS, distr_sample, distr_pmf


def reconc_MCMC(A, base_forecasts, distr, num_samples=10000, tuning_int=100, init_scale=1, burn_in=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Ensure that data inputs are valid
    if distr == "gaussian":
        raise NotImplementedError("MCMC for Gaussian distributions is not implemented")

    n_bottom = A.shape[1]
    n_ts = A.shape[0] + A.shape[1]

    # Transform distr into list
    if not isinstance(distr, list):
        distr = [distr] * n_ts

    # Check input
    check_input_BUIS(A, base_forecasts, in_type=['params'] * n_ts, distr=distr)

    # the first burn_in samples will be removed
    num_samples += burn_in

    # Set the covariance matrix of the proposal (identity matrix)
    cov_mat_prop = np.eye(n_bottom)

    # Set the counter for tuning
    c_tuning = tuning_int

    # Initialize acceptance counter
    accept_count = 0

    # Create empty matrix for the samples from MCMC
    b = np.zeros((num_samples, n_bottom))

    # Get matrix A and bottom base forecasts
    split_hierarchy_res = {
        'A': A,
        'upper': base_forecasts[:A.shape[0]],
        'bottom': base_forecasts[A.shape[0]:],
        'upper_idxs': list(range(A.shape[0])),
        'bottom_idxs': list(range(A.shape[0], n_ts))
    }
    bottom_base_forecasts = split_hierarchy_res['bottom']
    bottom_distr = [distr[i] for i in split_hierarchy_res['bottom_idxs']]

    # Initialize first sample (draw from base distribution)
    b[0, :] = initialize_b(bottom_base_forecasts, bottom_distr)

    # Initialize prop list
    old_prop = {
        'b': b[0, :],
        'scale': init_scale
    }

    # Run the chain
    for i in range(1, num_samples):
        if c_tuning == 0:
            old_prop['acc_rate'] = accept_count / tuning_int  # set acc_rate
            accept_count = 0  # reset acceptance counter
            c_tuning = tuning_int  # reset tuning counter

        prop = proposal(old_prop, cov_mat_prop)
        b_prop = prop['b']
        alpha = accept_prob(b_prop, b[i - 1, :], A, distr, base_forecasts)

        if np.random.uniform() < alpha:
            b[i, :] = b_prop
            accept_count += 1
        else:
            b[i, :] = b[i - 1, :]

        old_prop = {
            'b': b[i, :],
            'scale': prop['scale']
        }

        c_tuning -= 1

    b_samples = b[burn_in:, :].T  # output shape: n_bottom x num_samples
    u_samples = A @ b_samples
    y_samples = np.vstack([u_samples, b_samples])

    return {
        'bottom_reconciled_samples': b_samples,
        'upper_reconciled_samples': u_samples,
        'reconciled_samples': y_samples
    }


def initialize_b(bottom_base_forecasts, bottom_distr):
    b = []
    for i in range(len(bottom_distr)):
        b.append(distr_sample(bottom_base_forecasts[i], bottom_distr[i], 1))
    return np.array(b)


def accept_prob(b, b0, A, distr, params):
    alpha = target_pmf(b, A, distr, params) / target_pmf(b0, A, distr, params)
    return min(1, alpha)


def target_pmf(b, A, distr, params):
    n_ts = A.shape[0] + A.shape[1]
    y = np.vstack([A, np.eye(A.shape[1])]) @ b
    pmf = 1
    for j in range(n_ts):
        pmf *= distr_pmf(y[j], params[j], distr[j])
    return pmf


def proposal(prev_prop, cov_mat_prop):
    b0 = prev_prop['b']
    old_scale = prev_prop['scale']
    acc_rate = prev_prop.get('acc_rate', None)

    if acc_rate is not None:
        scale = tune(old_scale, acc_rate)
    else:
        scale = old_scale

    n_x = len(b0)

    if n_x != cov_mat_prop.shape[0]:
        raise ValueError(
            f"Error in proposal: previous state dim ({n_x}) and covariance dim ({cov_mat_prop.shape[0]}) do not match")

    dd = np.random.normal(size=n_x) * np.diag(cov_mat_prop) * scale
    dd = np.round(dd, 0)
    b = b0 + dd

    return {'b': b, 'acc_rate': acc_rate, 'scale': scale}


def tune(scale, acc_rate):
    if acc_rate < 0.001:
        return scale * 0.1
    elif acc_rate < 0.05:
        return scale * 0.5
    elif acc_rate < 0.2:
        return scale * 0.9
    elif acc_rate > 0.5:
        return scale * 1.1
    elif acc_rate > 0.75:
        return scale * 2
    elif acc_rate > 0.95:
        return scale * 10
    return scale
