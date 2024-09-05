import numpy as np
from scipy import stats
from BayesReconPy.utils import check_input_TD, check_weights, resample, MVN_density
from BayesReconPy.PMF import pmf_from_samples, pmf_from_params, pmf_sample

def reconc_MixCond(A, fc_bottom, fc_upper, bottom_in_type="pmf", distr=None,
                   num_samples=20000, return_type="pmf", suppress_warnings=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Check inputs
    check_input_TD(A, fc_bottom, fc_upper, bottom_in_type, distr, return_type)

    n_u = A.shape[0]
    n_b = A.shape[1]

    # Prepare samples from the base bottom distribution
    if bottom_in_type == "pmf":
        B = np.vstack([pmf_sample(fc, num_samples) for fc in fc_bottom])
        B = B.T
    elif bottom_in_type == "samples":
        B = np.vstack(fc_bottom)
        B = B.T
        num_samples = B.shape[0]
    elif bottom_in_type == "params":
        L_pmf = [pmf_from_params(fc, distr) for fc in fc_bottom]
        B = np.vstack([pmf_sample(pmf, num_samples) for pmf in L_pmf])
        B = B.T

    # Get mean and covariance matrix of the MVN upper base forecasts
    mu_u = fc_upper['mu']
    Sigma_u = np.array(fc_upper['Sigma'])

    # IS using MVN
    U = B @ A.T
    weights = MVN_density(U, mu_u, Sigma_u)

    check_weights_res = check_weights(weights)
    if check_weights_res['warning'] and not suppress_warnings:
        warning_msg = check_weights_res['warning_msg']
        print(f"Warning: {warning_msg}")

    if not (check_weights_res['warning'] and (1 in check_weights_res['warning_code'])):
        B = resample(B, weights, num_samples)

    ESS = np.sum(weights) ** 2 / np.sum(weights ** 2)

    B = B.T
    U = A @ B

    # Prepare output: include the marginal pmfs and/or the samples
    result = {
        'bottom_reconciled': {},
        'upper_reconciled': {},
        'ESS': ESS
    }

    if return_type in ['pmf', 'all']:
        upper_pmf = [pmf_from_samples(U[i, :]) for i in range(n_u)]
        bottom_pmf = [pmf_from_samples(B[i, :]) for i in range(n_b)]

        result['bottom_reconciled']['pmf'] = bottom_pmf
        result['upper_reconciled']['pmf'] = upper_pmf

    if return_type in ['samples', 'all']:
        result['bottom_reconciled']['samples'] = B
        result['upper_reconciled']['samples'] = U

    return result
