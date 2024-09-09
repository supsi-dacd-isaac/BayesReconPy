import numpy as np
from BayesReconPy.utils import DEFAULT_PARS, check_input_TD
from BayesReconPy.hierarchy import lowest_lev, get_Au
from BayesReconPy.reconc_gaussian import reconc_gaussian
from BayesReconPy.PMF import pmf_from_samples, pmf_from_params, pmf_check_support, pmf_bottom_up
from BayesReconPy.utils import MVN_sample
from typing import Union

def cond_biv_sampling(u, pmf1, pmf2):
    # Initialize switch flag
    sw = False
    if len(pmf1) > len(pmf2):
        pmf1, pmf2 = pmf2, pmf1
        sw = True

    b1 = np.full(len(u), np.nan)  # Initialize empty array

    for u_uniq in np.unique(u):  # Loop over unique values of u
        len_supp1 = len(pmf1)
        supp1 = np.arange(len_supp1)  # Support for pmf1
        p1 = pmf1

        supp2 = int(u_uniq) - supp1
        #supp2[supp2 < 0] = np.zero  # Trick to get NaN when accessing pmf2 outside the support
        # Handle out-of-bounds access
        p2 = np.array([pmf2[i] if i < len(pmf2) and i > 0 else 0 for i in supp2])
        # Normalize probabilities
        p = p1 * p2
        p /= p.sum()

        # Sample from supp1 based on the probability vector p
        u_posit = (u == u_uniq)
        b1[u_posit] = np.random.choice(supp1, size=u_posit.sum(), replace=True, p=p)

    if sw:
        b1 = u - b1  # If we switched pmf1 and pmf2, switch back

    return b1, u - b1


def TD_sampling(u, bott_pmf, toll=DEFAULT_PARS['TOLL'], Rtoll=DEFAULT_PARS['RTOLL'], smoothing=True,
                al_smooth=DEFAULT_PARS['ALPHA_SMOOTHING'], lap_smooth=DEFAULT_PARS['LAP_SMOOTHING']):
    if len(bott_pmf) == 1:
        return np.tile(u, (1, 1))

    l_l_pmf = pmf_bottom_up(bott_pmf, toll=toll, Rtoll=Rtoll, return_all=True,
                            smoothing=smoothing, alpha_smooth=al_smooth, laplace_smooth=lap_smooth)
    l_l_pmf = l_l_pmf[::-1] # flipping the list sa that we skip the upper pmf

    b_old = np.reshape(u, (1, -1))
    for l_pmf in l_l_pmf[1:]:
        L = len(l_pmf)
        b_new = np.empty((L, len(u)))
        for j in range(L // 2):
            b = cond_biv_sampling(b_old[j, :], l_pmf[2 * j], l_pmf[2 * j + 1])
            b_new[2 * j, :] = b[0]
            b_new[2 * j + 1, :] = b[1]
        if L % 2 == 1:
            b_new[L - 1, :] = b_old[L // 2, :]
        b_old = b_new

    return b_new


def reconc_TDcond(A, fc_bottom:Union[np.array, dict], fc_upper:dict, bottom_in_type="pmf", distr=None,
                  num_samples=20000, return_type="pmf", suppress_warnings=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Check inputs
    check_input_TD(A, fc_bottom, fc_upper, bottom_in_type, distr, return_type)

    # Find the "lowest upper"
    n_u = A.shape[0]
    n_b = A.shape[1]
    lowest_rows = lowest_lev(A)
    n_u_low = len(lowest_rows)  # number of lowest upper
    n_u_upp = n_u - n_u_low  # number of "upper upper"

    # Get mean and covariance matrix of the MVN upper base forecasts
    mu_u = fc_upper['mu']
    mu_u = np.array([mu_u[key] for key in mu_u]) if isinstance(mu_u, dict) else mu_u

    Sigma_u = np.array(fc_upper['Sigma'])

    # Get upper samples
    if n_u == n_u_low:
        U = MVN_sample(num_samples, mu_u, Sigma_u)  # (dim: num_samples x n_u_low)
        U = np.round(U)  # round to integer
        U_js = [U[:, i] for i in range(U.shape[1])]
    else:
        # Reconcile the upper
        A_u = get_Au(A, lowest_rows)
        mu_u_ord = np.concatenate([np.delete(mu_u,lowest_rows), mu_u[lowest_rows]])
        Sigma_u_ord = np.zeros((n_u, n_u))
        Sigma_u_ord[:n_u_upp, :n_u_upp] = np.delete(np.delete(Sigma_u, lowest_rows, axis=0), lowest_rows, axis=1)
        Sigma_u_ord[:n_u_upp, n_u_upp:] = np.delete(Sigma_u, lowest_rows, axis=0)[:, lowest_rows]
        Sigma_u_ord[n_u_upp:, :n_u_upp] = np.delete(Sigma_u, lowest_rows, axis=1)[lowest_rows, :]
        Sigma_u_ord[n_u_upp:, n_u_upp:] = Sigma_u[np.ix_(lowest_rows, lowest_rows)]

        rec_gauss_u = reconc_gaussian(A_u, mu_u_ord, Sigma_u_ord)
        U = MVN_sample(num_samples, rec_gauss_u['bottom_reconciled_mean'], rec_gauss_u['bottom_reconciled_covariance'])
        U = np.round(U)  # round to integer
        U_js = [U[:, i] for i in range(U.shape[1])]

    # Prepare list of bottom pmf
    if bottom_in_type == "pmf":
        L_pmf = [v for v in fc_bottom.values()]
    elif bottom_in_type == "samples":
        if isinstance(fc_bottom, dict):
            L_pmf = [pmf_from_samples(fc) for fc in fc_bottom.values()]
        else:
            L_pmf = [pmf_from_samples(fc) for fc in fc_bottom]
    elif bottom_in_type == "params":
        L_pmf = [pmf_from_params(fc, distr) for key, fc in fc_bottom.items()]

    # Prepare list of lists of bottom pmf relative to each lowest upper
    L_pmf_js = []
    for j in lowest_rows:
        Aj = A[j, :]
        L_pmf_js.append([L_pmf[i] for i in range(len(Aj)) if Aj[i]])

    # Check that each multiv. sample of U is contained in the supp of the bottom-up distr
    samp_ok = np.array([pmf_check_support(u_j, L_pmf_j) for u_j, L_pmf_j in zip(U_js, L_pmf_js)])
    samp_ok = np.sum(samp_ok, axis=0) == n_u_low

    U_js = [U_j[samp_ok] for U_j in U_js]
    num_samples_ok = np.sum(samp_ok)

    if num_samples_ok != num_samples and not suppress_warnings:
        print(f"Warning: Only {np.floor(np.sum(samp_ok) / num_samples * 1000) / 10}% of the upper samples "
              "are in the support of the bottom-up distribution; the others are discarded.")

    # Get bottom samples via the prob top-down
    B = np.empty((n_b, num_samples_ok))
    for j in range(n_u_low):

        mask_j = A[lowest_rows[j], :]  # mask for the position of the bottom referring to lowest upper j
        B[np.where(mask_j)[0], :] = TD_sampling(U_js[j], L_pmf_js[j])

    U = A @ B  # dim: n_upper x num_samples
    B = B.astype(int)
    U = U.astype(int)

    # Prepare output: include the marginal pmfs and/or the samples (depending on "return" inputs)
    result = {'bottom_reconciled': {}, 'upper_reconciled': {}}

    if return_type in ['pmf', 'all']:
        upper_pmf = [pmf_from_samples(U[i, :]) for i in range(n_u)]
        bottom_pmf = [pmf_from_samples(B[i, :]) for i in range(n_b)]
        result['bottom_reconciled']['pmf'] = bottom_pmf
        result['upper_reconciled']['pmf'] = upper_pmf

    if return_type in ['samples', 'all']:
        result['bottom_reconciled']['samples'] = B
        result['upper_reconciled']['samples'] = U

    return result
