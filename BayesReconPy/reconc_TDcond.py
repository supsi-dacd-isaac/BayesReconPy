import numpy as np
from BayesReconPy.utils import DEFAULT_PARS, check_input_TD
from BayesReconPy.hierarchy import lowest_lev, get_Au
from BayesReconPy.reconc_gaussian import reconc_gaussian
from BayesReconPy.PMF import pmf_from_samples, pmf_from_params, pmf_check_support, pmf_bottom_up
from BayesReconPy.utils import MVN_sample


def cond_biv_sampling(u, pmf1, pmf2):
    # Swap pmf1 and pmf2 if necessary
    if len(pmf1) > len(pmf2):
        pmf1, pmf2 = pmf2, pmf1
        swapped = True
    else:
        swapped = False

    b1 = np.full(len(u), np.nan)  # initialize empty vector

    for u_uniq in np.unique(u):  # loop over different values of u
        len_supp1 = len(pmf1)
        supp1 = np.arange(len_supp1)
        p1 = np.array(pmf1)

        supp2 = u_uniq - supp1
        supp2[supp2 < 0] = np.inf  # trick to get NA when accessing pmf2 outside the support
        p2 = np.array([pmf2[int(s)] if s >= 0 else 0 for s in supp2])

        p = p1 * p2
        p /= np.sum(p)

        u_posit = (u == u_uniq)
        b1[u_posit] = np.random.choice(supp1, size=np.sum(u_posit), replace=True, p=p)

    if swapped:
        b1 = u - b1  # if we have swapped, switch back

    return [b1.tolist(), (u - b1).tolist()]


def TD_sampling(u, bott_pmf, toll=DEFAULT_PARS['TOLL'], Rtoll=DEFAULT_PARS['RTOLL'], smoothing=True,
                al_smooth=DEFAULT_PARS['ALPHA_SMOOTHING'], lap_smooth=DEFAULT_PARS['LAP_SMOOTHING']):
    if len(bott_pmf) == 1:
        return np.tile(u, (1, 1))

    l_l_pmf = pmf_bottom_up(bott_pmf, toll=toll, Rtoll=Rtoll, return_all=True,
                            smoothing=smoothing, al_smooth=al_smooth, lap_smooth=lap_smooth)

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


def reconc_TDcond(A, fc_bottom, fc_upper, bottom_in_type="pmf", distr=None,
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
    Sigma_u = np.array(fc_upper['Sigma'])

    # Get upper samples
    if n_u == n_u_low:
        U = MVN_sample(num_samples, mu_u, Sigma_u)  # (dim: num_samples x n_u_low)
        U = np.round(U)  # round to integer
        U_js = [U[:, i] for i in range(U.shape[1])]
    else:
        # Reconcile the upper
        A_u = get_Au(A, lowest_rows)
        mu_u_ord = np.concatenate([mu_u[np.arange(n_u_upp)], mu_u[lowest_rows]])
        Sigma_u_ord = np.zeros((n_u, n_u))
        Sigma_u_ord[:n_u_upp, :n_u_upp] = Sigma_u[np.ix_(np.arange(n_u_upp), np.arange(n_u_upp))]
        Sigma_u_ord[:n_u_upp, n_u_upp:] = Sigma_u[np.ix_(np.arange(n_u_upp), lowest_rows)]
        Sigma_u_ord[n_u_upp:, :n_u_upp] = Sigma_u[np.ix_(lowest_rows, np.arange(n_u_upp))]
        Sigma_u_ord[n_u_upp:, n_u_upp:] = Sigma_u[np.ix_(lowest_rows, lowest_rows)]

        rec_gauss_u = reconc_gaussian(A_u, mu_u_ord, Sigma_u_ord)
        U = MVN_sample(num_samples, rec_gauss_u['bottom_reconciled_mean'], rec_gauss_u['bottom_reconciled_covariance'])
        U = np.round(U)  # round to integer
        U_js = [U[:, i] for i in range(U.shape[1])]

    # Prepare list of bottom pmf
    if bottom_in_type == "pmf":
        L_pmf = fc_bottom
    elif bottom_in_type == "samples":
        L_pmf = [pmf_from_samples(fc) for fc in fc_bottom]
    elif bottom_in_type == "params":
        L_pmf = [pmf_from_params(fc, distr) for fc in fc_bottom]

    # Prepare list of lists of bottom pmf relative to each lowest upper
    L_pmf_js = [L_pmf[Aj.astype(bool)] for Aj in A[lowest_rows, :]]

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
        B[mask_j, :] = TD_sampling(U_js[j], L_pmf_js[j])

    U = A @ B  # dim: n_upper x num_samples

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
