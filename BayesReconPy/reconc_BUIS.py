import numpy as np
from scipy import stats
from scipy.stats import norm
from BayesReconPy.utils import DEFAULT_PARS, check_input_BUIS, check_weights, resample, distr_sample, distr_pmf
from BayesReconPy.hierarchy import check_hierarchical, get_HG


def check_hierfamily_rel(sh_res, distr, debug=False):
    bottom_idxs = sh_res['bottom_idxs']
    upper_idxs = sh_res['upper_idxs']
    A = sh_res['A']

    for bi in range(len(bottom_idxs)):
        distr_bottom = distr[bottom_idxs[bi]]
        rel_upper_i = A[:, bi]
        rel_distr_upper = [distr[i] for i in range(A.shape[0]) if rel_upper_i[i] == 1]

        err_message = "A continuous bottom distribution cannot be child of a discrete one."

        if distr_bottom == "continuous":
            if "discrete" in rel_distr_upper or any(d in DEFAULT_PARS['DISCR_DISTR'] for d in rel_distr_upper):
                if debug:
                    return -1
                else:
                    raise ValueError(err_message)

        if distr_bottom in DEFAULT_PARS['CONT_DISTR']:
            if "discrete" in rel_distr_upper or any(d in DEFAULT_PARS['DISCR_DISTR'] for d in rel_distr_upper):
                if debug:
                    return -1
                else:
                    raise ValueError(err_message)

    if debug:
        return 0



def emp_pmf(l, density_samples):
    # Compute the empirical PMF from the density samples
    values, counts = np.unique(density_samples, return_counts=True)
    empirical_pmf = dict(zip(values, counts / len(density_samples)))

    # Extract PMF values for the indices specified in `l`
    w = np.array([empirical_pmf.get(i, 0) for i in l])
    return w


def compute_weights(b, u, in_type_, distr_):
    if in_type_ == "samples":
        if distr_ == "discrete":
            # Discrete samples
            w = emp_pmf(b, u)
        elif distr_ == "continuous":
            # KDE (Kernel Density Estimation)
            kde = stats.gaussian_kde(u, bw_method='scott')
            w = kde.evaluate(b)
        # Ensure no NA values are returned, replace with 0
        w[np.isnan(w)] = 0
    elif in_type_ == "params":
        # This function needs to be defined separately
        w = distr_pmf(b, u, distr_)

    # Ensure not to return all 0 weights, return ones instead
    if np.sum(w) == 0:
        w = np.ones_like(w)

    return w



def reconc_BUIS(A, base_forecasts, in_type, distr, num_samples=20000, suppress_warnings=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_upper = A.shape[0]
    n_bottom = A.shape[1]
    n_tot = len(base_forecasts)

    # Transform distr and in_type into lists
    if not isinstance(distr, list):
        distr = [distr] * n_tot
    if not isinstance(in_type, list):
        in_type = [in_type] * n_tot

    # Ensure that data inputs are valid
    check_input_BUIS(A, base_forecasts, in_type, distr)

    # Split bottoms and uppers
    upper_base_forecasts = base_forecasts[:n_upper]
    bottom_base_forecasts = base_forecasts[n_upper:]

    split_hierarchy_res = {
        'A': A,
        'upper': upper_base_forecasts,
        'bottom': bottom_base_forecasts,
        'upper_idxs': list(range(n_upper)),
        'bottom_idxs': list(range(n_upper, n_tot))
    }

    # Check on continuous/discrete in relationship to the hierarchy
    check_hierfamily_rel(split_hierarchy_res, distr)

    # H, G
    is_hier = check_hierarchical(A)
    if is_hier:
        H = A
        G = None
        upper_base_forecasts_H = upper_base_forecasts
        upper_base_forecasts_G = None
        in_typeH = [in_type[i] for i in split_hierarchy_res['upper_idxs']]
        distr_H = [distr[i] for i in split_hierarchy_res['upper_idxs']]
        in_typeG = None
        distr_G = None
    else:
        get_HG_res = get_HG(A, upper_base_forecasts, [distr[i] for i in split_hierarchy_res['upper_idxs']],
                            [in_type[i] for i in split_hierarchy_res['upper_idxs']])
        H = get_HG_res['H']
        upper_base_forecasts_H = get_HG_res['Hv']
        G = get_HG_res['G']
        upper_base_forecasts_G = get_HG_res['Gv']
        in_typeH = get_HG_res['Hin_type']
        distr_H = get_HG_res['Hdistr']
        in_typeG = get_HG_res['Gin_type']
        distr_G = get_HG_res['Gdistr']

    # Reconciliation using BUIS
    # 1. Bottom samples
    B = []
    in_type_bottom = [in_type[i] for i in split_hierarchy_res['bottom_idxs']]
    for bi in range(n_bottom):
        if in_type_bottom[bi] == "samples":
            B.append(np.array(bottom_base_forecasts[bi]))
        elif in_type_bottom[bi] == "params":
            B.append(
                distr_sample(bottom_base_forecasts[bi], distr[split_hierarchy_res['bottom_idxs']][bi], num_samples))
    B = np.column_stack(B)  # B is a matrix (num_samples x n_bottom)

    # Bottom-Up IS on the hierarchical part
    for hi in range(H.shape[0]):
        c = H[hi, :]
        b_mask = (c != 0)
        weights = compute_weights(
            b=np.dot(B, c),
            u=upper_base_forecasts_H[hi],
            in_type_=in_typeH[hi],
            distr_=distr_H[hi]
        )
        check_weights_res = check_weights(weights)
        if check_weights_res['warning'] and not suppress_warnings:
            warning_msg = check_weights_res['warning_msg']
            upper_fromA_i = [i for i, row in enumerate(A) if np.allclose(row, c)]
            for wmsg in warning_msg:
                wmsg = f"{wmsg} Check the upper forecast at index: {upper_fromA_i}."
                print(f"Warning: {wmsg}")
        if check_weights_res['warning'] and 1 in check_weights_res['warning_code']:
            continue
        B[:, b_mask] = resample(B[:, b_mask], weights)

    if G is not None:
        # Plain IS on the additional constraints
        weights = np.ones(B.shape[0])
        for gi in range(G.shape[0]):
            c = G[gi, :]
            weights *= compute_weights(
                b=np.dot(B, c),
                u=upper_base_forecasts_G[gi],
                in_type_=in_typeG[gi],
                distr_=distr_G[gi]
            )
        check_weights_res = check_weights(weights)
        if check_weights_res['warning'] and not suppress_warnings:
            warning_msg = check_weights_res['warning_msg']
            upper_fromA_i = [i for i in range(n_upper) for gi in range(G.shape[0]) if np.allclose(A[i, :], G[gi, :])]
            for wmsg in warning_msg:
                wmsg = f"{wmsg} Check the upper forecasts at index: {{{', '.join(map(str, upper_fromA_i))}}}."
                print(f"Warning: {wmsg}")
        if not (check_weights_res['warning'] and 1 in check_weights_res['warning_code']):
            B = resample(B, weights)

    B = B.T
    U = np.dot(A, B)
    Y_reconc = np.vstack([U, B])

    return {
        'bottom_reconciled_samples': B,
        'upper_reconciled_samples': U,
        'reconciled_samples': Y_reconc
    }

