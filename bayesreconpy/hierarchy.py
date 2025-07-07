import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD
import pandas as pd


def _get_hier_rows(A, scale=196):
    # This function is equivalent to .get_hier_rows in python

    A = np.array(A)
    k, m = A.shape

    # Matrix C of the coefficients of the non-linear problem
    C = np.dot(A, A.T)

    for i in range(k):
        for j in range(k):
            diff_ij = A[i, :] - A[j, :]
            C[i, j] = C[i, j] * np.sum(diff_ij * (diff_ij - 1)) * np.sum((A[j, :] - A[i, :]) * (A[j, :] - A[i, :] - 1))

    # Linearized Problem
    f_obj = [-1] * k + [0] * (k ** 2) + [1 - 1 / (2 * k)]

    coeff = [0] * k + list(C.flatten()) + [0]

    M1 = np.zeros((k ** 2, k + k ** 2 + 1))  # z_{ij} <= x_i
    for i in range(k):
        temp = np.zeros((k, k))
        temp[i, :] = -1
        M1[:, i] = temp.flatten()
    M1[:, k:(k + k ** 2)] = np.eye(k ** 2)

    M2 = np.zeros((k ** 2, k + k ** 2 + 1))  # z_{ij} <= x_j
    for i in range(k):
        temp = np.zeros((k, k))
        temp[:, i] = -1
        M2[:, i] = temp.flatten()
    M2[:, k:(k + k ** 2)] = np.eye(k ** 2)

    M3 = np.zeros((k ** 2, k + k ** 2 + 1))  # z_{ij} >= x_i + x_j - 1
    M3[:, :k] = M1[:, :k] + M2[:, :k]
    M3[:, k:(k + k ** 2)] = np.eye(k ** 2)

    M4 = np.zeros((m, k + k ** 2 + 1))  # sum_i x_i A_{ij} <= y
    M4[:, :k] = A.T
    M4[:, k + k ** 2] = -1

    f_con = np.vstack([coeff, M1, M2, M3, M4])

    f_dir = ['='] + ['<='] * (k ** 2) + ['<='] * (k ** 2) + ['>='] * (k ** 2) + ['<='] * m
    f_rhs = [0] + [0] * (k ** 2) + [0] * (k ** 2) + [-1] * (k ** 2) + [0] * m

    prob = LpProblem("Minimize_Steps", LpMinimize)

    vars = LpVariable.dicts("x", range(k + k ** 2 + 1), 0, 1, cat='Binary')

    prob += lpSum([f_obj[i] * vars[i] for i in range(k + k ** 2 + 1)])

    #for i in range(len(f_con)):
    #    prob += lpSum([f_con[i][j] * vars[j] for j in range(k + k ** 2 + 1)]) <= f_rhs[i]

    # first constraint
    prob += lpSum([f_con[0][j] * vars[j] for j in range(k + k ** 2 + 1)]) == f_rhs[0]
    # second set of constraints
    for i in np.arange(1, k ** 2 + 1):
        prob += lpSum([f_con[i][j] * vars[j] for j in range(k + k ** 2 + 1)]) <= f_rhs[i]
    # third set of constraints
    for i in np.arange(k ** 2 + 1, 2 * k ** 2 + 1):
        prob += lpSum([f_con[i][j] * vars[j] for j in range(k + k ** 2 + 1)]) <= f_rhs[i]
    # fourth set of constraints
    for i in np.arange(2 * k ** 2 + 1, 3 * k ** 2 + 1):
        prob += lpSum([f_con[i][j] * vars[j] for j in range(k + k ** 2 + 1)]) >= f_rhs[i]
    # fifth set of constraints
    for i in np.arange(3 * k ** 2 + 1, 3 * k ** 2 + 1 + m):
        prob += lpSum([f_con[i][j] * vars[j] for j in range(k + k ** 2 + 1)]) <= f_rhs[i]

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=scale))

    indices_sol = np.array([vars[i].varValue for i in range(k)])

    return indices_sol


def _get_HG(A, v, d, it):
    indices_sol = _get_hier_rows(A)

    ind_h = indices_sol.astype(bool)
    H = A[ind_h]
    v_h = np.array(v)[ind_h]
    d_h = np.array(d)[ind_h]
    it_h = np.array(it)[ind_h]

    ord = np.argsort(np.sum(H, axis=1))
    H = H[ord]
    v_h = v_h[ord]
    d_h = d_h[ord]
    it_h = it_h[ord]

    ind_g = ~ind_h
    if np.sum(ind_g) == 0:
        G = None
        v_g = None
        d_g = None
        it_g = None
    else:
        G = A[ind_g]
        v_g = np.array(v)[ind_g]
        d_g = np.array(d)[ind_g]
        it_g = np.array(it)[ind_g]

    return {
        'H': H,
        'G': G,
        'Hv': v_h,
        'Gv': v_g,
        'Hdistr': d_h,
        'Gdistr': d_g,
        'Hin_type': it_h,
        'Gin_type': it_g
    }


def _gen_monthly():
    H = np.zeros((10, 12), dtype=int)
    for j in range(6):
        H[j, (2 * j):(2 * (j + 1))] = 1
    for j in range(3):
        H[6 + j, (4 * j):(4 * (j + 1))] = 1
    H[9, :] = 1

    G = np.zeros((6, 12), dtype=int)
    for j in range(4):
        G[j, (3 * j):(3 * (j + 1))] = 1
    for j in range(2):
        G[4 + j, (6 * j):(6 * (j + 1))] = 1

    return np.vstack([H, G])


def _gen_weekly():
    H = np.zeros((40, 52), dtype=int)
    for j in range(26):
        H[j, (2 * j):(2 * (j + 1))] = 1
    for j in range(13):
        H[26 + j, (4 * j):(4 * (j + 1))] = 1
    H[39, :] = 1

    G = np.zeros((6, 52), dtype=int)
    for j in range(4):
        G[j, (13 * j):(13 * (j + 1))] = 1
    for j in range(2):
        G[4 + j, (26 * j):(26 * (j + 1))] = 1

    return np.vstack([H, G])


def _check_hierarchical(A):
    A = np.array(A)
    k, m = A.shape

    for i in range(k):
        for j in range(k):
            if i < j:
                cond1 = np.dot(A[i, :], A[j, :]) != 0
                cond2 = np.any(A[j, :] > A[i, :])
                cond3 = np.any(A[i, :] > A[j, :])
                if cond1 and cond2 and cond3:
                    return False
    return True



def _lowest_lev(A):
    A = np.array(A)
    if not _check_hierarchical(A):
        raise ValueError("Matrix A is not hierarchical")

    indexes = np.unique(A, return_index=True, axis=0)[1];
    A_uni = np.vstack([A[index] for index in sorted(indexes)])
    k, m = A_uni.shape

    low_rows_A_uni = []
    for i in range(k):
        low_rows_A_uni.append(i)
        for j in range(k):
            if i != j:
                if np.all(A_uni[j, :] <= A_uni[i, :]):
                    low_rows_A_uni.pop()
                    break


    # Keep all rows except those that have no descendants among the uppers
    low_rows_A_uni = list(dict.fromkeys(low_rows_A_uni))  # Remove duplicates and keep order

    # Now, change the indices of the lowest rows to match with A (instead of A_uni)
    #low_rows_A = np.arange(A.shape[0])[np.unique(A, axis=0, return_index=True)[1]][low_rows_A_uni]

    _, unique_idxs = np.unique(A, axis=0, return_index=True)
    low_rows_A = np.sort(unique_idxs)[low_rows_A_uni]

    # The sum of the rows corresponding to the lowest level should be a vector of 1
    if not np.all(np.sum(A[low_rows_A, :], axis=0) == 1):
        unbal_bott = np.where(np.sum(A[low_rows_A, :], axis=0) != 1)[0]
        err_mess = "It is impossible to find the lowest upper level. Probably the hierarchy is unbalanced, the following bottom should be duplicated (see example): "
        err_mess += " ".join(map(str, unbal_bott))
        raise ValueError(err_mess)
    return low_rows_A


def _get_Au(A, lowest_rows=None):
    if lowest_rows is None:
        lowest_rows = _lowest_lev(A)

    if len(lowest_rows) == len(A):
        print("Warning: All the upper are lowest-upper. Return None")
        return None

    A = np.array(A)
    lowest_rows = np.array(lowest_rows)

    A_ = np.delete(A, lowest_rows, axis=0)
    n_upp_u = A_.shape[0]
    n_bott_u = len(lowest_rows)
    A_u = np.zeros((n_upp_u, n_bott_u), dtype=int)

    for j, l in enumerate(lowest_rows):
        for i in range(n_upp_u):
            A_u[i, j] = np.all(A[l, :] <= A_[i, :])  # check if "lower upper" j is a descendant of "upper upper" i

    return A_u


def _temporal_aggregation(y, agg_levels=None):
    """
    Temporally aggregates a time series at specified levels.

    Parameters:
    - y: DataFrame, the time series data with columns as months and index as years.
    - agg_levels: list of int, aggregation levels in terms of months per period.

    Returns:
    - dict with aggregated series at each level.
    """
    print("Checking if pandas is recognized:", pd)
    # Determine the frequency (e.g., 12 for monthly data)
    f = 12  # Assuming monthly frequency for the input data (12 months per year)
    L = y.shape[0] * f  # Total length in terms of months

    # Default aggregation levels if none are specified
    if agg_levels is None:
        agg_levels = [i for i in range(1, f + 1) if f % i == 0 and L >= i]

    # Sort and ensure aggregation includes 1 (monthly level) if not already present
    agg_levels = sorted([k for k in agg_levels if k <= L])
    if 1 not in agg_levels:
        agg_levels = [1] + agg_levels

    # Store aggregated results
    aggregated_data = {}

    # Convert the DataFrame to a 1D array to handle monthly data as a single series
    y_flat = y.to_numpy().flatten()

    for k in agg_levels:
        num_aggs = L // k
        y_trunc = y_flat[-num_aggs * k:]  # Truncate the series for even aggregation
        y_matrix = y_trunc.reshape(-1, k)  # Reshape based on the aggregation level
        y_agg = y_matrix.sum(axis=1)  # Sum across rows to aggregate

        # Define frequency and start time
        y_f = f // k
        start_year = y.index[0] + (L - num_aggs * k) // f
        agg_index = pd.date_range(start=f'{start_year}-01-01', periods=num_aggs, freq=f'{12 // y_f}M')

        # Store the result in DataFrame form
        aggregated_data[f"{k}-Monthly"] = pd.Series(y_agg, index=agg_index)

    # Reverse the dictionary to match the order of output in R
    return dict(reversed(list(aggregated_data.items())))


def _get_reconc_matrices(agg_levels, h):
    """
    Generates aggregation and reconciliation matrices for hierarchical forecasting.

    Parameters:
    - agg_levels: list of int, specifying the aggregation levels.
    - h: int, forecasting horizon (number of future time steps).

    Returns:
    - A dictionary with:
        - 'A': Aggregation matrix.
        - 'S': Reconciliation matrix (combining 'A' and identity matrix).
    """
    A_matrices = []

    for k in agg_levels:
        if k == 1:
            continue  # Skip if the aggregation level is 1 (no aggregation)

        k_r = int(h / k)  # Number of rows for this aggregation level
        k_A = np.zeros((k_r, h))  # Initialize the aggregation matrix for level k

        col_idx = 0
        for r in range(k_r):
            k_A[r, col_idx:col_idx + k] = 1  # Fill in the row with ones for aggregation
            col_idx += k  # Move to the next block

        A_matrices.append(k_A)

    # Combine all individual A matrices into a single matrix by stacking them vertically
    A = np.vstack(A_matrices[::-1])  # Reverse the order of matrices before stacking

    # Create the reconciliation matrix S by combining A with an identity matrix of size h
    S = np.vstack([A, np.eye(h)])

    return {'A': A, 'S': S}
