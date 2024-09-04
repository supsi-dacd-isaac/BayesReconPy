import numpy as np
import pulp


def get_hier_rows(A, scale=196):
    A = np.array(A)
    k, m = A.shape

    # Matrix C of the coefficients of the non-linear problem
    C = np.dot(A, A.T)

    for i in range(k):
        for j in range(k):
            diff_ij = A[i, :] - A[j, :]
            C[i, j] = C[i, j] * np.sum(diff_ij * (diff_ij - 1)) * np.sum((A[j, :] - A[i, :]) * (A[j, :] - A[i, :] - 1))

    # Linearized Problem
    # Number of variables: k + k^2 + 1
    # Number of constraints: 1 + k^2 + k^2 + k^2 + m

    # Set coefficients of the objective function
    f_obj = [-1] * k + [0] * (k ** 2) + [1 - 1 / (2 * k)]

    # Set matrix corresponding to coefficients of constraints by rows
    coeff = [0] * k + list(C.flatten()) + [0]  # first constraint

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
    M4[:, :k] = A
    M4[:, k + k ** 2] = -1

    f_con = np.vstack([coeff, M1, M2, M3, M4])

    # Set inequality/equality signs
    f_dir = ['='] + ['<='] * (k ** 2) + ['<='] * (k ** 2) + ['>='] * (k ** 2) + ['<='] * m

    # Set right hand side coefficients
    f_rhs = [0] + [0] * (k ** 2) + [0] * (k ** 2) + [-1] * (k ** 2) + [0] * m

    # Define the problem
    prob = pulp.LpProblem("Minimize_Steps", pulp.LpMinimize)

    # Define variables
    vars = pulp.LpVariable.dicts("x", range(k + k ** 2 + 1), 0, 1, pulp.LpBinary)

    # Objective function
    prob += pulp.lpSum([f_obj[i] * vars[i] for i in range(k + k ** 2 + 1)])

    # Constraints
    for i in range(len(f_con)):
        prob += pulp.lpSum([f_con[i][j] * vars[j] for j in range(k + k ** 2 + 1)]) <= f_rhs[i]

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=scale))

    # Return the solution
    indices_sol = np.array([pulp.value(vars[i]) for i in range(k)])

    return indices_sol


def get_hier_rows(A, scale=196):
    # This function is equivalent to .get_hier_rows in python
    from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD

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
    M4[:, :k] = A
    M4[:, k + k ** 2] = -1

    f_con = np.vstack([coeff, M1, M2, M3, M4])

    f_dir = ['='] + ['<='] * (k ** 2) + ['<='] * (k ** 2) + ['>='] * (k ** 2) + ['<='] * m
    f_rhs = [0] + [0] * (k ** 2) + [0] * (k ** 2) + [-1] * (k ** 2) + [0] * m

    prob = LpProblem("Minimize_Steps", LpMinimize)

    vars = LpVariable.dicts("x", range(k + k ** 2 + 1), 0, 1, cat='Binary')

    prob += lpSum([f_obj[i] * vars[i] for i in range(k + k ** 2 + 1)])

    for i in range(len(f_con)):
        prob += lpSum([f_con[i][j] * vars[j] for j in range(k + k ** 2 + 1)]) <= f_rhs[i]

    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=scale))

    indices_sol = np.array([vars[i].varValue for i in range(k)])

    return indices_sol


def get_HG(A, v, d, it):
    indices_sol = get_hier_rows(A)

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


def gen_monthly():
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


def gen_weekly():
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


def temporal_aggregation(y, agg_levels=None):
    f = len(y) // (np.arange(1, len(y) + 1) % len(y)).max()
    L = len(y)
    s = np.arange(len(y)) / f + 1
    if agg_levels is None:
        agg_levels = [i for i in range(1, f + 1) if f % i == 0 and L >= i]
    else:
        agg_levels = sorted([level for level in agg_levels if level <= L])
        if 1 not in agg_levels:
            agg_levels = [1] + agg_levels

    out = []
    for k in agg_levels:
        num_aggs = L // k
        y_trunc = y[-num_aggs * k:]
        y_matrix = y_trunc.reshape((k, num_aggs))
        y_start = s[-num_aggs * k] + (L - num_aggs * k) / f
        y_f = f // k
        y_agg = np.sum(y_matrix, axis=0)
        out.append((y_agg, y_f, y_start))

    return out[::-1]


def get_reconc_matrices(agg_levels, h):
    A = []
    for k in agg_levels:
        if k == 1:
            continue
        k_r = h // k
        k_A = np.zeros((k_r, h), dtype=int)
        for r in range(k_r):
            k_A[r, r * k:(r + 1) * k] = 1
        A.append(k_A)

    A = np.vstack(A[::-1])
    S = np.vstack([A, np.eye(h, dtype=int)])
    return {'A': A, 'S': S}


def get_A_from_S(S):
    S = np.array(S)
    bottom_idxs = np.where(np.sum(S, axis=1) == 1)[0]
    if len(bottom_idxs) < S.shape[1]:
        raise ValueError("Check S: some bottom rows are missing")
    upper_idxs = np.setdiff1d(np.arange(S.shape[0]), bottom_idxs)
    A = S[upper_idxs, :]
    return {'A': A, 'upper_idxs': upper_idxs, 'bottom_idxs': bottom_idxs}


def split_hierarchy(S, Y):
    if len(S) != len(Y):
        raise ValueError(f"Error: summing matrix rows ({len(S)}) != length base forecasts ({len(Y)})")

    result = get_A_from_S(S)
    upper = np.array(Y)[result['upper_idxs']]
    bottom = np.array(Y)[result['bottom_idxs']]
    return {
        'A': result['A'],
        'upper': upper,
        'bottom': bottom,
        'upper_idxs': result['upper_idxs'],
        'bottom_idxs': result['bottom_idxs']
    }


def check_hierarchical(A):
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


def check_BU_matr(A):
    A = np.array(A)
    k, m = A.shape

    for i in range(k):
        for j in range(k):
            if i < j:
                cond1 = np.dot(A[i, :], A[j, :]) != 0
                cond2 = np.any(A[i, :] > A[j, :])
                if cond1 and cond2:
                    return False
    return True


def lowest_lev(A):
    A = np.array(A)
    if not check_hierarchical(A):
        raise ValueError("Matrix A is not hierarchical")

    A_uni = np.unique(A, axis=0)
    k, m = A_uni.shape

    low_rows_A_uni = []
    for i in range(k):
        low_rows_A_uni.append(i)
        for j in range(k):
            if i != j:
                if np.all(A_uni[j, :] <= A_uni[i, :]):
                    low_rows_A_uni.pop()
                    break

    low_rows_A = np.array([row for idx, row in enumerate(A) if idx in low_rows_A_uni])

    if np.any(np.sum(A[low_rows_A, :], axis=0) != 1):
        unbal_bott = np.where(np.sum(A[low_rows_A, :], axis=0) != 1)[0]
        raise ValueError(
            f"It is impossible to find the lowest upper level. Probably the hierarchy is unbalanced, the following bottom should be duplicated: {', '.join(map(str, unbal_bott))}")

    return low_rows_A


def get_Au(A, lowest_rows=None):
    if lowest_rows is None:
        lowest_rows = lowest_lev(A)

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


def check_ordered_A(A):
    A = np.array(A)
    aggregates_sum = np.sum(A, axis=1)
    ordered_aggreg = np.argsort(aggregates_sum)[::-1]  # Sort indices in decreasing order

    if np.all(aggregates_sum == aggregates_sum[ordered_aggreg]):
        return {'value': True}
    else:
        return {'value': False, 'order': ordered_aggreg}


