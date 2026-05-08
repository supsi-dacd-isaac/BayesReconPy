import numpy as np

def compute_es(y_true, y_samples):
    """
    Compute the multivariate Energy Score (ES).

    Parameters
    ----------
    y_true : array of shape (n_series, n_splits)
        True multivariate observations.
    y_samples : array of shape (n_series, n_splits, n_samples)
        Predictive samples for each split.

    Returns
    -------
    float
        Mean multivariate energy score across all splits.
    """
    n_series, n_splits, n_samples = y_samples.shape
    es_total = 0.0

    for t in range(n_splits):
        x = y_samples[:, t, :].T  # (n_samples, n_series)
        y = y_true[:, t]          # (n_series,)

        # First term: E||X - y||
        term1 = np.mean(np.linalg.norm(x - y, axis=1))

        # Second term: 0.5 * E||X - X'||
        term2 = 0.5 * np.mean(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2))

        es_total += term1 - term2

    return es_total / n_splits

def compute_crps(y_true, y_samples, q_min=None):
    """
    Compute (univariate) CRPS from Monte Carlo samples, aggregated like compute_es.

    Parameters
    ----------
    y_true : array of shape (n_series, n_splits)
        True observations (each series evaluated independently).
    y_samples : array of shape (n_series, n_splits, n_samples)
        Predictive samples.
    q_min : float in [0, 1], optional
        If given, discard the lower q_min fraction of samples *per (series, split)*
        after sorting (useful e.g. to ignore extreme lower tail).

    Returns
    -------
    float
        Mean CRPS across splits (and series), analogous to compute_es returning a scalar.

    Notes
    -----
    For samples X_1..X_m and observation y:
        CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
    """
    y_true = np.asarray(y_true)
    y_samples = np.asarray(y_samples)

    if y_samples.ndim != 3 or y_true.ndim != 2:
        raise ValueError("Expected y_true (n_series, n_splits) and y_samples (n_series, n_splits, n_samples).")
    if y_true.shape[:2] != y_samples.shape[:2]:
        raise ValueError("First two dims must match: y_true.shape == y_samples.shape[:2].")

    n_series, n_splits, n_samples = y_samples.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to compute the second term.")

    # Optionally drop lower tail after sorting along samples axis
    if q_min is not None:
        if not (0.0 <= q_min < 1.0):
            raise ValueError("q_min must be in [0, 1).")
        start_from = int(np.floor(n_samples * q_min))
        if start_from >= n_samples - 1:
            raise ValueError("q_min discards too many samples; need >= 2 samples remaining.")
        ys = np.sort(y_samples, axis=2)[..., start_from:]
    else:
        ys = y_samples

    # Aggregate like compute_es: average over splits, returning a single scalar.
    crps_total = 0.0
    m = ys.shape[2]

    for t in range(n_splits):
        x = ys[:, t, :]          # (n_series, m)
        y = y_true[:, t]         # (n_series,)

        # term1 = E|X - y|
        term1 = np.mean(np.abs(x - y[:, None]), axis=1)  # per-series

        # term2 = 0.5 * E|X - X'|
        # Efficient O(m log m) formula for mean absolute pairwise differences:
        # mean_{i,j} |x_i - x_j| = (2 / m^2) * sum_{k=1}^m (2k - m - 1) * x_(k)
        x_sorted = np.sort(x, axis=1)
        k = np.arange(1, m + 1, dtype=x_sorted.dtype)
        w = (2 * k - m - 1)[None, :]                       # (1, m)
        sum_abs_pairs = 2.0 * np.sum(w * x_sorted, axis=1)  # per-series, equals sum_{i,j}|x_i-x_j|
        mean_abs_pairs = sum_abs_pairs / (m * m)
        term2 = 0.5 * mean_abs_pairs

        crps_total += np.mean(term1 - term2)  # average across series for this split

    return crps_total / n_splits