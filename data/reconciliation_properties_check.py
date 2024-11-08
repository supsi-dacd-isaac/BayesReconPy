import numpy as np
import pandas as pd
import time
from scipy.stats import nbinom
from numpy.random import default_rng
from bayesreconpy.reconc_BUIS import reconc_BUIS

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------Data and base forecasts-------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

n_b = 5
n_u = 1
n = n_b + n_u

A = np.ones((n_u, n_b))

# Actual values:
actuals = pd.read_pickle('extr_mkt_events.pkl')
#Bse forecasts:
base_fc = (pd.read_pickle('extr_mkt_events_basefc.pkl'))

N = actuals.shape[0]  # number of days (3508)

"""
If you want to run on ly N reconciliations (instead of 3508)
N = 200
actuals = actuals[:N,:]
base_fc['mu'] = np.array(base_fc['mu'])[:N,:]
"""

#-----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------Reconciliation via conditioning---------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

# Initialize matrices to store mean, median, lower, and upper quantiles
rec_means = np.full((N, n), np.nan)
rec_medians = np.full((N, n), np.nan)
rec_L = np.full((N, n), np.nan)
rec_U = np.full((N, n), np.nan)

# Interval coverage
int_cov = 0.9
q1 = (1 - int_cov) / 2
q2 = (1 + int_cov) / 2

# Number of samples for reconciled distribution
N_samples = int(1e4)
rng = default_rng(42)  # Seeded random number generator for reproducibility

# Start timing
start = time.time()

# Loop through each reconciliation
for j in range(N):
    # Prepare base forecasts for the current iteration
    base_fc_j = []
    for i in range(n):
        # Fetch `size` and `mu` for each forecast from the corresponding DataFrame column and row
        size_value = base_fc['size'].iloc[0, i]  # Use row 0 of `size` for all j's
        mu_value = base_fc['mu'].iloc[j, i]
        base_fc_j.append({"size": size_value, "mu": mu_value})

    # Reconcile via importance sampling
    buis = reconc_BUIS(A, base_fc_j, "params", "nbinom", num_samples=N_samples, seed=42)
    samples_y = buis['reconciled_samples']

    # Save mean, median, and quantiles
    rec_means[j, :] = np.mean(samples_y, axis=1)  # Mean along rows
    rec_medians[j, :] = np.median(samples_y, axis=1)  # Median along rows
    rec_L[j, :] = np.quantile(samples_y, q1, axis=1)  # Lower quantile
    rec_U[j, :] = np.quantile(samples_y, q2, axis=1)  # Upper quantile

# End timing
stop = time.time()

# Print computation time
print(f"Computational time for {N} reconciliations: {round(stop - start, 2)} seconds")


# Initialize matrices
base_means = base_fc['mu'].values  # Convert to numpy array for easier manipulation
base_medians = np.full((N, n), np.nan)
base_L = np.full((N, n), np.nan)
base_U = np.full((N, n), np.nan)

# Loop through each column (i.e., each forecast variable)
for i in range(n):
    size_value = base_fc['size'].iloc[0, i]  # Use row 0 of `size` for each i (if size is constant across rows)

    # Calculate the median, lower, and upper quantiles for each value of `mu` in the column
    base_medians[:, i] = [nbinom.ppf(0.5, size_value, 1 / (1 + mu)) for mu in base_means[:, i]]
    base_L[:, i] = [nbinom.ppf(q1, size_value, 1 / (1 + mu)) for mu in base_means[:, i]]
    base_U[:, i] = [nbinom.ppf(q2, size_value, 1 / (1 + mu)) for mu in base_means[:, i]]








