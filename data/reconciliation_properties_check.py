import numpy as np
import pandas as pd
import time
from scipy.stats import nbinom
from numpy.random import default_rng
from bayesreconpy.reconc_buis import reconc_buis

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
    buis = reconc_buis(A, base_fc_j, "params", "nbinom", num_samples=N_samples, seed=42)
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



# Compute the squared errors
SE_base = (base_means - actuals) ** 2
SE_rec = (rec_means - actuals) ** 2

# Compute the absolute errors
AE_base = np.abs(base_medians - actuals)
AE_rec = np.abs(rec_medians - actuals)

# Define the interval score function
def interval_score(l, u, actual, int_cov=0.9):
    is_score = (u - l) + \
               (2 / (1 - int_cov)) * (actual - u) * (actual > u) + \
               (2 / (1 - int_cov)) * (l - actual) * (l > actual)
    return is_score

# Vectorized computation of interval scores
IS_base = np.vectorize(interval_score)(base_L, base_U, actuals)
IS_rec = np.vectorize(interval_score)(rec_L, rec_U, actuals)

# Reshape the results into N x n matrices if needed
IS_base = IS_base.reshape(N, n)
IS_rec = IS_rec.reshape(N, n)



# Compute skill scores (SS) for Absolute Error, Squared Error, and Interval Score
SS_AE = (AE_base - AE_rec) / (AE_base + AE_rec) * 2
SS_SE = (SE_base - SE_rec) / (SE_base + SE_rec) * 2
SS_IS = (IS_base - IS_rec) / (IS_base + IS_rec) * 2

# Replace NaN values with 0
SS_AE = np.nan_to_num(SS_AE, nan=0)
SS_SE = np.nan_to_num(SS_SE, nan=0)
SS_IS = np.nan_to_num(SS_IS, nan=0)

# Calculate column means for each skill score matrix
mean_skill_scores = np.round([
    SS_IS.mean(axis=0),
    SS_SE.mean(axis=0),
    SS_AE.mean(axis=0)
], 2)

# Convert to DataFrame and structure the output
mean_skill_scores_df = pd.DataFrame(mean_skill_scores,
                                    index=["Interval score", "Squared error", "Absolute error"],
                                    columns=actuals.columns if isinstance(actuals, pd.DataFrame) else range(n))

# Display the results as a table
print(mean_skill_scores_df.to_markdown())

#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------Reconciled mean and variance----------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


# Define the number of samples to draw
N_samples = int(1e5)

# Example of concordant-shift effect for j = 123 (124 in R)
j = 123
base_fc_j = []

# Prepare base forecast for the specified index
for i in range(n):
    size_value = base_fc['size'].iloc[0, i]  # size is constant across rows
    mu_value = base_fc['mu'].iloc[j, i]
    base_fc_j.append({"size": size_value, "mu": mu_value})

# Reconcile via importance sampling
buis = reconc_buis(A, base_fc_j, "params", "nbinom", num_samples=N_samples, seed=42)
samples_y = buis['reconciled_samples']

# Compute the means
base_upper_mean = round(base_fc['mu'].iloc[j, 0], 2)
bottom_up_upper_mean = round(base_fc['mu'].iloc[j, 1:].sum(), 2)
reconciled_upper_mean = round(np.mean(samples_y[0, :]), 2)

# Display results in a structured format
means = [base_upper_mean, bottom_up_upper_mean, reconciled_upper_mean]
col_names = ["Base upper mean", "Bottom-up upper mean", "Reconciled upper mean"]

# Create a DataFrame to display the results as a table
means_df = pd.DataFrame([means], columns=col_names)
print(means_df.to_markdown())



j = 1699
base_fc_j = []

# Prepare base forecast for the specified index
for i in range(n):
    size_value = base_fc['size'].iloc[0, i]  # Assume size is constant across rows
    mu_value = base_fc['mu'].iloc[j, i]
    base_fc_j.append({"size": size_value, "mu": mu_value})

# Reconcile via importance sampling
buis = reconc_buis(A, base_fc_j, "params", "nbinom", num_samples=N_samples, seed=42)
samples_y = buis['reconciled_samples']

# Compute the means
base_upper_mean = round(base_fc['mu'].iloc[j, 0], 2)
bottom_up_upper_mean = round(base_fc['mu'].iloc[j, 1:].sum(), 2)
reconciled_upper_mean = round(np.mean(samples_y[0, :]), 2)

# Display results in a structured format
means = [base_upper_mean, bottom_up_upper_mean, reconciled_upper_mean]
col_names = ["Base upper mean", "Bottom-up upper mean", "Reconciled upper mean"]

# Create a DataFrame to display the results as a table
means_df = pd.DataFrame([means], columns=col_names)
print(means_df.to_markdown())




j = 2307

# Prepare base forecast for the specified index
base_fc_j = []
for i in range(n):
    size_value = base_fc['size'].iloc[0, i]  # Assume size is constant across rows
    mu_value = base_fc['mu'].iloc[j, i]
    base_fc_j.append({"size": size_value, "mu": mu_value})

# Reconcile via importance sampling
buis = reconc_buis(A, base_fc_j, "params", "nbinom", num_samples=N_samples, seed=42)
samples_y = buis['reconciled_samples']

# Compute variance of the base bottom forecasts
# Generate 100,000 samples for each pair of (size, mu) and calculate the variance
base_bottom_var = [
    np.var(np.random.negative_binomial(n=size, p=size / (size + mu), size=int(1e5)))
    for mu, size in zip(base_fc['mu'].iloc[j, 1:],
                        base_fc['size'].iloc[0, 1:])
]

# Compute variance of the reconciled bottom forecasts
# Take the variance along columns for each row in samples_y[1:, :]
rec_bottom_var = np.var(samples_y[1:, :], axis=1)

# Combine base and reconciled variances and display results
bottom_var = np.vstack([base_bottom_var, rec_bottom_var])
bottom_var_df = pd.DataFrame(bottom_var, index=["var base", "var reconc"])

# Display as a table with two decimal places
print(bottom_var_df.round(2).to_markdown())