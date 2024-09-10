# Demonstration and test on the M5 dataset as presented in the vignette of the original BayesRecon package in R
import pandas as pd
import numpy as np
import time
from BayesReconPy.PMF import pmf_get_mean as PMF_get_mean
from BayesReconPy.PMF import pmf_get_var as PMF_get_var
from BayesReconPy.shrink_cov import schafer_strimmer_cov
from BayesReconPy.reconc_gaussian import reconc_gaussian
from BayesReconPy.reconc_MixCond import reconc_MixCond
from BayesReconPy.reconc_TDcond import reconc_TDcond

M5_CA1_basefc = pd.read_pickle('data/M5_CA1_basefc.pkl')

n_b = 3049
n_u = 11
n = n_u + n_b

#Load A matrix

A = M5_CA1_basefc['A']

# Load base forecasts
base_fc_upper = M5_CA1_basefc['upper']
base_fc_bottom = M5_CA1_basefc['bottom']

# Initialize a dictionary to store the results
rec_fc = {
    'Gauss': {},
    'Mixed_cond': {},
    'TD_cond': {}
}

# Parameters of the upper base forecast distributions
mu_u = {k:fc['mu'] for k, fc in base_fc_upper.items()} # upper means

# Create a dictionary to store the names with their corresponding residuals
residuals_dict = {fc: np.array(base_fc_upper[fc]['residuals']) for fc in base_fc_upper if 'residuals' in base_fc_upper[fc]}
for name, residuals in residuals_dict.items():
    print(f"Name: {name}, Residuals shape: {residuals.shape}")

residuals_upper = np.vstack([residuals for residuals in residuals_dict.values()]).T

# Compute the (shrinked) covariance matrix of the residuals

Sigma_u = schafer_strimmer_cov(residuals_upper)['shrink_cov']  # Assuming a custom function for shrinkage
Sigma_u = {
    'names': list(residuals_dict.keys()),  # List of names corresponding to the diagonal elements
    'Sigma_u': Sigma_u           # Covariance matrix
}

# Parameters of the bottom base forecast distributions
mu_b = {}
sd_b = {}

# Loop through base_fc_bottom and calculate the mean and standard deviation for each pmf
for k, fc in base_fc_bottom.items():
    pmf = fc['pmf']  # Access 'pmf' inside each forecast entry

    # Calculate the mean and standard deviation
    mu_b_value = PMF_get_mean(pmf)
    sd_b_value = PMF_get_var(pmf) ** 0.5

    # Store the results in dictionaries with the key as the name
    mu_b[k] = mu_b_value
    sd_b[k] = sd_b_value

# Create the covariance matrix (Sigma_b)
Sigma_b = np.diag(np.array(list(sd_b.values())) ** 2)
Sigma_b = {
    'names': list(sd_b.keys()),  # List of names corresponding to the diagonal elements
    'Sigma_b': Sigma_b           # Covariance matrix
}

# Mean and covariance matrix of the base forecasts
base_forecasts_mu = {**mu_u, **mu_b}
base_forecasts_Sigma = np.zeros((n, n))
# Fill the upper-left block with Sigma_u
base_forecasts_Sigma[:n_u, :n_u] = Sigma_u['Sigma_u']  # Upper block
# Fill the bottom-right block with Sigma_b
base_forecasts_Sigma[n_u:, n_u:] = Sigma_b['Sigma_b']  # Bottom block
# Combine the names from both Sigma_u and Sigma_b
combined_names = Sigma_u['names'] + Sigma_b['names']
# Store the combined matrix and names in a dictionary
base_forecasts_Sigma = {
    'names': combined_names,       # Combined list of names
    'Sigma': base_forecasts_Sigma  # Full covariance matrix
}

#-----------------------------------------------------------------------------------------------------
#-------------------------GAUSSIAN RECONCILIATION-----------------------------------------------------
#-----------------------------------------------------------------------------------------------------

start = time.time()
gauss = reconc_gaussian(A, list(base_forecasts_mu.values()),
                        base_forecasts_Sigma['Sigma'])
stop = time.time()

# Create a dictionary for the reconciled forecasts, similar to rec_fc$Gauss in R
rec_fc['Gauss'] = {
    'mu_b': gauss['bottom_reconciled_mean'],            # Bottom-level reconciled mean
    'Sigma_b': gauss['bottom_reconciled_covariance'],   # Bottom-level reconciled covariance
    'mu_u': A @ gauss['bottom_reconciled_mean'],        # Upper-level reconciled mean
    'Sigma_u': A @ gauss['bottom_reconciled_covariance'] @ A.T  # Upper-level reconciled covariance
}

# Calculate the time taken for reconciliation
Gauss_time = round(stop - start, 2)

# Output the time taken for reconciliation
print(f"Time taken by Gaussian reconciliation: {Gauss_time} seconds")

#-----------------------------------------------------------------------------------------------------
#----------------------------MIXED RECONCILIATION-----------------------------------------------------
#-----------------------------------------------------------------------------------------------------

seed = 1
N_samples_IS = int(5e4)  # 50,000 samples

# Base forecasts
Sigma_u_np = np.array(Sigma_u['Sigma_u'])
fc_upper_4rec = {'mu': mu_u, 'Sigma': Sigma_u_np}  # Dictionary for upper forecasts
fc_bottom_4rec = {k: np.array(fc['pmf']) for k, fc in base_fc_bottom.items()}

# Set random seed for reproducibility
np.random.seed(seed)

# Start timing
start = time.time()

# Perform MixCond reconciliation (assuming reconc_MixCond is implemented)
mix_cond = reconc_MixCond(A, fc_bottom_4rec, fc_upper_4rec, bottom_in_type="pmf",
                          num_samples=N_samples_IS, return_type="pmf", seed=seed)

# Stop timing
stop = time.time()

# Create a dictionary for storing MixCond reconciliation results, similar to rec_fc$Mixed_cond in R
rec_fc['Mixed_cond'] = {
    'bottom': mix_cond['bottom_reconciled']['pmf'],  # Bottom-level reconciled PMFs
    'upper': mix_cond['upper_reconciled']['pmf'],    # Upper-level reconciled PMFs
    'ESS': mix_cond['ESS']                           # Effective Sample Size (ESS)
}

# Calculate the time taken for MixCond reconciliation
MixCond_time = round(stop - start, 2)

# Output the time taken for MixCond reconciliation
print(f"Computational time for Mix-cond reconciliation: {MixCond_time} seconds")


#-----------------------------------------------------------------------------------------------------
#-------------------------TOP-DOWN RECONCILIATION-----------------------------------------------------
#-----------------------------------------------------------------------------------------------------

N_samples_TD = int(1e4)

# Start timing
start = time.time()

# Perform TD-cond reconciliation (assuming reconc_TDcond is implemented)
# This will raise a warning if upper samples are discarded
td = reconc_TDcond(A, fc_bottom_4rec, fc_upper_4rec,
                   bottom_in_type="pmf", num_samples=N_samples_TD,
                   return_type="pmf", seed=seed)

# Stop timing
stop = time.time()

# Store the results in the rec_fc dictionary
rec_fc['TD_cond'] = {
    'bottom': td['bottom_reconciled']['pmf'],
    'upper': td['upper_reconciled']['pmf']
}

# Calculate the time taken for TD-cond reconciliation
TDCond_time = round(stop - start, 2)
print(f"Computational time for TD-cond reconciliation: {TDCond_time} seconds")



