# Demonstration and test on the M5 dataset as presented in the vignette of the original BayesRecon package in R
import pandas as pd
import numpy as np
import time
from bayesreconpy.PMF import _pmf_get_mean as PMF_get_mean
from bayesreconpy.PMF import _pmf_get_var as PMF_get_var
from bayesreconpy.shrink_cov import _schafer_strimmer_cov
from bayesreconpy.reconc_gaussian import reconc_gaussian
from bayesreconpy.reconc_mix_cond import reconc_mix_cond
from bayesreconpy.reconc_td_cond import reconc_td_cond
from bayesreconpy.utils import _MVN_sample, _samples_from_pmf
from bayesreconpy.reconc_buis import reconc_buis

M5_CA1_basefc = pd.read_pickle('../data/M5_CA1_basefc.pkl')

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

Sigma_u = _schafer_strimmer_cov(residuals_upper)['shrink_cov']  # Assuming a custom function for shrinkage
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

# Perform MixCond reconciliation (assuming reconc_mix_cond is implemented)
mix_cond = reconc_mix_cond(A, fc_bottom_4rec, fc_upper_4rec, bottom_in_type="pmf",
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

# Perform TD-cond reconciliation 
# This will raise a warning if upper samples are discarded
td = reconc_td_cond(A, fc_bottom_4rec, fc_upper_4rec,
                   bottom_in_type="pmf", num_samples=N_samples_TD,
                   return_type="samples", seed=seed)

# Stop timing
stop = time.time()

# Store the results in the rec_fc dictionary
rec_fc['TD_cond'] = {
    'bottom': td['bottom_reconciled']['samples'],
    'upper': td['upper_reconciled']['samples']
}

# Calculate the time taken for TD-cond reconciliation
TDCond_time = round(stop - start, 2)
print(f"Computational time for TD-cond reconciliation: {TDCond_time} seconds")


#-----------------------------------------------------------------------------------------------------
#-------------------------BUIS RECONCILIATION-----------------------------------------------------
#-----------------------------------------------------------------------------------------------------

n_buis = int(1e5)
mus = np.array(list(fc_upper_4rec['mu'].values()))
upp_fore_samp = _MVN_sample(n_buis, mus, fc_upper_4rec['Sigma'])
upp_fore_samp = np.maximum(0,upp_fore_samp)

idx = list(fc_bottom_4rec.keys())

bot_fore_samp = np.zeros((n_buis,n_b))
for i in range(n_b):
    bot_fore_samp[:,i] = _samples_from_pmf(fc_bottom_4rec[idx[i]], n_buis)

fc_samples = np.column_stack((upp_fore_samp, bot_fore_samp))
fc_4buis = []

for i in range(fc_samples.shape[1]):
    fc_4buis.append(np.round(fc_samples[:, i]))

start = time.time()
BUIS_rec = reconc_buis(
  A,
  base_forecasts = fc_4buis,
  in_type = "samples",
  distr = "discrete",
  seed = 1
)

stop = time.time()

# Calculate the time taken for BUIS reconciliation
BUIS_time = round(stop - start, 2)
print(f"Computational time for BUIS reconciliation: {BUIS_time} seconds")
