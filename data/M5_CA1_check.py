# Demonstration and test on the M5 dataset as presented in the vignette of the original BayesRecon package in R
import pandas as pd
import numpy as np
from BayesReconPy.PMF import pmf_get_mean as PMF_get_mean
from BayesReconPy.PMF import pmf_get_var as PMF_get_var
from BayesReconPy.shrink_cov import schafer_strimmer_cov

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
mu_u = {k:fc['mu'] for k, fc in base_fc_upper.items()}  # upper means

# Compute the (shrinked) covariance matrix of the residuals
residuals_upper = np.array([fc['residuals'] for fc in base_fc_upper]).T
Sigma_u = schafer_strimmer_cov(residuals_upper)['shrink_cov']  # Assuming a custom function for shrinkage

# Parameters of the bottom base forecast distributions
mu_b = []
sd_b = []

for fc_b in base_fc_bottom:
    pmf = fc_b['pmf']
    mu_b.append(PMF_get_mean(pmf))  # Assuming PMF_get_mean is a defined function
    sd_b.append(PMF_get_var(pmf)**0.5)  # Assuming PMF_get_var is a defined function

mu_b = np.array(mu_b)
sd_b = np.array(sd_b)
Sigma_b = np.diag(sd_b**2)

# Mean and covariance matrix of the base forecasts
base_forecasts_mu = np.concatenate([mu_u, mu_b])
n = len(base_forecasts_mu)
n_u = len(mu_u)

base_forecasts_Sigma = np.zeros((n, n))
base_forecasts_Sigma[:n_u, :n_u] = Sigma_u
base_forecasts_Sigma[n_u:, n_u:] = Sigma_b


