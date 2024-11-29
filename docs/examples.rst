Examples and Tutorials
======================

This page contains both code examples for using the `bayesreconpy` package and links to Jupyter notebooks for detailed walkthroughs.

Example 1: Bayesian Reconciliation
-----------------------------------

This page reproduces the results as presented in *Probabilistic reconciliation of mixed-type hierarchical time series* (Zambon et al. 2024), published at UAI 2024 (the 40th Conference on Uncertainty in Artificial Intelligence).

In particular, we replicate the reconciliation of the one-step ahead (h=1) forecasts of one store of the M5 competition (Makridakis, Spiliotis, and Assimakopoulos 2022). Sect. 5 of the paper presents the results for 10 stores, each reconciled 14 times using rolling one-step ahead forecasts.
The original vignette containing the R counterpart of this page can be found `here <https://cran.r-project.org/web/packages/bayesRecon/vignettes/mixed_reconciliation.html>`_.

Data and Base Forecasts
=======================

The M5 competition (Makridakis, Spiliotis, and Assimakopoulos 2022) is about daily time series of sales data referring to 10 different stores. Each store has the same hierarchy: 3049 bottom time series (single items) and 11 upper time series, obtained by aggregating the items by department, product category, and store; see the figure below.

.. image:: https://raw.githubusercontent.com/supsi-dacd-isaac/BayesReconPy/refs/heads/main/pictures/M5.png
   :alt: M5 competition data hierarchy

We reproduce the results of the store “CA_1”. The base forecasts (for h=1) of the bottom and upper time series are stored in `M5_CA1_basefc` available as data in the original `bayesRecon <https://cran.r-project.org/web/packages/bayesRecon/index.html>`_ package in R. The base forecasts are computed using ADAM (Svetunkov and Boylan 2023), implemented in the R package smooth (Svetunkov 2023).


.. code-block:: python

   import pandas as pd
   import numpy as np
   import time

   M5_CA1_basefc = pd.read_pickle('data/M5_CA1_basefc.pkl')

   # Hierarchy composed of 3060 time series: 3049 bottom and 11 upper
   n_b = 3049
   n_u = 11
   n = n_u + n_b

   # Load A matrix
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


Gaussian Reconciliation
=======================

We first perform Gaussian reconciliation (`Gauss`, Corani et al. (2021)). It assumes all forecasts to be Gaussian, even though the bottom base forecasts are not Gaussian.

We assume the upper base forecasts to be a multivariate Gaussian and we estimate their covariance matrix from the in-sample residuals. We assume also the bottom base forecasts to be independent Gaussians.

Example Code
------------

The following Python code demonstrates how to perform the Gaussian reconciliation:

.. code-block:: python

   # Parameters of the upper base forecast distributions
   mu_u = {k: fc['mu'] for k, fc in base_fc_upper.items()}  # upper means

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

We reconcile using the function :ref:`reconc_gaussian() <bayesreconpy.reconc_gaussian>`, which takes as input:

- the summing matrix `A`;
- the means of the base forecast, `base_forecasts_mu`;
- the covariance of the base forecast, `base_forecasts_Sigma`.

The function returns the reconciled mean and covariance for the bottom time series.

.. code-block:: python

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
   # Time taken by Gaussian reconciliation: 0.33 seconds


