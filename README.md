# BayesReconPy
[![codecov](https://codecov.io/github/supsi-dacd-isaac/BayesReconPy/graph/badge.svg?token=KND9ZJ4GOQ)](https://codecov.io/github/supsi-dacd-isaac/BayesReconPy)
[![License: LGPL (\>= 3)](https://img.shields.io/badge/license-LGPL%20(%3E=%203)-yellow.svg)](https://www.gnu.org/licences/lgpl-3.0)
![PyPI](https://img.shields.io/pypi/v/bayesreconpy)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
[![Downloads](https://static.pepy.tech/badge/bayesreconpy)](https://pepy.tech/project/bayesreconpy)

<img src="https://github.com/supsi-dacd-isaac/BayesReconPy/blob/main/pictures/BAyesreCONpy.png?raw=true" width="400"/>

## Bayesian Reconciliation for Hierarchical Forecasting 

Forecast reconciliation ensures that probabilistic forecasts across hierarchical or grouped time series remain coherent, meaning that forecasts at disaggregated levels (e.g., local components) are consistent with those at aggregated levels (e.g., system totals). It is a post-processing technique applied to a set of independently generated incoherent base forecasts that do not obey these hierarchical constraints.

While several methods exist for point forecast reconciliation, BayesReconPy focuses on probabilistic reconciliation of different kinds of hierarchical and grouped time series in python. Most of the existing tools in forecast reconciliation are limited to Gaussian or continuous inputs, lack support for discrete or mixed-type forecasts, or are implemented only in R. This package supports reconciliation of discrete and non-Gaussian forecast distributions using Bayesian forecast reconciliation via conditioning methods, which are common in domains such as energy systems, demand forecasting, and risk analysis.

This code is an implementation of the 
[original R package](https://cran.r-project.org/web/packages/bayesRecon/index.html). 

The python package is available in pip, use the following line of command for installation.

```python
pip install bayesreconpy
```
The documentation of the reconciliation functions is available [here](https://bayesreconpy.readthedocs.io/en/latest/).

## Introduction

This page reproduces the results as presented in *Probabilistic reconciliation of mixed-type hierarchical time series* (Zambon et al. 2024), published at UAI 2024 (the 40th Conference on Uncertainty in Artificial Intelligence).

In particular, we replicate the reconciliation of the one-step ahead (h=1) forecasts of one store of the M5 competition (Makridakis, Spiliotis, and Assimakopoulos 2022). Sect. 5 of the paper presents the results for 10 stores, each reconciled 14 times using rolling one-step ahead forecasts.
The original vignette containing the R counterpart of this page can be found [here](https://cran.r-project.org/web/packages/bayesRecon/vignettes/mixed_reconciliation.html).

## Data and base forecasts

The M5 competition (Makridakis, Spiliotis, and Assimakopoulos 2022) is about daily time series of sales data referring to 10 different stores. Each store has the same hierarchy: 3049 bottom time series (single items) and 11 upper time series, obtained by aggregating the items by department, product category, and store; see the figure below.

![M5.png](https://github.com/supsi-dacd-isaac/BayesReconPy/blob/main/pictures/M5.png?raw=true)

We reproduce the results of the store “CA_1”. The base forecasts (for h=1) of the bottom and upper time series are stored in `M5_CA1_basefc` available as data in the original **‘[bayesRecon](https://cran.r-project.org/web/packages/bayesRecon/index.html)’** package in R. The base forecast are computed using ADAM (Svetunkov and Boylan 2023), implemented in the R package smooth (Svetunkov 2023).

```python
import pandas as pd
import numpy as np
import time

M5_CA1_basefc = pd.read_pickle('data/M5_CA1_basefc.pkl')

# Hierarchy composed by 3060 time series: 3049 bottom and 11 upper
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
```

## Gaussian Reconciliation

We first perform Gaussian reconciliation (`Gauss`, Corani et al. (2021)). It assumes all forecasts to be Gaussian, even though the bottom base forecasts are not Gaussian.

We assume the upper base forecasts to be a multivariate Gaussian and we estimate their covariance matrix from the in-sample residuals. We assume also the bottom base forecasts to be independent Gaussians.

```python
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
```

We reconcile using the function `reconc_gaussian()`, which takes as input:

- the summing matrix `A`;
- the means of the base forecast, `base_forecasts_mu`;
- the covariance of the base forecast, `base_forecasts_Sigma`.

The function returns the reconciled mean and covariance for the bottom time series.

```python
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
#Time taken by Gaussian reconciliation: 0.33 seconds
```

## Reconciliation with mixed-conditioning

We now reconcile the forecasts using the mixed-conditioning approach of Zambon et al. (2024), Sect. 3. The algorithm is implemented in the function `reconc_mix_cond()`. The function takes as input:

- the aggregation matrix `A`;
- the probability mass functions of the bottom base forecasts, stored in the list `fc_bottom_4rec`;
- the parameters of the multivariate Gaussian distribution for the upper variables, `fc_upper_4rec`;
- additional function parameters; among those note that `num_samples` specifies the number of samples used in the internal importance sampling (IS) algorithm.

The function returns the reconciled forecasts in the form of probability mass functions for both the upper and bottom time series. The function parameter `return_type` can be changed to `samples` or `all` to obtain the IS samples.

```python
seed = 1
N_samples_IS = int(5e4)  # 50,000 samples

# Base forecasts
Sigma_u_np = np.array(Sigma_u['Sigma_u'])
fc_upper_4rec = {'mu': mu_u, 'Sigma': Sigma_u_np}  # Dictionary for upper forecasts
fc_bottom_4rec = {k: np.array(fc['pmf']) for k, fc in base_fc_bottom.items()}

# Set random seed for reproducibility
np.random.seed(seed)

start = time.time()

# Perform MixCond reconciliation
mix_cond = reconc_mix_cond(A, fc_bottom_4rec, fc_upper_4rec, bottom_in_type="pmf",
                          num_samples=N_samples_IS, return_type="pmf", seed=seed)

stop = time.time()

rec_fc['Mixed_cond'] = {
    'bottom': mix_cond['bottom_reconciled']['pmf'],  # Bottom-level reconciled PMFs
    'upper': mix_cond['upper_reconciled']['pmf'],    # Upper-level reconciled PMFs
    'ESS': mix_cond['ESS']                           # Effective Sample Size (ESS)
}

# Calculate the time taken for MixCond reconciliation
MixCond_time = round(stop - start, 2)

print(f"Computational time for Mix-cond reconciliation: {MixCond_time} seconds")
#Computational time for Mix-cond reconciliation: 8.51 seconds
```

As discussed in Zambon et al. (2024), Sect. 3, conditioning with mixed variables performs poorly in high dimensions. This is because the bottom-up distribution, built by assuming the bottom forecasts to be independent, is untenable in high dimensions. Moreover, forecasts for count time series are usually biased and their sum tends to be strongly biased; see Zambon et al. (2024), Fig. 3, for a graphical example.

## Top down conditioning

Top down conditioning (TD-cond; see Zambon et al. (2024), Sect. 4) is a more reliable approach for reconciling mixed variables in high dimensions. The algorithm is implemented in the function `reconc_td_cond()`; it takes the same arguments as `reconc_mix_cond()` and returns reconciled forecasts in the same format.

```python
N_samples_TD = int(1e4)

start = time.time()

# This will raise a warning if upper samples are discarded
td = reconc_td_cond(A, fc_bottom_4rec, fc_upper_4rec,
                   bottom_in_type="pmf", num_samples=N_samples_TD,
                   return_type="pmf", seed=seed)
#Warning: Only 99.6% of the upper samples are in the support of the 
#bottom-up distribution; the others are discarded.
stop = time.time()
```

The algorithm TD-cond raises a warning regarding the incoherence between the joint bottom-up and the upper base forecasts. We will see that this warning does not impact the performances of TD-cond. An important note to be made here is, R and python uses different sampling schemes even with the same seed. So we might see some minor deviations of the results than the ones presented in R. But as we increase `N_samples_TD` , it becomes negligible.

```python
rec_fc['TD_cond'] = {
    'bottom': td['bottom_reconciled']['pmf'],
    'upper': td['upper_reconciled']['pmf']
}

TDCond_time = round(stop - start, 2)
print(f"Computational time for TD-cond reconciliation: {TDCond_time} seconds")
#Computational time for TD-cond reconciliation: 10.03 seconds
```

The computational time required for the Gaussian reconciliation is 0.33 seconds, Mix-cond requires 8.51 seconds and TD-cond requires 10.03 seconds.

# Comparison with results from the R version

Let us see the results as obtained from the R and python versions and how much they differ depending on the number of samples that we select for the methods. For all the tested methods we report the relative absolute difference between the two implementations.

## Gaussian Reconciliation

### Upper reconciled forecasts

In the following we report the results for python and R (first and second blocks respectively) and their relative absolute differences (third block) for the top level and the first two levels of aggregations (1 + 3 + 7).   

```python
**# Python results block**

mean_upp      sd_upp
4854.296252   52290.797028
541.381159    4462.815651
1008.554623   2181.136420
3304.360470   35798.312200
496.550790    4344.021302
44.830369     88.922831
781.222525    1687.316414
227.332098    267.074008
307.255027    1438.765176
561.940723    1418.313872
2435.164720   28499.425414
```

```r
**# R results block**

mean_upp      sd_upp
4854.29625   52290.79703
541.38116    4462.81565
1008.55462   2181.13642
3304.36047   35798.31220
496.55079    4344.02130
44.83037     88.92283
781.22252    1687.31641
227.33210    267.07401
307.25503    1438.76518
561.94072    1418.31387
2435.16472   28499.42541
```

```
**# Relative difference in results**

mean_upp      sd_upp
3.372457e-15  1.127067e-14
6.299822e-16  2.445527e-14
1.352670e-15  8.339641e-16
5.504815e-16  1.097543e-14
3.434296e-16  2.679898e-14
1.584958e-16  6.392444e-16
1.309718e-15  1.078037e-15
3.750686e-16  2.128377e-16
3.700081e-16  1.896407e-15
0.000000e+00  8.015633e-16
7.469677e-16  1.506281e-14
```

Thus, it is evident that for the upper reconciled forecast in the Gaussian case, the results in both the versions are exactly the same. 

### Bottom reconciled forecasts

Since there are 3049 bottom time series, possibly of different scales, creating a tabulated or graphical representation might not be very fruitful to understand the difference in the results from the two versions. We report here the average, min and max (across bottom time series) means and standard deviations of the absolute differences of the reconciled distribution for the two implementations. 

| **Type of absolute diff** | **mean** | **min** | **max** |
| --- | --- | --- | --- |
| **bottom mean** | 1.151872e-15 | 0 | 1.417533e-12 |
| **bottom sd** | 7.205875e-14 | 0 | 1.818989e-10 |

Even for the bottom reconciled forecasts, it is evident that the results are identical in both versions.

## Mixed Reconciliation

### Upper reconciled forecasts

In the following we report results using `5e4` samples:

```python
**# Python results block**

mean_upp      sd_upp
4834.84272    26843.743863
526.45896     2894.190676
1005.55656    1845.834761
3302.82720    18079.291580
483.79212     2808.996466
42.66684      70.406284
778.38700     1486.009071
227.16956     215.722249
308.13638     869.121860
557.81176     1247.466166
2436.87906    14259.761434
```

```r
**# R results block**

mean_upp      sd_upp
4831.47620   26874.53939
525.86408    2923.07265
1004.65180   1837.12684
3300.96032   17803.50131
483.29234    2827.82364
42.57174     68.62841
777.16284    1479.15352
227.48896    217.11252
308.49018    835.84250
557.65042    1242.31781
2434.81972   14176.55698
```

```
**# Relative difference in results**

mean_upp      sd_upp
0.0006967891  0.001145900
0.0011312429  0.009880688
0.0009005707  0.004739969
0.0005655566  0.015490789
0.0010341153  0.006657831
0.0022338763  0.025905758
0.0015751654  0.004634778
0.0014040242  0.006403448
0.0011468761  0.039815344
0.0002893210  0.004144150
0.0008457875  0.005869158
```

As mentioned earlier, due to separate sampling schemes in the two methods, the results are not exactly the same, but are not significantly different as well; as shown in the third column.  Below we show the results for another run with `1e5` samples.

```python
**# Python results block**

mean_upp      sd_up
4832.21057   26997.942230
526.09658    2918.252292
1004.42229   1827.972121
3301.69170   18146.535071
483.49047    2836.275769
42.60611     70.210821
777.42705    1475.302418
226.99524    214.773697
308.57387    846.660863
557.73657    1245.414535
2435.38126   14448.807701
```

```r
**# R results block**

mean_upp      sd_upp
4831.7351   27201.38314
526.1165    2911.60358
1004.9725   1844.02286
3300.6462   18111.43334
483.4630    2831.65269
42.6535     69.79374
777.7951    1490.83197
227.1773    217.35468
308.1514    851.71696
557.4704    1239.35909
2435.0244   14369.48835
```

```
**# Relative difference in results**

mean_upp      sd_upp
9.839529e-05 0.007479065
3.780532e-05 0.002283521
5.474478e-04 0.008704198
3.167501e-04 0.001938098
5.688130e-05 0.001632644
1.111046e-03 0.005975936
4.732095e-04 0.010416702
8.016204e-04 0.011874507
1.371079e-03 0.005936355
4.773885e-04 0.004885952
1.465488e-04 0.005519984
```

It is clear that the slight disparity in the results obtained from the two programs were possibly due to the different source of randomness used by python and R, as doubling the sample number gave us a lower relative difference.

### Bottom reconciled forecasts

| **Type of absolute diff** | **mean** | **min** | **max** |
| --- | --- | --- | --- |
| **bottom mean** | 0.03162734  | 1.345008e-16  |    3 |
| **bottom sd** | 0.07311957  | 2.235638e-05  | 34.20402 |

In this case, although the mean is not significantly different across the results over the bottoms, there seems to be some values with considerably moderate deviations. This is again due to the fact that the two programs use different sampling schemes. Below we show a similar table with important samples number `1e5` , and conclude that consequently the difference in results between the two versions become insignifacnt with the increase in the number of importance samples.

### Bottom reconciled forecasts

| **Type of absolute diff** | **mean** | **min** | **max** |
| --- | --- | --- | --- |
| **bottom mean** | 0.02055766  | 1.685271e-16  | 0.8253752 |
| **bottom sd** | 0.03650475 |  8.648061e-07  | 1.708742 |

## Top-Down Reconciliation

### Upper reconciled forecasts

```python
**# Python results block**

mean_upp      sd_upp
4691.813190   178932.958979
527.268721    7482.460313
977.571572    7751.962003
3186.972897   99096.312656
482.917988    7169.427625
44.350733     125.796882
761.602188    5380.489909
215.969384    628.263367
279.059627    2912.150229
557.031721    5094.022483
2350.881550   61367.289322
```

```r
**# R results block**

mean_upp      sd_upp
4692.00432   177133.1774
527.76819    7560.4684
978.36842    7888.2681
3185.86771   97841.0370
483.48191    7245.0635
44.28629     122.3439
762.26739    5540.3372
216.10103    617.9973
279.97939    2998.3752
557.63209    4931.4386
2348.25623   60725.2483
```

```
**# Relative difference in results**

mean_upp      sd_upp
4.073576e-05 0.01016061
9.463880e-04 0.01031789
8.144619e-04 0.01727959
3.469023e-04 0.01282975
1.166367e-03 0.01043964
1.455170e-03 0.02822397
8.726624e-04 0.02885155
6.091673e-04 0.01661178
3.285121e-03 0.02875724
1.076635e-03 0.03296885
1.117986e-03 0.01057288
```

The upper reconciled forecasts are quite similar in both the versions. The results obtained are for a reconciled sample size of `1e4` . It can be observed that if we increase the number to `1e5`  the results become even more similar (shown below), indicating that the difference in results could be due to the different sampling schemes taken by the two programs.

```python
**# Python results block**

mean_upp      sd_upp
4693.260694   178635.484472
528.060886    7352.920225
977.689407    7860.534812
3187.510400   99278.321735
483.717462    7052.173030
44.343424     124.671416
761.671115    5468.203702
216.018292    627.203095
279.819869    2923.050604
558.017319    5084.134336
2349.673212   61380.150522
```

```r
**# R results block**

mean_upp      sd_upp
4693.21820   178460.7946
528.00051    7430.1095
977.79582    7848.2358
3187.42187   98890.6581
483.64660    7131.1263
44.35391     124.9065
761.67391    5470.2377
216.12191    622.5927
279.97954    2939.8416
557.93892    5072.3048
2349.50341   61111.4013
```

```
**# Relative difference in results**

mean_upp      sd_upp
9.054159e-06  0.0009788700
1.143458e-04  0.0103887085
1.088269e-04  0.0015671022
2.777449e-05  0.0039201237
1.465075e-04  0.0110716432
2.363542e-04  0.0018819917
3.667139e-06  0.0003718258
4.794394e-04  0.0074052223
5.702926e-04  0.0057115334
1.405151e-04  0.0023321895
7.227078e-05  0.0043976929
```

### Bottom reconciled forecasts

| **Type of absolute diff** | **mean** | **min** | **max** |
| --- | --- | --- | --- |
| **bottom mean** | 0.03565712 |  1.131059e-06 |  5.798125 |
| **bottom sd** | 0.1171438 |  6.838849e-06  | 139.6543 |

From the above table with a reconciled sample size `1e4` it seems that there could be a few outliers, but the results are not so different. Below we do another run with the increased number of TD samples `1e5` and observe that the situation is similar to the one obtained for the uppers - the difference looks negligible.

| **Type of absolute diff** | **mean** | **min** | **max** |
| --- | --- | --- | --- |
| **bottom mean** | 0.01033775 |  1.794681e-06  | 0.6009102 |
| **bottom sd** | 0.0184495 |  2.177446e-05  | 1.254872 |

## BUIS algorithm for the M5 dataset

Below we demonstrate the BUIS algorithm on the same portion of the M5 dataset as before. The code to generate the reconciled forecasts are shown below in both R and python, since the original vignette demonstrating the reconciliation on the M5 dataset in R does not include the BUIS algorithm.

```python
n_buis = int(1e4)
seed = 1

mus = np.array(list(fc_upper_4rec['mu'].values()))
upp_fore_samp = MVN_sample(n_buis, mus, fc_upper_4rec['Sigma'])
idx = list(fc_bottom_4rec.keys())

bot_fore_samp = np.zeros((n_buis,n_b))
for i in range(n_b):
    bot_fore_samp[:,i] = samples_from_pmf(fc_bottom_4rec[idx[i]], n_buis)

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

#Computational time for BUIS reconciliation: 2.45 seconds
```

```r
n_buis <- 1e4
seed <- 1

upp_fore_samp <- .MVN_sample(n_buis,fc_upper_4rec$mu,fc_upper_4rec$Sigma)

bot_fore_samp <- matrix(0,n_buis,n_b)
for(i in seq(n_b)){
  support <- seq(0,length(fc_bottom_4rec[[i]])-1,1)
  bot_fore_samp[,i] <- sample(support,n_buis,prob=fc_bottom_4rec[[i]],replace = TRUE)
}

fc_samples <- cbind(upp_fore_samp,bot_fore_samp)
fc_4buis <- list()

for(i in seq(ncol(fc_samples))){
  fc_4buis[[i]] <- round(fc_samples[,i])
}

start <- Sys.time()   
BUIS_rec <-reconc_buis(
  A,
  base_forecasts = fc_4buis,
  in_type = "samples",
  distr = "discrete",
  seed = seed
)

stop <- Sys.time()

BUIS_time <- as.double(round(difftime(stop, start, units = "secs"), 2))
cat("Computational time for BUIS reconciliation: ", BUIS_time, "s")

#Computational time for BUIS reconciliation: 2.26 seconds
```

We illustrate the results in a similar way as we did for the previous methods for this run (with `1e4` reconciled samples) below.

## Upper reconciled Forecasts

```python
**# Python results block**

mean_upp      sd_upp
4772.1497  136.045621
519.1889   45.743155
991.3990   40.570958
3261.5618  121.508043
477.1020   44.952344
42.0869    8.586463
765.9321   37.330466
225.4669   15.919683
306.7326   30.112162
551.4558   36.895534
2403.3734  111.923216
```

```r
**# R results block**

mean_upp      sd_upp
4774.3788   137.112955
516.3398    46.860373
993.4836    41.448058
3264.5554   122.146641
474.3685    46.167331
41.9713     8.459494
768.2570    38.570775
225.2266    15.589983
306.2673    29.512078
550.6324    37.037344
2407.6557   112.394686
```

```
**# Relative difference in results**

mean_upp      sd_upp
0.0004668880 0.007784335
0.0055178780 0.023841419
0.0020982732 0.021161440
0.0009170008 0.005228124
0.0057623978 0.026317037
0.0027542630 0.015009108
0.0030262009 0.032156717
0.0010669255 0.021148217
0.0015192611 0.020333504
0.0014953715 0.003828831
0.0017786181 0.004194767
```
From the above tables, it seems evident that the results in both the implementations are similar with some narrow deviations. The third table shows that the relative difference is also quite small.

## Bottom reconciled Forecasts

| **Type of absolute diff** | **mean** | **min** | **max** |
| --- | --- | --- | --- |
| **bottom mean** | 0.07381948    |  0  | 8.583333 |
| **bottom sd** | 0.4295341  | 1.661562e-05  | 1.999992 |

For the bottom reconciled forecasts we get results similar to the ones we have obtained so far. There are a few outliers, but apart from them, the results are similar. Here we see larger deviations than the other methods, but those might be due to the different randomization in the two implementations.  

## Contributors
<div style="display: flex; justify-content: center; gap: 40px;">

  <div style="text-align: center;">
    <img src="https://avatars.githubusercontent.com/u/171348088?v=4" width="200"/>
    <div><a href="mailto:anubhab.biswas@supsi.ch">anubhab.biswas@supsi.ch</a></div>
  </div>

  <div style="text-align: center;">
    <img src="https://avatars.githubusercontent.com/u/8004629?v=4" width="200"/>
    <div><a href="mailto:lorenzo.nespoli@supsi.ch">lorenzo.nespoli@supsi.ch</a></div>
  </div>

</div>