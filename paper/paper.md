---
title: 'BayesReconPy: A Python package for forecast reconciliation'
tags:
  - Python
  - forecasting
  - time series
  - hierarchical forecasting
  - forecast reconciliation
authors:
  - name: Anubhab Biswas
    orcid: 0009-0001-2780-7555
    equal-contrib: true
    affiliation: 1
  - name: Lorenzo Nespoli
    orcid: 0000-0001-9710-8517
    equal-contrib: true 
    affiliation: 1
  - name: Dario Azzimonti
    orcid: 0000-0001-5080-3061
    corresponding: false
    affiliation: 2
  - name: Lorenzo Zambon
    orcid: 0000-0002-8939-993X
    corresponding: false
    affiliation: 2
  - name: Nicolò Rubattu
    orcid: 0000-0002-2703-1005
    corresponding: false
    affiliation: 2
  - name: Giorgio Corani
    orcid: 0000-0002-1541-8384
    corresponding: false
    affiliation: 2

affiliations:
 - name: SUPSI, ISAAC (Istituto Sostenibilità Applicata all'Ambiente Costruito), Mendrisio, Switzerland
   index: 1
   ror: 05ep8g269
 - name: SUPSI, IDSIA (Dalle Molle Institute for Artificial Intelligence), Lugano-Viganello, Switzerland
   index: 2
   ror: 013355g38

date: 2 April 2025
bibliography: paper.bib
---

# Summary

`BayesReconpy` implements methods for probabilistic forecast reconciliation in Python. It reconciles hierarchies containing real-valued time series,  discrete time series and a mixture of real-valued and discrete time series (mixed hierarchies). The package is released under the LGPL (≥ 3) license and available on [GitHub](https://github.com/supsi-dacd-isaac/BayesReconPy).

# Statement of Need

Forecast reconciliation is essential for ensuring coherence across hierarchical time series, where aggregate values must match the sum of their components. A common example is retail sales: item-level sales can be aggregated by product, category, or store [@MAKRIDAKIS20221325]. While the observed data are coherent by construction, independently generated forecasts (base forecasts) typically violate aggregation constraints, resulting in incoherent predictions.

Early work addressed this issue for point forecasts using projection-based methods such as OLS and MinT [@Hyndman_Ahmed_Athanasopoulos_Shang_2011; @Wickramasuriya_Athanasopoulos_Hyndman_2019]. More recently, probabilistic reconciliation methods have been proposed to reconcile entire forecast distributions [@Jeon_Panagiotelis_Petropoulos_2019; @taieb2021hierarchical; @Panagiotelis_Gamakumara_Athanasopoulos_Hyndman_2023; @Girolimetto_Athanasopoulos_DiFonzo_Hyndman_2024], offering a richer and more informative forecasting framework.

Despite this progress, most existing software tools for reconciliation have significant limitations (**Table 1**). Many only support Gaussian or continuous distributions, lack support for discrete or mixed-type forecasts, or are implemented only in R. Others, such as ProbReco [@Panagiotelis_Gamakumara_Athanasopoulos_Hyndman_2023] and DiscreteRecon [@ZHANG2024143], are no longer actively maintained. Notably, no existing Python library offers a unified interface for probabilistic reconciliation across distribution types.

`bayesReconPy` fills this gap. It is a Python package that implements probabilistic reconciliation via both conditioning and projection-based methods, including OLS and MinT. It supports a broad range of forecast types:

- Gaussian forecasts [@Corani_Azzimonti_Augusto_Zaffalon_2021]
- Continuous non-Gaussian forecasts [@Zambon_Azzimonti_Corani_2024]
- Discrete forecasts [@CoraniAzzimontiRubattu2024; @Zambon_Azzimonti_Corani_2024]
- Mixed discrete-continuous hierarchies [@zambon2024probabilistic]

`bayesReconPy` is an extension of the R package `bayesRecon` [@Azzimonti_Rubattu_Zambon_Corani_bayesRecon], offering the same functionality in a Python-native interface. As shown in a comparison with other reconciliation libraries, it is the only actively maintained Python package that enables probabilistic reconciliation across all major distribution types. It includes comprehensive documentation and Python notebooks that replicate key experiments from the literature [@Corani_Azzimonti_Augusto_Zaffalon_2021; @CoraniAzzimontiRubattu2024; @Zambon_Azzimonti_Corani_2024], making it accessible for researchers and practitioners alike.

### Table 1: Probabilistic reconciliation methods comparison

| Library                                                  | Cross-temp |     Gaussian     | Continuous (non-Gaussian) |    Discrete     |      Mixed      |
|----------------------------------------------------------|:----------:|:----------------:|:--------------------------:|:---------------:|:---------------:|
| **bayesReconPy (Ours)**                                  |     ❌      |       ✅         |            ✅             |       ✅        |       ✅        |
| fable / fabletools [@fable_O_Hara_Wild_etal2024]         |     ✅      |       ✅         |            ✅             |       ❌        |       ❌        |
| FoReco [@FoReco]                                         |     ✅      |       ✅         |            ✅             |       ❌        |       ❌        |
| gluonts [@gluonTS_Alexandrov_etal2020]                   |     ❌      |       ✅         |            ✅             |       ❌        |       ❌        |
| hierarchicalforecast [@olivares2022hierarchicalforecast] |     ❌      |       ✅         |            ✅             |       ❌        |       ❌        |
| thief [@thief_HyndmanKourentzes2018]                     |     ✅      |       ✅         |            ❌             |       ❌        |       ❌        |

# Usage

A hierarchy can contain, at each level, Gaussian, continuous non-Gaussian, or discrete forecast distributions (see **Figure 1**). Continuous base forecasts can be provided either in parametric form or as samples. Discrete base forecasts can be provided in parametric form, as samples, or as a probability mass function (PMF).

<div align="center">

<img src="../pictures/types_of_reconciliation.png" alt="Types of reconciliation" width="600"/>

**Figure 1:** Types of reconciliation — Gaussian, discrete, and mixed input forecasts, and output forms (parametric, PMF, samples).

</div>

Below, we describe the suitable reconciliation algorithms for each case:

1. **All forecast distributions are Gaussian**  
   Reconciliation is computed analytically using the function `reconc_gaussian`, following the approach in [@Corani_Azzimonti_Augusto_Zaffalon_2021; @ZAMBON20241438]. This is equivalent to minimum trace reconciliation (MinT) [@Wickramasuriya_Athanasopoulos_Hyndman_2019].

2. **All forecast distributions are continuous (non-Gaussian) or all are discrete**  
   Reconciliation is performed using Bottom-Up Importance Sampling (BUIS), via the function `reconc_BUIS` [@Zambon_Azzimonti_Corani_2024].

3. **Mixed distributions: discrete forecasts at the bottom level and Gaussian at the upper levels**  
   These hierarchies can be reconciled using either Mixed Conditioning (`reconc_MixCond`) or Top-Down Conditioning (`reconc_TDcond`) [@zambon2024probabilistic].

Depending on the reconciliation function used, the output is returned as:

- *Reconciled parameters* (for the Gaussian case, using `reconc_gaussian`)  
- *Samples* (for continuous non-Gaussian or discrete forecasts, using `reconc_BUIS`)  
- *Probability Mass Functions (PMFs)* (for mixed cases, using `reconc_MixCond` or `reconc_TDcond`)

Note that in the case of MinT or OLS reconciliation, the input is expected to be numpy arrays and the same is returned as the reconciled forecast. Documentation for the expected shape of these arrays is provided in the function descriptions.

## Examples

### Reconciliation of Negative Binomial Forecasts

We demonstrate the use of `bayesReconPy` on a hierarchy of extreme market events in five economic sectors over the period 2005–2018. This hierarchy consists of five bottom-level and one top-level time series (the total). These are count-valued time series, and the predictive distributions are modeled as negative binomial.

The dataset `extr_mkt_events`, included in the package, contains both the observed time series and the corresponding base forecasts. It was used in the experiments of [@ZAMBON20241438]. A related Python notebook reproducing the results is available: [Properties of the Reconciled Distribution via Conditioning](https://github.com/supsi-dacd-isaac/BayesReconPy/blob/main/notebooks/Properties%20of%20the%20reconciled%20distribution%20via%20conditioning.ipynb)

The code below shows how to apply the `reconc_BUIS` function. The function takes the summing matrix `A`, the base forecast parameters, and the desired number of samples. Reconciliation is completed within seconds:

```python
# Reconcile via importance sampling
buis = reconc_BUIS(A, base_fc_j, 
        "params", "nbinom",
        num_samples=N_samples, seed=42)
samples_y = buis['reconciled_samples']
# Computational time for 3508 reconciliations: 20.13 seconds
```

### Reconciliation of a Large Mixed Hierarchy

The M5 forecasting competition dataset [@MAKRIDAKIS20221325] includes daily sales time series for 10 stores. Each store contains 3049 bottom-level series and 11 upper-level series.

During the competition, existing reconciliation methods failed to process the hierarchy due to the dataset’s size and the requirement for non-negative forecasts [@MAKRIDAKIS20221325]. `bayesReconPy` successfully reconciles 1-step-ahead base forecasts for one store, `"CA_1"`, returning non-negative and probabilistic forecasts.

Base forecasts are included in the package and were generated using the ADAM method [@svetunkov2023iets], implemented in the `smooth` R package [@smooth_pkg]. A Python notebook illustrates this example, available as: [Reconciliation of M5 hierarchy with mixed-type forecasts](https://github.com/supsi-dacd-isaac/BayesReconPy/blob/main/notebooks/Reconciliation%20of%20M5%20hierarchy%20with%20mixed-type%20forecasts.ipynb)

Below is a code snippet demonstrating reconciliation using `reconc_TDcond`. Here, the bottom-level forecasts are discrete, and the upper-level forecasts are continuous. The reconciliation completes in just a few seconds:

```python
N_samples_TD = int(1e4)

# TDCond reconciliation
start = time.time()
td = reconc_TDcond(
    A,
    fc_bottom_4rec, 
    fc_upper_4rec, 
    bottom_in_type="pmf",
    num_samples=N_samples_TD, 
    return_type="pmf", 
    seed=seed
)
stop = time.time()

# Reconciliation time: 10.11 s
```


# Acknowledgements

Research funded by the Swiss National Science Foundation (grant 200021_212164), the Hasler Foundation (project: *hierarchical forecasting with mixed hierarchies*), the European Union (project: 101160720 — ENERGENIUS), and the Swiss State Secretariat for Education, Research and Innovation (SERI) in the context of the Horizon Europe research and innovation programme project DR-RISE (Grant Agreement No 101104154).

# References