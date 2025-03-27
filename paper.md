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
    corresponding: true
    affiliation: 2
  - name: Lorenzo Zambon
    orcid: 0000-0002-8939-993X
    corresponding: true
    affiliation: 2
  - name: Nicolò Rubattu
    orcid: 0000-0002-2703-1005
    corresponding: true
    affiliation: 2
  - name: Giorgio Corani
    orcid: 0000-0002-1541-8384
    corresponding: true
    affiliation: 2

affiliations:
 - name: SUPSI, ISAAC (Istituto Sostenibilità Applicata all'Ambiente Costruito), Mendrisio, Switzerland
   index: 1
   ror: 05ep8g269
 - name: SUPSI, IDSIA (Dalle Molle Institute for Artificial Intelligence), Lugano-Viganello, Switzerland
   index: 2
   ror: 013355g38

date: 28 march 2025
bibliography: paper.bib

# Summary

`BayesReconpy` implement methods for probabilistic forecast reconciliation in Python. It reconciles hierarchies containing real-valued time series,  discrete time series and a mixture of real-valued and discrete time series (mixed hierarchies). The package is released under 

## Introduction

Time series are often structured hierarchically—for example, item-level sales can be aggregated by product, category, or store [Makridakis et al., 2022]. These aggregations define linear constraints that the data naturally satisfy (*coherent* data).

However, forecasts generated independently for each series (*base forecasts*) typically violate these constraints, leading to *incoherent* predictions. Reconciliation methods adjust base forecasts to restore coherence.

Early work focused on reconciling point forecasts [Hyndman et al., 2011; Wickramasuriya et al., 2019], while more recent research has proposed probabilistic methods that produce coherent predictive distributions [Jeon et al., 2019; Taieb et al., 2021; Panagiotelis et al., 2023; Girolimetto et al., 2024].

**bayesReconPy**, the Python adaptation of the R package **bayesRecon**, implements probabilistic reconciliation via conditioning. It supports hierarchies with:

- Gaussian base forecasts [Corani et al., 2021]  
- Continuous non-Gaussian [Zambon et al., 2024]  
- Discrete [Corani et al., 2024; Zambon et al., 2024]  
- Mixed continuous and discrete distributions [Zambon, 2024]

The package also includes projection-based reconciliation methods and their probabilistic extensions:

- OLS [Hyndman et al., 2011]  
- MinT [Wickramasuriya et al., 2019]

Documentation and tutorials are available at:  
👉 [https://bayesreconpy.readthedocs.io/en/latest/](https://bayesreconpy.readthedocs.io/en/latest/)  

Three vignettes partially reproduce the experiments in [Corani et al., 2021; Corani et al., 2024; Zambon et al., 2024] and demonstrate how to get started with the package.



