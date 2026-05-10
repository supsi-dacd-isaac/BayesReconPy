.. bayesreconpy documentation master file, created by
   sphinx-quickstart on Fri Nov 22 14:00:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

    <img src="https://raw.githubusercontent.com/supsi-dacd-isaac/BayesReconPy/main/pictures/BAyesreCONpy.png"
         alt="Forecast example at M5"
         width="500"
         style="display: block; margin: auto;">


Welcome to bayesreconpy's documentation!
========================================

.. image:: https://codecov.io/github/supsi-dacd-isaac/BayesReconPy/graph/badge.svg?token=KND9ZJ4GOQ
   :target: https://codecov.io/github/supsi-dacd-isaac/BayesReconPy
   :alt: Codecov coverage badge

.. image:: https://img.shields.io/badge/license-LGPL%20(%3E=%203)-yellow.svg
   :target: https://www.gnu.org/licences/lgpl-3.0
   :alt: License: LGPL (\>= 3)

.. image:: https://img.shields.io/pypi/v/bayesreconpy
   :target: https://pypi.org/project/bayesreconpy/
   :alt: PyPI

.. image:: https://img.shields.io/badge/python-3.12%2B-green
   :alt: Python 3.10+

.. image:: https://static.pepy.tech/badge/bayesreconpy
   :target: https://pepy.tech/project/bayesreconpy
   :alt: Total downloads

Bayesian Reconciliation for Hierarchical and Constrained Forecasting
====================================================================

Forecast reconciliation ensures that probabilistic forecasts across hierarchical, grouped, or constrained time series remain coherent. In this context, coherence means that forecasts at disaggregated levels, such as local components, are consistent with forecasts at aggregated levels, such as system totals. Forecast reconciliation is a post-processing technique applied to independently generated base forecasts that may not satisfy the relevant hierarchical or structural constraints.

While several methods exist for point forecast reconciliation, ``BayesReconPy`` focuses on probabilistic reconciliation for different types of hierarchical and grouped time series in Python. Many existing tools for forecast reconciliation are limited to Gaussian or continuous inputs, lack support for discrete or mixed-type forecasts, or are available only in R. ``BayesReconPy`` supports the reconciliation of discrete and non-Gaussian forecast distributions using Bayesian forecast reconciliation via conditioning methods. These methods are relevant in several application domains, including energy systems, demand forecasting, and risk analysis.

Installation
------------

The Python package is available on PyPI and can be installed with:

.. code-block:: bash

   pip install bayesreconpy

Documentation
-------------

The documentation of the reconciliation functions is available at:

`BayesReconPy documentation <https://bayesreconpy.readthedocs.io/en/latest/>`_

Citation
--------

Please cite the following JOSS paper when using this package in your research:

   Biswas et al. (2025). `BayesReconPy: A Python package for forecast reconciliation <https://doi.org/10.21105/joss.08336>`_. *Journal of Open Source Software*, 10(111), 8336. https://doi.org/10.21105/joss.08336

The code for linear forecast reconciliation is an implementation of the
`original R package <https://cran.r-project.org/web/packages/bayesRecon/index.html>`_.
A comparison of the results obtained from the R and Python versions can be found in the old README file of this package.

Extension to Nonlinear Reconciliation
-------------------------------------

From version ``0.5.0``, ``BayesReconPy`` also includes algorithms for reconciling time series with nonlinear constraints. Bayesian reconciliation methods based on conditioning are available in the new nonlinear module of the package.

The original functions for linear forecast reconciliation can also be called from the new linear module of the package.

An Important Note on Projection-Based Approaches
------------------------------------------------

Although the main goal of this package is to perform probabilistic forecast reconciliation using Bayesian approaches, projection-based approaches are also included for completeness. This allows users to compare different reconciliation methods across both linear and nonlinear settings.

The github page for this python implementation can be found `here <https://github.com/supsi-dacd-isaac/BayesReconPy>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   functions
   linear
   nonlinear


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
