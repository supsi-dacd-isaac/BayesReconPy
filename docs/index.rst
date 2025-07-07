.. bayesreconpy documentation master file, created by
   sphinx-quickstart on Fri Nov 22 14:00:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bayesreconpy's documentation!
========================================

.. image:: https://codecov.io/github/supsi-dacd-isaac/BayesReconPy/graph/badge.svg?token=KND9ZJ4GOQ
   :target: https://codecov.io/github/supsi-dacd-isaac/BayesReconPy
   :alt: Codecov coverage badge

.. image:: https://img.shields.io/badge/license-LGPL%20(%3E=%203)-yellow.svg
   :target: https://www.gnu.org/licences/lgpl-3.0
   :alt: License: LGPL (\>= 3)


Bayesian Reconciliation for Hierarchical Forecasting
----------------------------------------------------

Forecast reconciliation ensures that probabilistic forecasts across hierarchical or grouped time series remain coherent, meaning that forecasts at disaggregated levels (e.g., local components) are consistent with those at aggregated levels (e.g., system totals). It is a post-processing technique applied to a set of independently generated incoherent base forecasts that do not obey these hierarchical constraints.

.. raw:: html

    <img src="https://raw.githubusercontent.com/supsi-dacd-isaac/BayesReconPy/main/pictures/M5.png"
         alt="Forecast example at M5"
         width="600"
         style="display: block; margin: auto;">

While several methods exist for point forecast reconciliation, BayesReconPy focuses on probabilistic reconciliation of different kinds of hierarchical and grouped time series in python. Most of the existing tools in forecast reconciliation are limited to Gaussian or continuous inputs, lack support for discrete or mixed-type forecasts, or are implemented only in R. This package supports reconciliation of discrete and non-Gaussian forecast distributions using Bayesian forecast reconciliation via conditioning methods, which are common in domains such as energy systems, demand forecasting, and risk analysis.

This code is an python implementation of the `original R package <https://cran.r-project.org/web/packages/bayesRecon/index.html>`_

The python package is available in pip, use the following line of command in your terminal for the installation:

.. code-block:: bash

   pip install bayesreconpy

The github page for this python implementation can be found `here <https://github.com/supsi-dacd-isaac/BayesReconPy>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   functions
   examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
