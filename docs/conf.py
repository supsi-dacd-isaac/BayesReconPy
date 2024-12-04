# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bayesreconpy'
copyright = '2024, A. Biswas, L. Nespoli'
author = 'A. Biswas, L. Nespoli'
release = '0.1'

import pydata_sphinx_theme
import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath('../'))  # Adjust this path if needed

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/supsi-dacd-isaac/BayesReconPy",  # Optional
}

# -- Extensions --------------------------------------------------------------
extensions = [
    "pydata_sphinx_theme",          # PyData theme
    "sphinx.ext.autodoc",           # Automatically document modules/classes
    "sphinx.ext.napoleon",          # Support for Google/NumPy style docstrings
    "sphinx_autodoc_typehints",      # Show type hints in the documentation
    "nbsphinx"
]

nbsphinx_execute = 'auto'

# Exclude private members by default
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'exclude-members': '_*',  # This excludes private functions (those starting with '_')
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']