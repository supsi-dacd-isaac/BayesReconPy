"""Typed bridge to legacy utilities implementation.

This module loads the historical ``bayesreconpy/utils.py`` file and exposes
its symbols with explicit names so IDEs can resolve imports from the new
``bayesreconpy.utils`` package modules.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_LEGACY_UTILS_PATH = Path(__file__).resolve().parent / "utils.py"
_LEGACY_SPEC = spec_from_file_location("bayesreconpy._legacy_utils_impl", _LEGACY_UTILS_PATH)
_LEGACY_MODULE = module_from_spec(_LEGACY_SPEC)
assert _LEGACY_SPEC is not None and _LEGACY_SPEC.loader is not None
_LEGACY_SPEC.loader.exec_module(_LEGACY_MODULE)

TOL = _LEGACY_MODULE.TOL
RTOL = _LEGACY_MODULE.RTOL
ALPHA_SMOOTHING = _LEGACY_MODULE.ALPHA_SMOOTHING
ALPHA = _LEGACY_MODULE.ALPHA
LAP_SMOOTHING = _LEGACY_MODULE.LAP_SMOOTHING
DISTR_TYPES = _LEGACY_MODULE.DISTR_TYPES
DISCR_DISTR = _LEGACY_MODULE.DISCR_DISTR
CONT_DISTR = _LEGACY_MODULE.CONT_DISTR
DEFAULT_PARS = _LEGACY_MODULE.DEFAULT_PARS

_check_S = _LEGACY_MODULE._check_S
_check_A = _LEGACY_MODULE._check_A
_check_cov = _LEGACY_MODULE._check_cov
_check_real_number = _LEGACY_MODULE._check_real_number
_check_positive_number = _LEGACY_MODULE._check_positive_number
_check_implemented_distr = _LEGACY_MODULE._check_implemented_distr
_check_distr_params = _LEGACY_MODULE._check_distr_params
_check_discrete_samples = _LEGACY_MODULE._check_discrete_samples
_check_input_BUIS = _LEGACY_MODULE._check_input_BUIS
_check_input_TD = _LEGACY_MODULE._check_input_TD
_check_weights = _LEGACY_MODULE._check_weights
_distr_sample = _LEGACY_MODULE._distr_sample
_MVN_sample = _LEGACY_MODULE._MVN_sample
_MVN_density = _LEGACY_MODULE._MVN_density
_resample = _LEGACY_MODULE._resample
_distr_pmf = _LEGACY_MODULE._distr_pmf
_shape = _LEGACY_MODULE._shape
_gen_gaussian = _LEGACY_MODULE._gen_gaussian
_gen_poisson = _LEGACY_MODULE._gen_poisson
_samples_from_pmf = _LEGACY_MODULE._samples_from_pmf

__all__ = [name for name in globals() if not name.startswith("_LEGACY_")]
