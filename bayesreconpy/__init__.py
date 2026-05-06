"""Public API for bayesreconpy.

Canonical imports are organized under `bayesreconpy.linear`, `bayesreconpy.core`,
and `bayesreconpy.utils`. Historical top-level module imports remain supported
for backward compatibility.
"""

from .base import ReconcilerBase

_LAZY_EXPORTS = {
    "reconc_gaussian": ("bayesreconpy.linear", "reconc_gaussian"),
    "reconc_buis": ("bayesreconpy.linear", "reconc_buis"),
    "reconc_mcmc": ("bayesreconpy.linear", "reconc_mcmc"),
    "reconc_mix_cond": ("bayesreconpy.linear", "reconc_mix_cond"),
    "reconc_td_cond": ("bayesreconpy.linear", "reconc_td_cond"),
    "reconc_ols": ("bayesreconpy.linear", "reconc_ols"),
    "get_S_from_A": ("bayesreconpy.linear", "get_S_from_A"),
    "reconc_mint": ("bayesreconpy.linear", "reconc_mint"),
    "estimate_cov_matrix": ("bayesreconpy.linear", "estimate_cov_matrix"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, symbol = _LAZY_EXPORTS[name]
        module = __import__(module_name, fromlist=[symbol])
        value = getattr(module, symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'bayesreconpy' has no attribute '{name}'")

__all__ = [
    "ReconcilerBase",
    "reconc_gaussian",
    "reconc_buis",
    "reconc_mcmc",
    "reconc_mix_cond",
    "reconc_td_cond",
    "reconc_ols",
    "get_S_from_A",
    "reconc_mint",
    "estimate_cov_matrix",
]
