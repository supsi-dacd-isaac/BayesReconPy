"""Linear reconciliation methods.

This package provides canonical import locations for linear and conditioning
reconciliation algorithms while preserving backward compatibility with
historical top-level modules.
"""

from .gaussian import reconc_gaussian
from .projection import estimate_cov_matrix, get_S_from_A, reconc_mint, reconc_ols
from .conditioning import reconc_buis, reconc_mcmc, reconc_mix_cond, reconc_td_cond

__all__ = [
    "reconc_gaussian",
    "get_S_from_A",
    "reconc_ols",
    "estimate_cov_matrix",
    "reconc_mint",
    "reconc_buis",
    "reconc_mcmc",
    "reconc_mix_cond",
    "reconc_td_cond",
]
