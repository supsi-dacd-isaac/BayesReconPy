"""Canonical covariance utilities module."""

from bayesreconpy.shrink_cov import _cov2cor, _schafer_strimmer_cov
import numpy as np

__all__ = ["_cov2cor", "_schafer_strimmer_cov"]

def _to_precision(cov, eps=1e-6):
    cov = 0.5 * (cov + cov.T)
    lam = eps * np.trace(cov) / cov.shape[0]
    return np.linalg.pinv(cov + lam * np.eye(cov.shape[0]))
