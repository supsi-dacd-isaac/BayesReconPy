"""Shared base interfaces for reconcilers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ReconcilerBase(ABC):
    """Abstract base class for reconciliation model implementations."""

    @abstractmethod
    def reconcile(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run reconciliation and return structured outputs."""
        raise NotImplementedError
