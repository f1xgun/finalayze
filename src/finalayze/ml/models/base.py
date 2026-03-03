"""Abstract ML model base class (Layer 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BaseMLModel(ABC):
    """Abstract base for all per-segment ML models."""

    segment_id: str

    @abstractmethod
    def predict_proba(self, features: dict[str, float]) -> float:
        """Return BUY probability in [0.0, 1.0]."""
        ...

    @abstractmethod
    def fit(
        self,
        X: list[dict[str, float]],  # noqa: N803
        y: list[int],
        *,
        sample_weight: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> None:
        """Train on feature dicts (X) and binary labels (y: 1=BUY, 0=SELL/HOLD).

        Args:
            X: Feature dictionaries.
            y: Binary labels.
            sample_weight: Optional per-sample weights for importance weighting.
        """
        ...
