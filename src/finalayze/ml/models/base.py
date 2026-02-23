"""Abstract ML model base class (Layer 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMLModel(ABC):
    """Abstract base for all per-segment ML models."""

    segment_id: str

    @abstractmethod
    def predict_proba(self, features: dict[str, float]) -> float:
        """Return BUY probability in [0.0, 1.0]."""
        ...

    @abstractmethod
    def fit(self, X: list[dict[str, float]], y: list[int]) -> None:  # noqa: N803
        """Train on feature dicts (X) and binary labels (y: 1=BUY, 0=SELL/HOLD)."""
        ...
