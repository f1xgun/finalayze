"""Per-segment ML model registry (Layer 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.ml.models.ensemble import EnsembleModel


class MLModelRegistry:
    """Maps segment IDs to trained EnsembleModel instances."""

    def __init__(self) -> None:
        self._models: dict[str, EnsembleModel] = {}

    def register(self, segment_id: str, model: EnsembleModel) -> None:
        """Register or replace a model for a segment."""
        self._models[segment_id] = model

    def get(self, segment_id: str) -> EnsembleModel | None:
        """Return the model for the segment, or None if not registered."""
        return self._models.get(segment_id)
