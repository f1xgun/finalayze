"""Journaling strategy combiner — records per-strategy signals via hooks.

Overrides StrategyCombiner's hook methods to capture per-strategy signals,
weights, features, and ML model probabilities without duplicating the
generate_signal() loop.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finalayze.strategies.combiner import StrategyCombiner

if TYPE_CHECKING:
    from decimal import Decimal

    from finalayze.core.schemas import Signal
    from finalayze.strategies.base import BaseStrategy


class JournalingStrategyCombiner(StrategyCombiner):
    """Records per-strategy signals before combining them.

    After generate_signal() is called, the last_signals and last_weights
    dicts are populated for the backtest engine to read.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        normalize_mode: str = "firing",
        allocation_mode: str = "static",
    ) -> None:
        super().__init__(strategies, normalize_mode, allocation_mode)
        self._last_signals: dict[str, Signal | None] = {}
        self._last_weights: dict[str, Decimal] = {}
        self._last_net_score: float | None = None
        self._last_features: dict[str, float] = {}
        self._last_model_probas: dict[str, float] | None = None
        self._last_segment_id: str | None = None

    @property
    def last_signals(self) -> dict[str, Signal | None]:
        """Per-strategy signals from the most recent generate_signal() call."""
        return dict(self._last_signals)

    @property
    def last_weights(self) -> dict[str, Decimal]:
        """Per-strategy weights from the most recent generate_signal() call."""
        return dict(self._last_weights)

    @property
    def last_net_score(self) -> float | None:
        """Net weighted score from the most recent generate_signal() call."""
        return self._last_net_score

    @property
    def last_features(self) -> dict[str, float]:
        """Aggregated features from all strategy signals, prefixed by strategy name."""
        return dict(self._last_features)

    @property
    def last_model_probas(self) -> dict[str, float] | None:
        """Per-model probabilities from MLStrategy's EnsembleModel, if present."""
        return dict(self._last_model_probas) if self._last_model_probas is not None else None

    # ── Hook overrides ──────────────────────────────────────────────────

    def _on_generate_start(self, symbol: str, segment_id: str) -> None:  # noqa: ARG002
        """Reset tracking state at the start of each generate_signal() call."""
        self._last_signals = {}
        self._last_weights = {}
        self._last_net_score = None
        self._last_features = {}
        self._last_model_probas = None
        self._last_segment_id = segment_id

    def _on_strategy_signal(
        self,
        name: str,
        strategy: BaseStrategy,
        signal: Signal | None,
        weight: Decimal,
    ) -> None:
        """Record per-strategy signal, weight, features, and ML probas."""
        self._last_signals[name] = signal
        self._last_weights[name] = weight

        if signal is None:
            return

        # Aggregate per-strategy features prefixed by strategy name
        for feat_key, feat_val in signal.features.items():
            self._last_features[f"{name}.{feat_key}"] = feat_val

        # Capture per-model probas from MLStrategy's EnsembleModel
        if hasattr(strategy, "_registry"):
            registry = strategy._registry
            seg_id = getattr(self, "_last_segment_id", None)
            ensemble = getattr(registry, "get", lambda _s: None)(seg_id) if seg_id else None
            if ensemble is not None and hasattr(ensemble, "last_model_probas"):
                probas = ensemble.last_model_probas
                if probas:
                    self._last_model_probas = dict(probas)

    def _on_normalized(self, net: float, features: dict[str, float]) -> None:  # noqa: ARG002
        """Record the net score after normalization."""
        self._last_net_score = net

    def _on_final_signal(
        self,
        signal: Signal | None,
        contributions: dict[str, float],
    ) -> None:
        """No-op — all journaling state is already captured by other hooks."""
