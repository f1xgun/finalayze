"""Unified position sizing pipeline (Layer 4).

Applies a chain of sizing adjustments: Kelly -> VolTarget -> Regime -> HardCaps.
All calculations use Decimal for financial precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Protocol

from finalayze.risk.evt import EVTRiskEstimator

if TYPE_CHECKING:
    from finalayze.risk.rub_oil_regime import RubOilRegimeSignal

_VOL_TARGET_LOWER = Decimal("0.25")
_VOL_TARGET_UPPER = Decimal("1.5")
_REGIME_FLOOR = Decimal("0.15")
_FOUR_DP = Decimal("0.0001")

_EVT_SCALE_FACTOR = Decimal("0.5")
_EVT_MIN_HISTORY = 100
_EVT_RECENT_WINDOW = 60


@dataclass(frozen=True, slots=True)
class SizingContext:
    """Input context for the position sizing pipeline.

    Attributes:
        equity: Total portfolio equity.
        base_position: Initial position size (e.g. from Kelly or fixed fraction).
        max_position_pct: Maximum single-position size as fraction of equity.
        min_position_size: Minimum viable position (0.5% of equity or $500).
        asset_vol: Annualized volatility of the asset.
        target_vol: Target portfolio volatility.
        regime_scale: Regime-based position scale (floored at 0.15 by RegimeStep).
        correlation_scale: Correlation-based scale (0.30 to 1.0).
        returns_history: Historical portfolio returns for EVT step.
        ml_confidence: P(profitable) from MetaLabeler [0, 1], or None if ML unavailable.
    """

    equity: Decimal
    base_position: Decimal
    max_position_pct: Decimal
    min_position_size: Decimal
    asset_vol: Decimal
    target_vol: Decimal
    regime_scale: Decimal
    correlation_scale: Decimal
    returns_history: tuple[float, ...] = ()
    ml_confidence: float | None = None  # P(profitable) from MetaLabeler [0, 1]


class PositionSizingStep(Protocol):
    """Protocol for a single step in the sizing pipeline."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal: ...


class KellyStep:
    """Pass-through step: base_position already includes Kelly sizing."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:  # noqa: ARG002
        """Kelly is already factored into base_position; return as-is."""
        return size


class VolTargetStep:
    """Scale position by target_vol / asset_vol, bounded [0.25x, 1.5x]."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if context.asset_vol <= 0:
            return size
        raw_ratio = context.target_vol / context.asset_vol
        clamped = max(_VOL_TARGET_LOWER, min(raw_ratio, _VOL_TARGET_UPPER))
        return (size * clamped).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


class RegimeStep:
    """Scale position by regime_scale with a floor of 0.15."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        scale = max(context.regime_scale, _REGIME_FLOOR)
        return (size * scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


class HardCapsStep:
    """Enforce max position cap (equity * max_position_pct)."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        max_cap = context.equity * context.max_position_pct
        return min(size, max_cap)


class CopulaStep:
    """Scale position by the correlation_scale derived from copula fits."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        return (size * context.correlation_scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


class EVTStep:
    """Scale position down when EVT tail risk is elevated."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if len(context.returns_history) < _EVT_MIN_HISTORY:
            return size
        recent_returns = context.returns_history[-_EVT_RECENT_WINDOW:]
        current_loss = abs(min(recent_returns))
        estimator = EVTRiskEstimator()
        if estimator.is_tail_risk_elevated(
            list(context.returns_history),
            current_loss=current_loss,
            confidence=0.99,
        ):
            return (size * _EVT_SCALE_FACTOR).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
        return size


_BRENT_RUB_THRESHOLD = 5000.0
_BRENT_GATE_SCALE = Decimal("0.5")


class RubOilRegimeStep:
    """Scale positions by RUB/oil decorrelation regime for ru_* segments.

    NORMAL (corr > 0.3) -> 1.0x, ELEVATED (0.1-0.3) -> 0.5x, CRISIS (< 0.1) -> 0.25x.
    Non-ru_* segments pass through unchanged.
    """

    def __init__(self, regime_signal: RubOilRegimeSignal, segment_id: str) -> None:
        self._regime_signal = regime_signal
        self._segment_id = segment_id

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:  # noqa: ARG002
        if not self._segment_id.startswith("ru_"):
            return size
        state = self._regime_signal.get_regime([], 0)
        return (size * state.position_scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


class BrentGateStep:
    """Gate energy sector positions when Brent-in-RUB below threshold.

    When Brent-in-RUB < 5000 RUB/bbl, scale energy positions by 0.5.
    Only applies to ru_energy segment. Non-energy and missing data pass through.
    """

    def __init__(
        self,
        brent_rub_price: float,
        segment_id: str,
        threshold: float = _BRENT_RUB_THRESHOLD,
        scale_below: Decimal = _BRENT_GATE_SCALE,
    ) -> None:
        self._brent_rub = brent_rub_price
        self._segment_id = segment_id
        self._threshold = threshold
        self._scale_below = scale_below

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:  # noqa: ARG002
        if self._segment_id != "ru_energy":
            return size
        if self._brent_rub <= 0:
            return size  # graceful degradation: missing data -> no gate
        if self._brent_rub >= self._threshold:
            return size
        return (size * self._scale_below).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


_META_LABEL_THRESHOLD = Decimal("0.40")


class MetaLabelStep:
    """Scale position by ML-predicted P(profitable).

    Maps [threshold, 1.0] -> [0.0, 1.0] linearly.
    Below threshold -> zero position (ML vetoes the trade).
    None -> pass-through (ML not available).
    """

    def __init__(self, threshold: Decimal = _META_LABEL_THRESHOLD) -> None:
        self._threshold = threshold

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if context.ml_confidence is None:
            return size  # No ML data -> pass through
        confidence = Decimal(str(context.ml_confidence))
        if confidence <= self._threshold:
            return Decimal(0)  # ML vetoes trade
        scaling_range = Decimal(1) - self._threshold
        factor = (confidence - self._threshold) / scaling_range
        return (size * factor).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


class PositionSizingPipeline:
    """Ordered pipeline of position sizing adjustments.

    Default step order: KellyStep -> VolTargetStep -> RegimeStep -> HardCapsStep.
    After all steps, positions below min_position_size are eliminated (return 0).
    """

    def __init__(self, steps: list[PositionSizingStep] | None = None) -> None:
        self._steps: list[PositionSizingStep] = steps or [
            KellyStep(),
            VolTargetStep(),
            RegimeStep(),
            HardCapsStep(),
        ]

    @property
    def steps(self) -> list[PositionSizingStep]:
        """Return the ordered list of sizing steps."""
        return list(self._steps)

    def compute(self, context: SizingContext) -> Decimal:
        """Run the pipeline and return the final position size.

        Returns Decimal(0) if the result is below min_position_size.
        RegimeStep enforces a 15% floor internally, so no pipeline-level floor is needed.
        """
        size = context.base_position
        for step in self._steps:
            size = step.adjust(size, context)
        # Guarded round-up: only if Kelly base was viable (positive expectancy)
        if context.base_position > context.min_position_size:
            half_min = context.min_position_size * Decimal("0.5")
            if half_min <= size < context.min_position_size:
                size = context.min_position_size
        if size < context.min_position_size:
            return Decimal(0)
        return min(size, context.equity * context.max_position_pct)
