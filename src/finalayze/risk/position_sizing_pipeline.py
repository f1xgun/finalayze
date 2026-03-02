"""Unified position sizing pipeline (Layer 4).

Applies a chain of sizing adjustments: Kelly -> VolTarget -> Regime -> HardCaps.
All calculations use Decimal for financial precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol

_VOL_TARGET_LOWER = Decimal("0.25")
_VOL_TARGET_UPPER = Decimal("2.0")
_REGIME_FLOOR = Decimal("0.10")
_FOUR_DP = Decimal("0.0001")


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
        regime_scale: Regime-based position scale (0.10 to 1.0).
        correlation_scale: Correlation-based scale (0.30 to 1.0).
    """

    equity: Decimal
    base_position: Decimal
    max_position_pct: Decimal
    min_position_size: Decimal
    asset_vol: Decimal
    target_vol: Decimal
    regime_scale: Decimal
    correlation_scale: Decimal


class PositionSizingStep(Protocol):
    """Protocol for a single step in the sizing pipeline."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal: ...


class KellyStep:
    """Pass-through step: base_position already includes Kelly sizing."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:  # noqa: ARG002
        """Kelly is already factored into base_position; return as-is."""
        return size


class VolTargetStep:
    """Scale position by target_vol / asset_vol, bounded [0.25x, 2.0x]."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if context.asset_vol <= 0:
            return size
        raw_ratio = context.target_vol / context.asset_vol
        clamped = max(_VOL_TARGET_LOWER, min(raw_ratio, _VOL_TARGET_UPPER))
        return (size * clamped).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


class RegimeStep:
    """Scale position by regime_scale with a floor of 0.10."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        scale = max(context.regime_scale, _REGIME_FLOOR)
        return (size * scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)


class HardCapsStep:
    """Enforce max position cap (equity * max_position_pct)."""

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        max_cap = context.equity * context.max_position_pct
        return min(size, max_cap)


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
        """
        size = context.base_position
        for step in self._steps:
            size = step.adjust(size, context)
        if size < context.min_position_size:
            return Decimal(0)
        return min(size, context.equity * context.max_position_pct)
