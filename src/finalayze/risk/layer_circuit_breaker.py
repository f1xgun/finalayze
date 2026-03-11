"""Per-layer circuit breakers for multi-asset portfolio (Layer 4).

Each layer has independent peak-to-trough drawdown monitoring with
layer-specific thresholds. Portfolio-level circuit breaker at 10%
triggers liquidation of Strategic + Tactical + Short (but NOT Core).

Layer risk parameters from plan:
- Core: L3 only (portfolio) -- MTM exempt, amortized cost for DD calc
- Strategic: L2 at -3%, L3 at -5%
- Tactical: L1 at -2%, L2 at -3%
- Short: L1 at -1.5%, L2 at -3%
- Portfolio: L3 at -10%

Levels:
- L1 (Caution): reduce new position sizes by 50%
- L2 (Halt): no new positions
- L3 (Liquidate): close all positions in affected layers
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum


class CircuitLevel(IntEnum):
    """Circuit breaker alert level."""

    NORMAL = 0
    CAUTION = 1  # L1: reduce sizing by 50%
    HALT = 2  # L2: no new positions
    LIQUIDATE = 3  # L3: close all positions


@dataclass(frozen=True)
class LayerCircuitConfig:
    """Circuit breaker thresholds for a single layer."""

    layer_id: str
    l1_threshold_pct: Decimal | None  # None = no L1 (e.g. Core)
    l2_threshold_pct: Decimal | None  # None = no L2
    l3_threshold_pct: Decimal | None  # None = no L3


# Default per-layer thresholds
LAYER_CIRCUIT_CONFIGS: dict[str, LayerCircuitConfig] = {
    "core": LayerCircuitConfig(
        layer_id="core",
        l1_threshold_pct=None,  # No layer-level circuit breakers
        l2_threshold_pct=None,  # Core uses amortized cost
        l3_threshold_pct=None,  # Only portfolio-level L3 affects Core
    ),
    "strategic": LayerCircuitConfig(
        layer_id="strategic",
        l1_threshold_pct=None,
        l2_threshold_pct=Decimal("0.03"),  # -3%
        l3_threshold_pct=Decimal("0.05"),  # -5%
    ),
    "tactical": LayerCircuitConfig(
        layer_id="tactical",
        l1_threshold_pct=Decimal("0.02"),  # -2%
        l2_threshold_pct=Decimal("0.03"),  # -3%
        l3_threshold_pct=None,
    ),
    "short": LayerCircuitConfig(
        layer_id="short",
        l1_threshold_pct=Decimal("0.015"),  # -1.5%
        l2_threshold_pct=Decimal("0.03"),  # -3%
        l3_threshold_pct=None,
    ),
}

# Portfolio-level threshold
PORTFOLIO_L3_THRESHOLD = Decimal("0.10")  # -10%


class LayerCircuitBreaker:
    """Monitor drawdown for a single layer and report circuit level."""

    def __init__(self, config: LayerCircuitConfig) -> None:
        self._config = config
        self._peak_equity = Decimal(0)
        self._current_equity = Decimal(0)
        self._current_level = CircuitLevel.NORMAL

    def update(self, equity: Decimal) -> CircuitLevel:
        """Update with current equity and return circuit level.

        Args:
            equity: Current layer equity.

        Returns:
            Current circuit level after update.
        """
        self._current_equity = equity

        self._peak_equity = max(self._peak_equity, equity)

        if self._peak_equity <= 0:
            self._current_level = CircuitLevel.NORMAL
            return self._current_level

        dd = (self._peak_equity - equity) / self._peak_equity

        # Check from highest severity down
        if self._config.l3_threshold_pct is not None and dd >= self._config.l3_threshold_pct:
            self._current_level = CircuitLevel.LIQUIDATE
        elif self._config.l2_threshold_pct is not None and dd >= self._config.l2_threshold_pct:
            self._current_level = CircuitLevel.HALT
        elif self._config.l1_threshold_pct is not None and dd >= self._config.l1_threshold_pct:
            self._current_level = CircuitLevel.CAUTION
        else:
            self._current_level = CircuitLevel.NORMAL

        return self._current_level

    @property
    def level(self) -> CircuitLevel:
        """Return the current circuit breaker level."""
        return self._current_level

    @property
    def drawdown_pct(self) -> Decimal:
        """Return the current peak-to-trough drawdown as a decimal fraction."""
        if self._peak_equity <= 0:
            return Decimal(0)
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def reset(self) -> None:
        """Reset peak tracking (e.g., after monthly equity redistribution)."""
        self._peak_equity = Decimal(0)
        self._current_equity = Decimal(0)
        self._current_level = CircuitLevel.NORMAL

    def sizing_multiplier(self) -> Decimal:
        """Position sizing multiplier based on circuit level.

        NORMAL: 1.0 (full sizing)
        CAUTION: 0.5 (half sizing)
        HALT/LIQUIDATE: 0.0 (no new positions)
        """
        if self._current_level == CircuitLevel.NORMAL:
            return Decimal(1)
        if self._current_level == CircuitLevel.CAUTION:
            return Decimal("0.5")
        return Decimal(0)


class PortfolioCircuitBreaker:
    """Monitor combined portfolio drawdown.

    At -10% DD: liquidate Strategic + Tactical + Short.
    Core is NOT force-liquidated (amortized cost accounting, committed cash flows).
    """

    def __init__(self, threshold: Decimal = PORTFOLIO_L3_THRESHOLD) -> None:
        self._threshold = threshold
        self._peak_equity = Decimal(0)
        self._triggered = False

    def update(self, total_equity: Decimal) -> bool:
        """Update and return True if portfolio L3 triggered."""
        self._peak_equity = max(self._peak_equity, total_equity)

        if self._peak_equity <= 0:
            return False

        dd = (self._peak_equity - total_equity) / self._peak_equity
        if dd >= self._threshold:
            self._triggered = True
        return self._triggered

    @property
    def is_triggered(self) -> bool:
        """Return whether the portfolio-level circuit breaker has been triggered."""
        return self._triggered

    @property
    def layers_to_liquidate(self) -> list[str]:
        """Layers that should be liquidated on portfolio L3."""
        if self._triggered:
            return ["strategic", "tactical", "short"]
        return []

    def reset(self) -> None:
        """Reset portfolio circuit breaker state."""
        self._peak_equity = Decimal(0)
        self._triggered = False
