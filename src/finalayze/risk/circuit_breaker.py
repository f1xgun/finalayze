"""Per-market and cross-market circuit breakers (Layer 4).

Drawdown thresholds gate position sizing and trading activity:
  L1 (CAUTION)   >= 5%  drawdown: halve size, raise min confidence
  L2 (HALTED)    >= 10% drawdown: no new trades
  L3 (LIQUIDATE) >= 15% drawdown: close all positions immediately

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

# ── Threshold defaults ──────────────────────────────────────────────────────
_DEFAULT_L1 = 0.05
_DEFAULT_L2 = 0.10
_DEFAULT_L3 = 0.15
_DEFAULT_CROSS_HALT = 0.10
_ZERO = Decimal(0)


class CircuitLevel(StrEnum):
    """Escalating circuit breaker states."""

    NORMAL = "normal"  # trade freely
    CAUTION = "caution"  # -5% daily: halve size, raise min confidence
    HALTED = "halted"  # -10% daily: no new trades
    LIQUIDATE = "liquidate"  # -15% daily: close all positions immediately


class CircuitBreaker:
    """Per-market circuit breaker that escalates level based on daily drawdown.

    Drawdown = (baseline_equity - current_equity) / baseline_equity.

    Reset rules:
        - ``reset_daily``: resets CAUTION / HALTED -> NORMAL; updates baseline.
          LIQUIDATE is NOT cleared by daily reset -- it requires operator action.
        - ``reset_manual``: clears LIQUIDATE -> NORMAL (operator action only).
    """

    def __init__(
        self,
        market_id: str,
        l1_threshold: float = _DEFAULT_L1,
        l2_threshold: float = _DEFAULT_L2,
        l3_threshold: float = _DEFAULT_L3,
        baseline: Decimal = _ZERO,
    ) -> None:
        self._market_id = market_id
        self._l1 = Decimal(str(l1_threshold))
        self._l2 = Decimal(str(l2_threshold))
        self._l3 = Decimal(str(l3_threshold))
        self._level: CircuitLevel = CircuitLevel.NORMAL
        self._baseline: Decimal = baseline

    @property
    def level(self) -> CircuitLevel:
        """Return the current circuit breaker level."""
        return self._level

    @property
    def market_id(self) -> str:
        """Return the market identifier this breaker guards."""
        return self._market_id

    @property
    def baseline(self) -> Decimal:
        """Return the stored baseline equity for the current trading day."""
        return self._baseline

    def check(self, current_equity: Decimal, baseline_equity: Decimal) -> CircuitLevel:
        """Compute drawdown, update level, and return the new level.

        Args:
            current_equity: Portfolio equity right now.
            baseline_equity: Equity at the start of the trading day (baseline).

        Returns:
            The updated :class:`CircuitLevel`.
        """
        if baseline_equity <= _ZERO:
            self._level = CircuitLevel.LIQUIDATE
            return self._level

        drawdown = (baseline_equity - current_equity) / baseline_equity

        if drawdown >= self._l3:
            self._level = CircuitLevel.LIQUIDATE
        elif drawdown >= self._l2:
            self._level = CircuitLevel.HALTED
        elif drawdown >= self._l1:
            self._level = CircuitLevel.CAUTION
        else:
            self._level = CircuitLevel.NORMAL

        return self._level

    def reset_daily(self, new_baseline: Decimal) -> None:
        """Daily auto-reset: clears CAUTION and HALTED; updates baseline.

        LIQUIDATE is intentionally preserved -- it requires ``reset_manual``.

        Args:
            new_baseline: New baseline equity (typically today's opening equity).
                Stored so ``_strategy_cycle`` can retrieve the day-start snapshot.
        """
        self._baseline = new_baseline
        if self._level in (CircuitLevel.CAUTION, CircuitLevel.HALTED):
            self._level = CircuitLevel.NORMAL

    def reset_manual(self) -> None:
        """Operator-initiated reset: clears LIQUIDATE -> NORMAL."""
        self._level = CircuitLevel.NORMAL


class CrossMarketCircuitBreaker:
    """Monitors combined drawdown across all markets.

    Trips when ``(sum(baselines) - sum(currents)) / sum(baselines) >= halt_threshold``.
    Returns ``True`` (halted) or ``False`` (clear).
    """

    def __init__(self, halt_threshold: float = _DEFAULT_CROSS_HALT) -> None:
        self._threshold = Decimal(str(halt_threshold))

    def check(
        self,
        market_equities: dict[str, Decimal],
        baseline_equities: dict[str, Decimal],
    ) -> bool:
        """Return True if combined drawdown exceeds the halt threshold.

        Args:
            market_equities: Mapping of market_id to current equity.
            baseline_equities: Mapping of market_id to baseline equity.

        Returns:
            ``True`` if all markets should halt; ``False`` otherwise.
        """
        total_baseline = sum(baseline_equities.values(), _ZERO)
        if total_baseline <= _ZERO:
            return False

        total_current = sum(market_equities.values(), _ZERO)
        combined_drawdown = (total_baseline - total_current) / total_baseline
        return combined_drawdown >= self._threshold

    def reset_daily(self, new_baselines: dict[str, Decimal]) -> None:
        """Update internal state on daily reset.

        The cross-market breaker is stateless (computes on demand), so this
        method exists for symmetry with per-market breakers and future use.

        Args:
            new_baselines: New baseline equities per market (unused currently,
                callers pass updated baselines to ``check`` directly).
        """
