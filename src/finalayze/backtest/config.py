"""BacktestConfig — frozen dataclass for backtest engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts
    from finalayze.backtest.decision_journal import DecisionJournal
    from finalayze.risk.circuit_breaker import CircuitBreaker
    from finalayze.risk.kelly import RollingKelly
    from finalayze.risk.loss_limits import LossLimitTracker

# Default per-strategy max holding periods (bars).
DEFAULT_STRATEGY_HOLD_BARS: dict[str, int] = {
    "momentum": 40,
    "mean_reversion": 20,
    "pairs": 15,
    "event_driven": 63,
    "rsi2_connors": 10,
    "ml_ensemble": 20,
}

_DEFAULT_HOLD_BARS_FALLBACK = 30


def resolve_max_hold_bars(
    max_hold_bars: int | dict[str, int],
    strategy_name: str,
) -> int:
    """Resolve the max hold bars for a given strategy.

    Args:
        max_hold_bars: Either a single int (applied to all strategies) or a
            dict mapping strategy names to their specific max hold bars.
        strategy_name: The name of the strategy that opened the position.

    Returns:
        The effective max hold bars for this strategy.
    """
    if isinstance(max_hold_bars, int):
        return max_hold_bars
    return max_hold_bars.get(strategy_name, _DEFAULT_HOLD_BARS_FALLBACK)


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable configuration for BacktestEngine.

    All fields mirror the original BacktestEngine constructor parameters
    with the same defaults. Pass an instance to BacktestEngine(config=...)
    or continue using keyword arguments for backward compatibility.
    """

    initial_cash: Decimal = Decimal(100000)
    max_position_pct: Decimal = Decimal("0.20")
    max_positions: int = 10
    kelly_fraction: Decimal = Decimal("0.5")
    atr_multiplier: Decimal = Decimal("3.0")
    transaction_costs: TransactionCosts | None = None
    trail_activation_atr: Decimal = Decimal("1.0")
    trail_distance_atr: Decimal = Decimal("1.5")
    circuit_breaker: CircuitBreaker | None = None
    rolling_kelly: RollingKelly | None = None
    loss_limits: LossLimitTracker | None = None
    target_vol: Decimal | None = None
    decision_journal: DecisionJournal | None = None
    profit_target_atr: Decimal = Decimal("5.0")
    max_hold_bars: int | dict[str, int] = field(default=30)

    # Stop-loss mode: "trailing" (default) or "chandelier"
    stop_loss_mode: str = "trailing"
    trend_filter_enabled: bool = False
    trend_sma_period: int = 200
