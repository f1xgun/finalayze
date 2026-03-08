"""BacktestConfig — frozen dataclass for backtest engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts
    from finalayze.backtest.decision_journal import DecisionJournal
    from finalayze.core.schemas import MarketContext
    from finalayze.risk.circuit_breaker import CircuitBreaker
    from finalayze.risk.kelly import RollingKelly
    from finalayze.risk.loss_limits import LossLimitTracker

# Default per-strategy max holding periods (bars).
# NOTE: These are candidates for walk-forward optimization (M5 issue).
# Each value could be tuned per-fold in WalkForwardOptimizer via param_grid
# rather than being hardcoded constants.
DEFAULT_STRATEGY_HOLD_BARS: dict[str, int] = {
    "momentum": 30,
    "dual_momentum": 30,
    "mean_reversion": 20,
    "ou_mean_reversion": 25,
    "pairs": 20,
    "event_driven": 63,
    "rsi2_connors": 5,
    "ml_ensemble": 20,
    "dividend_gap": 15,
    "pead": 63,
    "cbr_calendar": 30,
}

_DEFAULT_HOLD_BARS_FALLBACK = 30

# Per-strategy ATR stop-loss multipliers (wider stops for mean-reversion).
# NOTE: These are candidates for walk-forward optimization (M5 issue).
# Each value could be tuned per-fold in WalkForwardOptimizer via param_grid
# rather than being hardcoded constants.
DEFAULT_STRATEGY_STOP_ATR: dict[str, float] = {
    "momentum": 2.5,
    "dual_momentum": 3.0,
    "mean_reversion": 3.5,
    "ou_mean_reversion": 3.5,
    "pairs": 3.0,
    "event_driven": 3.0,
    "rsi2_connors": 2.5,
    "ml_ensemble": 2.0,
    "dividend_gap": 3.0,
    "pead": 3.0,
    "cbr_calendar": 3.0,
}

_DEFAULT_STOP_ATR_FALLBACK = 3.0


def resolve_stop_atr_multiplier(
    strategy_name: str,
    *,
    segment_id: str = "",
) -> Decimal:
    """Resolve the ATR stop-loss multiplier for a strategy.

    This function is called by ``BacktestEngine`` to set the trailing/chandelier
    stop distance when opening a position.  MOEX segments (``ru_*``) get a 1.2x
    uplift due to higher ATR/price ratios.

    Precedence note:
        The engine uses **this function** to determine the stop ATR multiplier.
        Individual strategies may define ``params.stop_atr_multiplier`` in their
        YAML preset, but that value is consumed only inside the strategy itself
        (e.g. for sizing or signal strength) -- it does NOT override the engine
        stop.  To change the engine stop, update ``DEFAULT_STRATEGY_STOP_ATR``.
    """
    base = DEFAULT_STRATEGY_STOP_ATR.get(strategy_name, _DEFAULT_STOP_ATR_FALLBACK)
    if segment_id.startswith("ru_"):
        base *= 1.2
    return Decimal(str(base))


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

    # Market impact model (Sprint 2)
    use_impact_model: bool = False
    impact_coeff: float = 0.1
    max_impact_bps: float = 50.0

    # EVT tail-risk sizing and copula correlation scaling (Sprint 3)
    use_evt_sizing: bool = False
    use_copula_scaling: bool = False

    # Ambient market data for cross-asset / regime features (Phase E)
    market_context: MarketContext | None = None
