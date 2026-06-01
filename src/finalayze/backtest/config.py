"""BacktestConfig — frozen dataclass for backtest engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.backtest.costs import TransactionCosts
    from finalayze.backtest.decision_journal import DecisionJournal
    from finalayze.core.schemas import MarketContext
    from finalayze.ml.meta_labeler import MetaLabeler
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
    "dividend_gap": 60,
    "pead": 63,
    "cbr_calendar": 30,
}

_DEFAULT_HOLD_BARS_FALLBACK = 30

# S1.4: ATR stop-loss multiplier source of truth moved to risk/stops.py
# (Layer 4) so backtest and live paths read the same numbers. Re-exports
# kept here to avoid breaking existing imports.
from finalayze.risk.stops import (  # noqa: E402
    DEFAULT_STRATEGY_STOP_ATR,
    resolve_stop_atr_multiplier,
)

__all__ = [
    "DEFAULT_STRATEGY_HOLD_BARS",
    "DEFAULT_STRATEGY_STOP_ATR",
    "MOEX_2022_BREAK",
    "BacktestConfig",
    "resolve_max_hold_bars",
    "resolve_stop_atr_multiplier",
]


_MOEX_HOLD_BARS_UPLIFT = 1.3


def resolve_max_hold_bars(
    max_hold_bars: int | dict[str, int],
    strategy_name: str,
    *,
    segment_id: str = "",
) -> int:
    """Resolve the max hold bars for a given strategy.

    Args:
        max_hold_bars: Either a single int (applied to all strategies) or a
            dict mapping strategy names to their specific max hold bars.
        strategy_name: The name of the strategy that opened the position.
        segment_id: Market segment identifier. MOEX segments (``ru_*``) get a
            1.3x uplift to account for lower liquidity and wider spreads.

    Returns:
        The effective max hold bars for this strategy.
    """
    if isinstance(max_hold_bars, int):
        base = max_hold_bars
    else:
        base = max_hold_bars.get(strategy_name, _DEFAULT_HOLD_BARS_FALLBACK)
    if segment_id.startswith("ru_"):
        base = int(base * _MOEX_HOLD_BARS_UPLIFT)
    return base


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
    # Liquidity-aware sizing: cap BUY quantity at N% of the fill bar's volume.
    # 5% of ADV is the institutional-execution convention. Set to 0 to disable.
    max_order_volume_pct: Decimal = Decimal("0.05")

    # EVT tail-risk sizing and copula correlation scaling (Sprint 3)
    use_evt_sizing: bool = False
    use_copula_scaling: bool = False

    # Per-segment position cap (independent of global max_positions)
    max_positions_per_segment: int = 8

    # D-09 / LIQ-07 (Phase 66): per-segment cap on the number of SIMULTANEOUSLY OPEN positions
    # in a shared-broker ``run_portfolio`` run. ``None`` (default) preserves the prior behaviour
    # (the portfolio-wide cap stays ``max_positions``). When set, it overrides ``max_positions``
    # for the shared ``PreTradeChecker`` so a wider universe cannot fragment capital across too
    # many tiny positions. Sourced from ``config.segments.SegmentConfig.max_concurrent_positions``
    # by ``scripts/run_iteration.py``; only effective in shared-broker ``run_portfolio`` (the
    # per-symbol ``run`` path gives each symbol its own broker, so the cap is silently ineffective
    # there -- see Phase-66 PATTERNS Pitfall 4).
    max_concurrent_positions: int | None = None

    # Ambient market data for cross-asset / regime features (Phase E)
    market_context: MarketContext | None = None

    # MetaLabeler for ML-based position sizing (predicts P(profitable))
    meta_labeler: MetaLabeler | None = None

    # Date ranges to exclude from vol/ATR calculations (e.g. MOEX 2022 closure).
    # Candles within these ranges remain in OHLCV for position tracking but are
    # skipped when computing volatility-based metrics.
    # Format: tuple of (start_date_iso, end_date_iso) inclusive strings.
    exclude_periods: tuple[tuple[str, str], ...] = ()

    # MOEX-specific sizing step data (Phase 9: Strategy Wiring)
    # RubOilRegimeSignal instance; typed as object to avoid circular import from config.py
    rub_oil_regime_signal: object | None = None
    # Brent-in-RUB price for BrentGateStep (0.0 = missing/disabled)
    brent_rub_price: float = 0.0

    # Phase 10: Macro Regime sizing step data
    # Yield curve slope (10Y-2Y) in basis points for CBRRegimeStep (0.0 = missing/disabled)
    yield_slope_bps: float = 0.0
    # CBR direction for SectorAllocationStep ("cut", "hold", "hike", "" = missing)
    cbr_direction: str = ""
    # Phase 60 (INTG-03): CPI YoY (decimal fraction) for the live/non-per-bar
    # CpiRiskOffStep caller (0.0 = missing/disabled). The backtest path resolves CPI
    # per-bar from the candle date instead; this field documents the live default.
    cpi_yoy_fraction: float = 0.0

    # S5.3: end-of-data close-out behaviour. ``False`` (default) leaves
    # positions open at the last bar — equity snapshots already reflect their
    # mark-to-market via the broker so Sharpe / max-DD remain honest, the
    # trade list omits a synthetic "exit at last close" that would otherwise
    # inflate Sharpe by skipping spread/slippage.  Set to ``True`` to recover
    # the old behaviour (every open position is closed at the last candle's
    # close price; useful for tooling that requires fully realised PnL).
    force_close_at_end: bool = False


# MOEX was closed Feb 28 - Mar 24 2022 with extreme dislocation before/after.
# This period distorts vol estimates 3-5x and teaches false mean-reversion patterns.
MOEX_2022_BREAK: tuple[tuple[str, str], ...] = (("2022-02-21", "2022-04-01"),)
