"""Unified ATR stop-loss multiplier resolution (Layer 4 — risk).

Single source of truth for the per-strategy ATR multiplier used to set
trailing/chandelier stop distance on BUY fills. Consumed by:

  * ``BacktestEngine`` via ``backtest/position_executor.py`` (knows segment_id)
  * ``signal_executor.py`` BUY-fill stop wiring (knows market_id at fill time)
  * ``position_manager.maybe_register_retroactive_stop`` (knows market_id +
    strategy via ``_entry_strategy`` map)

S1.4 consolidation: previously the live path used flat
``_ATR_MULTIPLIER_US = 2.0`` / ``_ATR_MULTIPLIER_MOEX = 2.5`` constants that
diverged from the backtest's per-strategy table. After S1.4 the same numbers
back both paths.

The function accepts either ``segment_id`` (preferred when known — backtest
path) or ``market_id`` (live fill path). MOEX uplift is triggered when
*either* indicates a MOEX position.

Why both args instead of one: in the live BUY-fill code path
(``signal_executor.py``) segment_id is not always available at the exact
moment we wire the stop — the broker round-trip carries only ``market_id``.
Forcing a segment lookup there would be a noisy refactor. The MOEX uplift
itself is a market-wide property (higher ATR/price ratio), not a segment
property, so accepting either is semantically clean.
"""

from __future__ import annotations

from decimal import Decimal

# Per-strategy ATR stop-loss multipliers. Wider stops for mean-reverting
# strategies (they fade noise; tight stops kill them). These are candidates
# for walk-forward optimisation (M5 issue) but currently hard-coded.
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

# Higher ATR/price ratio on MOEX (lower-liquidity book, wider intraday range)
# means a stop calibrated for US ranges trips too easily. 1.2x uplift was
# tuned in v9.1 walk-forward (Phase 47, asymmetric barriers).
_MOEX_UPLIFT = 1.2

# S3.1: Catastrophic-drop threshold that overrides the post-entry grace bar
# in both backtest (``BacktestEngine._iter_bars``) and live
# (``PositionTracker.check_stop_losses``). Quant-validated: 10 % is too tight
# for earnings/macro gaps; 15 % corresponds to a 3+ sigma daily move and is
# big enough that we accept the (probably fair) stop trigger even on the
# very first post-entry bar.
CATASTROPHIC_DROP_PCT = Decimal("0.15")


def resolve_stop_atr_multiplier(
    strategy_name: str,
    *,
    segment_id: str = "",
    market_id: str = "",
) -> Decimal:
    """Resolve the ATR stop-loss multiplier for a given strategy + market.

    Args:
        strategy_name: Strategy that opened the position (looked up in
            ``DEFAULT_STRATEGY_STOP_ATR``; unknown names fall back to 3.0).
        segment_id: ``ru_*`` prefix triggers MOEX uplift. Pass when known
            (backtest path).
        market_id: ``"moex"`` triggers MOEX uplift. Pass in live fill paths
            where segment_id is not in scope.

    Returns:
        Decimal multiplier. ``compute_atr_stop_loss(entry, candles,
        atr_multiplier=...)`` is the conventional consumer.
    """
    base = DEFAULT_STRATEGY_STOP_ATR.get(strategy_name, _DEFAULT_STOP_ATR_FALLBACK)
    is_moex = market_id == "moex" or segment_id.startswith("ru_")
    if is_moex:
        base *= _MOEX_UPLIFT
    return Decimal(str(base))
