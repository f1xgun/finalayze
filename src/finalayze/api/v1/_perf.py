"""Adapter from DailyEquitySnapshot rows to PortfolioState for PerformanceAnalyzer reuse.

Phase 56 Plan 04 (PERF-01) — D-09 + D-11.

The /portfolio/performance endpoint reuses ``backtest.performance.PerformanceAnalyzer``
(Layer 4) for Sharpe / Sortino / MaxDD computation. PerformanceAnalyzer expects
``list[PortfolioState]`` (one per timestamp, with ``.equity`` populated). The DB
gives us per-market ``DailyEquitySnapshot`` rows. This adapter:

  1. Sums equities across all markets sharing a timestamp (D-11 portfolio-aggregate).
  2. Returns one PortfolioState per unique timestamp, sorted ascending.

cash + positions are zero-padded — PerformanceAnalyzer only reads ``.equity``.
Currency mixing (RUB + USD) inherits the daily_reset 'already mixed' behavior;
FX-adjusted variant is a documented limitation per D-11.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.schemas import PortfolioState

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from finalayze.core.models import DailyEquitySnapshot


def equity_snapshots_to_portfolio_states(
    rows: Iterable[DailyEquitySnapshot],
) -> list[PortfolioState]:
    """Aggregate per-market snapshots into portfolio-wide PortfolioState series.

    Sums equities across all markets sharing the same timestamp. Returns one
    PortfolioState per unique timestamp, sorted ascending. ``cash`` and
    ``positions`` are zero-padded — PerformanceAnalyzer only reads ``.equity``.

    Currency mixing (RUB + USD) inherits the daily_reset 'already mixed'
    behavior — documented limitation per D-11.
    """
    by_ts: dict[datetime, Decimal] = defaultdict(lambda: Decimal(0))
    for r in rows:
        by_ts[r.timestamp] += Decimal(r.equity)
    return [
        PortfolioState(
            cash=Decimal(0),
            positions={},
            equity=eq,
            timestamp=ts,
        )
        for ts, eq in sorted(by_ts.items())
    ]
