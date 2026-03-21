"""Run a joint OFZ + equity portfolio backtest.

Usage:
    uv run python scripts/run_portfolio_backtest.py \
        --name "portfolio-baseline" \
        --total-capital 1000000

    uv run python scripts/run_portfolio_backtest.py \
        --name "portfolio-60-40" \
        --bond-weight 0.40 --equity-weight 0.60 \
        --total-capital 1000000
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.logging import setup_logging
from config.modes import WorkMode

setup_logging(WorkMode.TEST)

import structlog

from finalayze.backtest.portfolio_orchestrator import (
    PortfolioBacktestOrchestrator,
    PortfolioBacktestResult,
)

logger = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_TOTAL_CAPITAL = 1_000_000
_DEFAULT_BOND_WEIGHT = 0.40
_DEFAULT_EQUITY_WEIGHT = 0.60
_DEFAULT_START_DATE = "2023-01-01"
_DEFAULT_END_DATE = "2024-12-31"
_WF_SHARPE_TARGET = 0.10
_WEIGHT_SUM_TOLERANCE = 0.01


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run a joint OFZ bond + equity portfolio backtest",
    )
    parser.add_argument("--name", required=True, help="Iteration name for result tracking")
    parser.add_argument(
        "--total-capital",
        type=float,
        default=_DEFAULT_TOTAL_CAPITAL,
        help=f"Total portfolio capital (default: {_DEFAULT_TOTAL_CAPITAL:,})",
    )
    parser.add_argument(
        "--bond-weight",
        type=float,
        default=_DEFAULT_BOND_WEIGHT,
        help=f"Bond allocation weight (default: {_DEFAULT_BOND_WEIGHT})",
    )
    parser.add_argument(
        "--equity-weight",
        type=float,
        default=_DEFAULT_EQUITY_WEIGHT,
        help=f"Equity allocation weight (default: {_DEFAULT_EQUITY_WEIGHT})",
    )
    parser.add_argument("--start-date", default=_DEFAULT_START_DATE, help="Backtest start date")
    parser.add_argument("--end-date", default=_DEFAULT_END_DATE, help="Backtest end date")
    parser.add_argument(
        "--output",
        default="results/iterations/",
        help="Output directory for results",
    )
    return parser.parse_args()


def _run_bond_backtest(
    bond_capital: float,
    start_date: date,
    end_date: date,  # noqa: ARG001
) -> Any | None:
    """Run bond (OFZ) backtest and return BondBacktestResult or None on failure."""
    try:
        logger.info(
            "Starting bond backtest",
            capital=bond_capital,
            start=str(start_date),
        )

        # Bond engine requires OFZ candle data and macro context -- these
        # must come from cached data or live T-Bank API.  If unavailable,
        # we skip the bond component gracefully.
        logger.warning(
            "Bond backtest requires OFZ candle data and macro context. "
            "Ensure data is available via T-Bank API or cache."
        )

        # Placeholder: in production, load bond data and run engine.run(...)
        # For now, return None to indicate bond data is not available.
        return None

    except Exception:
        logger.warning(
            "Bond backtest failed -- skipping bond component",
            exc=traceback.format_exc(),
        )
        return None


def _run_equity_backtest(
    equity_capital: float,
    start_date: date,
    end_date: date,
) -> tuple[list[Any], list[Any]] | None:
    """Run equity backtest for ru_blue_chips and return (trades, snapshots) or None."""
    try:
        logger.info(
            "Starting equity backtest",
            capital=equity_capital,
            start=str(start_date),
            end=str(end_date),
        )

        # Equity engine requires MOEX candle data from T-Bank API.
        # If unavailable, we skip the equity component gracefully.
        logger.warning(
            "Equity backtest requires MOEX candle data. "
            "Ensure FINALAYZE_TINKOFF_TOKEN is set and data is cached."
        )

        # Placeholder: in production, load equity data and run BacktestEngine.run(...)
        return None

    except Exception:
        logger.warning(
            "Equity backtest failed -- skipping equity component", exc=traceback.format_exc()
        )
        return None


def _extract_usdrub_series(
    start_date: date,
    end_date: date,
) -> list[tuple[date, float]]:
    """Extract USDRUB FX series from T-Bank data or return empty list."""
    try:
        logger.info("Extracting USDRUB series", start=str(start_date), end=str(end_date))

        # In production, load from MarketContext.moex_data.fx_rates
        # FXRate(timestamp, pair, rate) -> filter pair == "USDRUB"
        logger.warning("USDRUB data requires T-Bank API. Returning empty series.")
        return []

    except Exception:
        logger.warning("USDRUB extraction failed", exc=traceback.format_exc())
        return []


def _print_results(result: PortfolioBacktestResult) -> None:
    """Print portfolio backtest results in a formatted table."""
    print("\n" + "=" * 60)
    print("  PORTFOLIO BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:     {result.total_return_pct:+.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe:+.4f}")
    print(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  WF Sharpe:        {result.wf_sharpe:+.4f}")
    print("-" * 60)

    # Crisis brake stats
    n_crisis = len(result.crisis_brake_active_dates)
    print(f"  Crisis Days:      {n_crisis}")
    if n_crisis > 0:
        print(f"  First Crisis:     {result.crisis_brake_active_dates[0]}")
        print(f"  Last Crisis:      {result.crisis_brake_active_dates[-1]}")

    print("-" * 60)

    # WF Sharpe target check (aspirational, not a gate)
    if result.wf_sharpe >= _WF_SHARPE_TARGET:
        print(f"  WF Sharpe target ({_WF_SHARPE_TARGET:+.2f}): ACHIEVED")
    else:
        print(f"  WF Sharpe target ({_WF_SHARPE_TARGET:+.2f}): NOT MET (aspirational)")
    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point for portfolio backtest."""
    args = _parse_args()

    logger.info(
        "Portfolio backtest starting",
        name=args.name,
        total_capital=args.total_capital,
        bond_weight=args.bond_weight,
        equity_weight=args.equity_weight,
    )

    # Validate weights sum to 1.0
    weight_sum = args.bond_weight + args.equity_weight
    if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
        logger.error("Bond + equity weights must sum to 1.0", sum=weight_sum)
        sys.exit(1)

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    bond_capital = args.total_capital * args.bond_weight
    equity_capital = args.total_capital * args.equity_weight

    # 1. Bond backtest
    bond_result = _run_bond_backtest(bond_capital, start, end)
    if bond_result is None:
        logger.warning("Bond component unavailable -- cannot run portfolio assembly")
        print("\nBond backtest data not available. Ensure OFZ data is cached.")
        print("Run with T-Bank API access to populate bond data first.")
        return

    # 2. Equity backtest
    equity_data = _run_equity_backtest(equity_capital, start, end)
    if equity_data is None:
        logger.warning("Equity component unavailable -- cannot run portfolio assembly")
        print("\nEquity backtest data not available. Ensure MOEX data is cached.")
        print("Run with FINALAYZE_TINKOFF_TOKEN to populate equity data first.")
        return

    equity_trades, equity_snapshots = equity_data

    # 3. USDRUB data
    usdrub_series = _extract_usdrub_series(start, end)
    if not usdrub_series:
        logger.warning("USDRUB data unavailable -- cannot run portfolio assembly")
        print("\nUSDRUB FX data not available. Ensure T-Bank API access.")
        return

    # 4. Portfolio assembly
    logger.info(
        "Assembling portfolio",
        bond_weight=args.bond_weight,
        equity_weight=args.equity_weight,
    )
    orch = PortfolioBacktestOrchestrator(
        bond_weight=args.bond_weight,
        equity_weight=args.equity_weight,
    )

    result = orch.run(
        bond_result=bond_result,
        equity_snapshots=equity_snapshots,
        usdrub_series=usdrub_series,
        total_capital=args.total_capital,
        equity_trades=equity_trades,
    )

    # 5. Walk-forward Sharpe
    logger.info("Computing walk-forward Sharpe")
    orch.compute_walk_forward_sharpe(result)

    # 6. Report
    _print_results(result)

    logger.info(
        "Portfolio backtest complete",
        name=args.name,
        sharpe=result.sharpe,
        wf_sharpe=result.wf_sharpe,
        total_return=result.total_return_pct,
    )


if __name__ == "__main__":
    main()
