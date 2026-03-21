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
import os
import sys
import traceback
from datetime import UTC, date, datetime
from decimal import Decimal
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

from finalayze.backtest.bond_engine import BondBacktestConfig, BondBacktestEngine
from finalayze.backtest.config import DEFAULT_STRATEGY_HOLD_BARS, MOEX_2022_BREAK, BacktestConfig
from finalayze.backtest.costs import MOEX_COSTS
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.backtest.portfolio_orchestrator import (
    PortfolioBacktestOrchestrator,
    PortfolioBacktestResult,
)
from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS, Candle
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import build_default_registry
from finalayze.strategies.bond_duration_rotation import BondDurationRotationStrategy
from finalayze.strategies.dual_momentum import DualMomentumStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy

logger = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_TOTAL_CAPITAL = 1_000_000
_DEFAULT_BOND_WEIGHT = 0.40
_DEFAULT_EQUITY_WEIGHT = 0.60
_DEFAULT_START_DATE = "2023-01-01"
_DEFAULT_END_DATE = "2024-12-31"
_WF_SHARPE_TARGET = 0.10
_WEIGHT_SUM_TOLERANCE = 0.01

# OFZ tickers (same as scripts/validate_ofz_data.py)
_OFZ_PD_TICKERS = [
    "SU26238RMFS4",
    "SU26239RMFS2",
    "SU26241RMFS8",
    "SU26243RMFS4",
    "SU26244RMFS2",
    "SU26246RMFS7",
    "SU26252RMFS5",
    "SU26253RMFS3",
]
_OFZ_PK_TICKERS = [
    "SU29007RMFS0",
    "SU29008RMFS8",
    "SU29009RMFS6",
    "SU29010RMFS4",
]
_ALL_OFZ_TICKERS = _OFZ_PD_TICKERS + _OFZ_PK_TICKERS

_EQUITY_SEGMENT = "ru_blue_chips"
_EQUITY_SYMBOLS = [
    "SBER",
    "LKOH",
    "GMKN",
    "NVTK",
    "ROSN",
    "TATN",
    "MGNT",
    "YNDX",
    "AFKS",
    "CHMF",
    "NLMK",
    "MAGN",
    "MOEX",
    "OZON",
    "POSI",
    "SBERP",
    "TATNP",
    "PLZL",
]

_USDRUB_FIGI = "BBG0013HGFT4"  # USD000UTSTOM spot on MOEX


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
    end_date: date,
) -> Any | None:
    """Run bond (OFZ) backtest and return BondBacktestResult or None on failure."""
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        logger.warning("FINALAYZE_TINKOFF_TOKEN not set -- skipping bond backtest")
        return None

    try:
        logger.info(
            "Starting bond backtest",
            capital=bond_capital,
            start=str(start_date),
            end=str(end_date),
        )

        fetcher = TinkoffFetcher(
            token=token,
            registry=build_default_registry(),
            sandbox=False,
        )

        candles_by_symbol: dict[str, list[Candle]] = {}
        bond_info_dict: dict[str, Any] = {}
        coupon_schedule: dict[str, list[Any]] = {}

        for ticker in _ALL_OFZ_TICKERS:
            logger.debug("fetching_bond_data", ticker=ticker)
            try:
                info = fetcher.fetch_bond_info(ticker)
                raw = fetcher.fetch_bond_candles(info.figi, start_date, end_date)
                if not raw:
                    logger.debug("no_candles", ticker=ticker)
                    continue
                candles = [
                    Candle(
                        symbol=ticker,
                        market_id="moex",
                        timeframe="1d",
                        open=r["close"],
                        high=r["close"],
                        low=r["close"],
                        close=r["close"],
                        volume=r["volume"],
                        timestamp=datetime.combine(
                            r["date"],
                            datetime.min.time(),
                            tzinfo=UTC,
                        ),
                    )
                    for r in raw
                ]
                coupons = fetcher.fetch_bond_coupons(info.figi, start_date, end_date)
                candles_by_symbol[ticker] = candles
                bond_info_dict[ticker] = info
                coupon_schedule[ticker] = coupons
            except Exception:
                logger.warning(
                    "bond_data_fetch_failed",
                    ticker=ticker,
                    exc=traceback.format_exc(),
                )

        if not candles_by_symbol:
            logger.warning("no_bond_data_available")
            return None

        # Build strategy from bond metadata
        durations = {sym: Decimal(5) for sym in candles_by_symbol}
        maturities = {sym: bond_info_dict[sym].maturity_date for sym in candles_by_symbol}
        coupon_rates = {sym: bond_info_dict[sym].coupon_rate for sym in candles_by_symbol}
        strategy = BondDurationRotationStrategy(
            bond_durations=durations,
            bond_maturities=maturities,
            coupon_rates=coupon_rates,
        )

        config = BondBacktestConfig(initial_cash=Decimal(str(int(bond_capital))))
        engine = BondBacktestEngine(config=config)

        return engine.run(
            candles_by_symbol=candles_by_symbol,
            bond_info=bond_info_dict,
            coupon_schedule=coupon_schedule,
            strategy_fn=strategy,
            layer_configs=DEFAULT_LAYER_CONFIGS,
            as_of_date=end_date,
        )

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
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        logger.warning("FINALAYZE_TINKOFF_TOKEN not set -- skipping equity backtest")
        return None

    try:
        logger.info(
            "Starting equity backtest",
            capital=equity_capital,
            start=str(start_date),
            end=str(end_date),
        )

        fetcher = TinkoffFetcher(
            token=token,
            registry=build_default_registry(),
            sandbox=False,
        )

        all_trades: list[Any] = []
        all_snapshots: list[Any] = []

        per_symbol_capital = equity_capital / max(len(_EQUITY_SYMBOLS), 1)

        for symbol in _EQUITY_SYMBOLS:
            logger.debug("fetching_equity_candles", symbol=symbol)
            try:
                start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=UTC)
                end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=UTC)
                candles = fetcher.fetch_candles(symbol, start_dt, end_dt)
                if not candles:
                    logger.debug("no_equity_candles", symbol=symbol)
                    continue

                strategies = [DualMomentumStrategy(), MeanReversionStrategy()]
                combiner = JournalingStrategyCombiner(
                    strategies=strategies,
                    allocation_mode="equal",
                )
                engine = BacktestEngine(
                    config=BacktestConfig(
                        initial_cash=Decimal(str(int(per_symbol_capital))),
                        transaction_costs=MOEX_COSTS,
                        exclude_periods=MOEX_2022_BREAK,
                        max_hold_bars=DEFAULT_STRATEGY_HOLD_BARS,
                    ),
                    strategy=combiner,
                )
                trades, snapshots = engine.run(
                    symbol=symbol,
                    segment_id=_EQUITY_SEGMENT,
                    candles=candles,
                )
                all_trades.extend(trades)
                all_snapshots.extend(snapshots)
            except Exception:
                logger.warning(
                    "equity_symbol_failed",
                    symbol=symbol,
                    exc=traceback.format_exc(),
                )

        if not all_snapshots:
            logger.warning("no_equity_snapshots_produced")
            return None

        return all_trades, all_snapshots

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
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        logger.warning("FINALAYZE_TINKOFF_TOKEN not set -- cannot fetch USDRUB")
        return []

    try:
        logger.info("Extracting USDRUB series", start=str(start_date), end=str(end_date))

        fetcher = TinkoffFetcher(
            token=token,
            registry=build_default_registry(),
            sandbox=False,
        )
        raw = fetcher.fetch_bond_candles(_USDRUB_FIGI, start_date, end_date)
        if not raw:
            logger.warning("usdrub_candles_empty")
            return []
        return [(r["date"], float(r["close"])) for r in raw]

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
