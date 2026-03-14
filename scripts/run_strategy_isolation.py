"""Run a single strategy in isolation to measure standalone performance.

Usage:
    uv run python scripts/run_strategy_isolation.py --strategy mean_reversion
    uv run python scripts/run_strategy_isolation.py --strategy ou_mean_reversion \
        --segment ru_blue_chips --start-date 2022-01-01 --end-date 2025-12-31
    uv run python scripts/run_strategy_isolation.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from dotenv import load_dotenv

load_dotenv()

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.costs import MOEX_COSTS, US_COSTS
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.core.schemas import PortfolioState, TradeResult
from finalayze.data.fetchers.caching import CachingFetcher
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.markets.instruments import build_default_registry
from finalayze.risk.kelly import RollingKelly
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.combiner import StrategyCombiner
from finalayze.strategies.dividend_gap import DividendGapStrategy
from finalayze.strategies.dual_momentum import DualMomentumStrategy
from finalayze.strategies.event_driven import EventDrivenStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.ou_mean_reversion import OUMeanReversionStrategy
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

# ── Strategy registry ────────────────────────────────────────────────────────

STRATEGY_CLASSES: dict[str, type[BaseStrategy]] = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "rsi2_connors": RSI2ConnorsStrategy,
    "ou_mean_reversion": OUMeanReversionStrategy,
    "dual_momentum": DualMomentumStrategy,
    "event_driven": EventDrivenStrategy,
    "dividend_gap": DividendGapStrategy,
}

# Default segment for isolation testing
DEFAULT_SEGMENT = "ru_blue_chips"

# Sharpe below this threshold → DROP the strategy
_SHARPE_DROP_THRESHOLD = -0.1

# Symbol universe (mirrors run_iteration.py)
UNIVERSE: dict[str, list[str]] = {
    "us_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "us_broad": ["SPY", "QQQ", "DIA", "IWM", "JNJ", "PG"],
    "us_finance": ["JPM", "BAC", "GS", "MS", "V", "MA"],
    "us_healthcare": ["UNH", "LLY", "PFE", "ABBV", "MRK", "TMO"],
    "ru_blue_chips": ["SBER", "LKOH", "GAZP", "YNDX", "MGNT", "ALRS", "VTBR", "POLY", "NVTK", "MTLR"],
    "ru_energy": ["LKOH", "GAZP", "ROSN", "NVTK", "TATN", "SNGS", "TRNFP", "BANEP"],
    "ru_finance": ["SBER", "SBERP", "VTBR", "TCSG", "CBOM", "BSPB", "MOEX"],
}


def _create_isolation_preset(strategy_name: str, tmp_dir: Path) -> Path:
    """Create a temporary YAML preset with only one strategy at weight=1.0."""
    preset = {
        "segment_id": "isolation",
        "normalize_mode": "firing",
        "min_combined_confidence": 0.20,
        "strategies": {
            strategy_name: {
                "enabled": True,
                "weight": 1.0,
                "params": {},
            },
        },
    }
    preset_path = tmp_dir / "isolation.yaml"
    with preset_path.open("w") as f:
        yaml.dump(preset, f)
    return preset_path


def _make_fetcher(segment: str) -> CachingFetcher:
    """Build a fetcher for isolation tests (TinkoffFetcher for MOEX, yfinance for US)."""
    if segment.startswith("ru_"):
        token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
        if token:
            registry = build_default_registry()
            return CachingFetcher(TinkoffFetcher(token=token, registry=registry, sandbox=False))
    market_id = "moex" if segment.startswith("ru_") else "us"
    return CachingFetcher(YFinanceFetcher(market_id=market_id))


def _build_config(segment: str, cash: Decimal) -> BacktestConfig:
    """Build backtest config with appropriate costs."""
    costs = MOEX_COSTS if segment.startswith("ru_") else US_COSTS
    return BacktestConfig(initial_cash=cash, transaction_costs=costs)


def _run_isolation(
    strategy_name: str,
    segment: str,
    start: datetime,
    end: datetime,
    cash: Decimal,
    output_dir: Path,
) -> dict[str, Any]:
    """Run a single strategy in isolation on a segment."""
    strategy = STRATEGY_CLASSES[strategy_name]()
    config = _build_config(segment, cash)
    fetcher = _make_fetcher(segment)
    symbols = UNIVERSE.get(segment, [])

    all_trades: list[TradeResult] = []
    all_snapshots: list[PortfolioState] = []

    print(f"\n{'=' * 60}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Segment:  {segment} ({len(symbols)} symbols)")
    print(f"  Period:   {start.date()} to {end.date()}")
    print(f"{'=' * 60}")

    # Create a combiner with just this one strategy and low threshold
    combiner = StrategyCombiner(
        strategies=[strategy],
        normalize_mode="firing",
    )
    # Override presets dir to use a temp dir with our isolation preset
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _write_isolation_preset(strategy_name, tmp_path, segment)
        combiner._presets_dir = tmp_path

        for symbol in symbols:
            try:
                candles = fetcher.fetch_candles(symbol, start, end)
                if not candles:
                    print(f"  {symbol:12s} | no data")
                    continue

                engine = BacktestEngine(
                    strategy=combiner,
                    config=config,
                    rolling_kelly=RollingKelly(fraction=0.75) if segment.startswith("ru_") else RollingKelly(),
                )
                trades, snapshots = engine.run(
                    symbol=symbol,
                    segment_id=segment,
                    candles=candles,
                )

                all_trades.extend(trades)
                all_snapshots.extend(snapshots)

                result = PerformanceAnalyzer().analyze(trades, snapshots)
                sharpe = float(result.sharpe) if result else 0.0
                wr = float(result.win_rate) if result else 0.0
                ret = float(result.total_return) if result else 0.0
                print(
                    f"  {symbol:12s} | {len(candles):4d} bars | "
                    f"{len(trades):3d} trades | "
                    f"Sharpe {sharpe:+7.3f} | "
                    f"WR {wr:5.1%} | "
                    f"Ret {ret:+7.3%}"
                )
            except Exception:
                print(f"  {symbol:12s} | ERROR — {traceback.format_exc().splitlines()[-1]}")

    # Compute aggregate metrics
    result = PerformanceAnalyzer().analyze(all_trades, all_snapshots) if all_trades else None

    summary: dict[str, Any] = {
        "strategy": strategy_name,
        "segment": segment,
        "period": f"{start.date()} to {end.date()}",
        "total_trades": len(all_trades),
        "sharpe": float(result.sharpe) if result else 0.0,
        "profit_factor": float(result.profit_factor) if result else 0.0,
        "max_drawdown": float(result.max_drawdown) if result else 0.0,
        "total_return": float(result.total_return) if result else 0.0,
        "win_rate": float(result.win_rate) if result else 0.0,
        "sortino_ratio": float(result.sortino_ratio) if result and result.sortino_ratio else 0.0,
    }

    # Print aggregate
    print(f"\n  AGGREGATE: {strategy_name} on {segment}")
    print(f"  Trades: {summary['total_trades']}")
    print(f"  Sharpe: {summary['sharpe']:+.4f}")
    print(f"  PF:     {summary['profit_factor']:.4f}")
    print(f"  DD:     {summary['max_drawdown']:.2%}")
    print(f"  Return: {summary['total_return']:+.2%}")
    print(f"  WR:     {summary['win_rate']:.1%}")

    # Save results
    strat_dir = output_dir / strategy_name
    strat_dir.mkdir(parents=True, exist_ok=True)
    summary_path = strat_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Saved:  {summary_path}")

    return summary


def _write_isolation_preset(strategy_name: str, tmp_dir: Path, segment: str) -> None:
    """Write a minimal preset YAML for isolation testing."""
    preset = {
        "segment_id": segment,
        "normalize_mode": "firing",
        "min_combined_confidence": 0.20,
        "min_exit_confidence": 0.05,
        "strategies": {
            strategy_name: {
                "enabled": True,
                "weight": 1.0,
                "params": {},
            },
        },
    }
    preset_path = tmp_dir / f"{segment}.yaml"
    with preset_path.open("w") as f:
        yaml.dump(preset, f)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run strategy isolation test")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_CLASSES.keys()),
        help="Strategy to test in isolation",
    )
    parser.add_argument("--all", action="store_true", help="Test all strategies")
    parser.add_argument("--segment", default=DEFAULT_SEGMENT, help="Segment to test on")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--cash", type=int, default=100_000)
    parser.add_argument("--output", default="results/isolation/")
    return parser.parse_args()


def main() -> None:
    """Run isolation tests."""
    args = _parse_args()

    if not args.strategy and not args.all:
        print("Error: specify --strategy <name> or --all")
        sys.exit(1)

    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    cash = Decimal(args.cash)
    output_dir = Path(args.output)

    strategies_to_test = list(STRATEGY_CLASSES.keys()) if args.all else [args.strategy]

    # Skip dividend_gap for non-RU segments
    if not args.segment.startswith("ru_"):
        strategies_to_test = [s for s in strategies_to_test if s != "dividend_gap"]

    results: list[dict[str, Any]] = []

    for strategy_name in strategies_to_test:
        summary = _run_isolation(
            strategy_name=strategy_name,
            segment=args.segment,
            start=start,
            end=end,
            cash=cash,
            output_dir=output_dir,
        )
        results.append(summary)

    # Print ranking table
    if len(results) > 1:
        print(f"\n{'=' * 72}")
        print("  ISOLATION TEST RANKING")
        print(f"{'=' * 72}")
        print(
            f"  {'Strategy':<22} {'Sharpe':>8} {'PF':>8} {'DD':>8} "
            f"{'Return':>8} {'WR':>6} {'Trades':>7}"
        )
        print(f"  {'-' * 70}")

        ranked = sorted(results, key=lambda r: r["sharpe"], reverse=True)
        for r in ranked:
            verdict = (
                "KEEP"
                if r["sharpe"] > 0.0
                else ("REDUCE" if r["sharpe"] > _SHARPE_DROP_THRESHOLD else "DROP")
            )
            print(
                f"  {r['strategy']:<22} {r['sharpe']:>+8.4f} {r['profit_factor']:>8.2f} "
                f"{r['max_drawdown']:>7.2%} {r['total_return']:>+7.2%} "
                f"{r['win_rate']:>5.1%} {r['total_trades']:>7d}  {verdict}"
            )

        # Save combined ranking
        ranking_path = output_dir / "ranking.json"
        ranking_path.parent.mkdir(parents=True, exist_ok=True)
        ranking_path.write_text(json.dumps(ranked, indent=2, default=str))
        print(f"\n  Ranking saved to: {ranking_path}")


if __name__ == "__main__":
    main()
