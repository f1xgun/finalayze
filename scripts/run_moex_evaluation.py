"""Run evaluation on real MOEX stocks via yfinance (.ME suffix).

yfinance provides MOEX data from 2020-01-01 to ~2022-05-24 (pre-sanctions cutoff).
This lets us validate strategy performance on actual Russian stocks instead of ETF proxies.

Usage:
    uv run python scripts/run_moex_evaluation.py --output results/moex-real/
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.backtest.decision_journal import DecisionJournal
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.registry import MLModelRegistry
from finalayze.risk.kelly import RollingKelly
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.ml_strategy import MLStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "finalayze" / "strategies" / "presets"
)

# Real MOEX stocks available via yfinance with .ME suffix
# Data available: 2020-01-01 to ~2022-05-24
MOEX_UNIVERSE: dict[str, list[str]] = {
    "ru_blue_chips": [
        "SBER.ME",  # Sberbank
        "GAZP.ME",  # Gazprom
        "LKOH.ME",  # Lukoil
        "GMKN.ME",  # Norilsk Nickel
        "VTBR.ME",  # VTB Bank
        "MOEX.ME",  # Moscow Exchange
    ],
    "ru_energy": [
        "NVTK.ME",  # Novatek
        "ROSN.ME",  # Rosneft
        "GAZP.ME",  # Gazprom (also energy)
        "LKOH.ME",  # Lukoil (also energy)
    ],
    "ru_finance": [
        "SBER.ME",  # Sberbank
        "VTBR.ME",  # VTB Bank
        "MOEX.ME",  # Moscow Exchange
    ],
}


def _load_preset(segment: str) -> dict[str, Any]:
    """Load YAML preset for a segment."""
    preset_path = _PRESETS_DIR / f"{segment}.yaml"
    if not preset_path.exists():
        return {}
    with preset_path.open() as f:
        return yaml.safe_load(f) or {}


def _setup_ml_strategy(segment: str, models_dir: Path) -> MLStrategy | None:
    """Try to load ML models for a segment."""
    segment_dir = models_dir / segment
    if not segment_dir.exists():
        return None
    registry = MLModelRegistry()
    try:
        registry.load_segment(segment, segment_dir)
    except Exception:
        return None
    return MLStrategy(registry=registry)


def _build_strategies(
    segment: str,
    models_dir: Path | None,
) -> list[BaseStrategy]:
    """Build the strategy list for evaluation (same as batch eval)."""
    strategies: list[BaseStrategy] = [
        MomentumStrategy(),
        MeanReversionStrategy(),
        RSI2ConnorsStrategy(),
    ]

    # ML Ensemble
    if models_dir is not None:
        ml = _setup_ml_strategy(segment, models_dir)
        if ml is not None:
            strategies.append(ml)

    return strategies


def _write_summary(output_dir: Path, all_results: list[dict[str, Any]]) -> None:
    """Write results JSON and print aggregate stats."""
    summary_path = output_dir / "moex_real_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"  DONE: {len(all_results)} successful evaluations")
    print(f"  Results: {summary_path}")
    print(f"{'=' * 70}")

    if all_results:
        sharpes = [float(r["metrics"]["sharpe"]) for r in all_results]
        win_rates = [float(r["metrics"]["win_rate"]) for r in all_results]
        trades_list = [r["total_trades"] for r in all_results]
        pos_sharpe = sum(1 for s in sharpes if s > 0)
        print("\n  MOEX AGGREGATE STATS:")
        print(f"  Avg trades/symbol:  {sum(trades_list) / len(trades_list):.1f}")
        print(f"  Avg win rate:       {sum(win_rates) / len(win_rates) * 100:.1f}%")
        print(f"  Avg Sharpe:         {sum(sharpes) / len(sharpes):+.3f}")
        print(f"  Positive Sharpe:    {pos_sharpe}/{len(all_results)}")


def run_moex_evaluation(
    output_dir: Path,
    models_dir: Path | None = None,
) -> None:
    """Run evaluation on real MOEX stocks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = YFinanceFetcher(market_id="moex")

    # yfinance MOEX data range
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2022, 6, 1, tzinfo=UTC)

    all_results: list[dict[str, Any]] = []

    for segment, symbols in MOEX_UNIVERSE.items():
        print(f"\n{'=' * 70}")
        print(f"  SEGMENT: {segment} ({len(symbols)} symbols, market=moex)")
        print(f"{'=' * 70}")

        strategies = _build_strategies(segment, models_dir)
        strat_names = [s.name for s in strategies]
        print(f"  Strategies: {', '.join(strat_names)}")

        if not strategies:
            print("  No strategies enabled — skipping")
            continue

        combiner = JournalingStrategyCombiner(strategies=strategies)

        seen = set()
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)

            try:
                candles = fetcher.fetch_candles(symbol, start, end)
                min_bars = 60
                if len(candles) < min_bars:
                    print(f"  {symbol:12s} | {len(candles)} bars — insufficient data, skipping")
                    continue

                sym_dir = output_dir / segment / symbol.replace(".", "_")
                sym_dir.mkdir(parents=True, exist_ok=True)
                journal = DecisionJournal(output_path=sym_dir / "decision_journal.jsonl")

                engine = BacktestEngine(
                    strategy=combiner,
                    initial_cash=Decimal(100000),
                    rolling_kelly=RollingKelly(window=50),
                    decision_journal=journal,
                )
                trades, snapshots = engine.run(symbol, segment, candles)
                journal.flush()

                if not trades:
                    print(
                        f"  {symbol:12s} | {len(candles)} bars |   0 trades | No trades generated"
                    )
                    continue

                analyzer = PerformanceAnalyzer()
                metrics = analyzer.analyze(trades, snapshots)

                sharpe = float(metrics.sharpe)
                wr = float(metrics.win_rate) * 100
                ret = float(metrics.total_return) * 100
                n_trades = metrics.total_trades

                buy_count = sum(1 for t in trades if t.side != "SELL")
                sell_count = sum(1 for t in trades if t.side == "SELL")

                print(
                    f"  {symbol:12s} | {len(candles):4d} bars | {n_trades:3d} trades | "
                    f"Sharpe {sharpe:+6.3f} | WR {wr:4.1f}% | Ret {ret:+6.3f}% | "
                    f"BUY {buy_count:3d} SELL {sell_count:3d}"
                )

                all_results.append(
                    {
                        "symbol": symbol,
                        "segment": segment,
                        "total_candles": len(candles),
                        "total_trades": n_trades,
                        "metrics": {
                            "sharpe": str(metrics.sharpe),
                            "win_rate": str(metrics.win_rate),
                            "profit_factor": str(metrics.profit_factor),
                            "total_return": str(metrics.total_return),
                            "max_drawdown": str(metrics.max_drawdown),
                            "total_trades": n_trades,
                        },
                    }
                )

            except Exception as e:
                print(f"  {symbol:12s} | ERROR: {e}")
                traceback.print_exc()

    _write_summary(output_dir, all_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOEX evaluation on real stocks")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--models-dir", default=None, help="Directory with trained ML models")
    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else None
    run_moex_evaluation(Path(args.output), models_dir)


if __name__ == "__main__":
    main()
