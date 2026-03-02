"""Run an evaluation backtest with full decision journaling.

Produces JSONL decision log + performance summary for evaluation agents.

Usage:
    uv run python scripts/run_evaluation.py \
        --symbol AAPL --segment us_tech \
        --start 2024-01-01 --end 2024-06-30 \
        --output results/2026-03-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

# Ensure project root is on path for config imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.backtest.decision_journal import DecisionJournal
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.core.schemas import BacktestResult, TradeResult
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.registry import MLModelRegistry
from finalayze.risk.kelly import RollingKelly
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.ml_strategy import MLStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.pairs import PairsStrategy

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "finalayze" / "strategies" / "presets"
)


def _load_preset(segment: str) -> dict[str, Any]:
    """Load YAML preset for a segment, returning empty dict on failure."""
    preset_path = _PRESETS_DIR / f"{segment}.yaml"
    if not preset_path.exists():
        return {}
    with preset_path.open() as f:
        return yaml.safe_load(f) or {}


def _setup_pairs_strategy(
    segment: str,
    fetcher: YFinanceFetcher,
    start: datetime,
    end: datetime,
) -> PairsStrategy | None:
    """Create a PairsStrategy with pre-loaded peer candles, or None if not configured."""
    preset = _load_preset(segment)
    pairs_cfg = preset.get("strategies", {}).get("pairs", {})
    if not pairs_cfg.get("enabled", False):
        return None

    raw_pairs: list[list[str]] = pairs_cfg.get("params", {}).get("pairs", [])
    if not raw_pairs:
        return None

    # Collect unique peer symbols
    peer_symbols: set[str] = set()
    for pair in raw_pairs:
        for sym in pair:
            peer_symbols.add(str(sym))

    strategy = PairsStrategy()

    for sym in peer_symbols:
        try:
            candles = fetcher.fetch_candles(sym, start, end)
            if candles:
                strategy.set_peer_candles(sym, candles)
                print(f"  Loaded {len(candles)} peer candles for {sym}")
        except Exception:
            print(f"  Failed to fetch peer candles for {sym}, skipping")

    return strategy


def _setup_ml_strategy(segment: str, models_dir: Path) -> MLStrategy | None:
    """Create an MLStrategy with loaded models, or None if no models found."""
    from finalayze.ml.models.ensemble import EnsembleModel  # noqa: PLC0415

    segment_dir = models_dir / segment
    if not segment_dir.is_dir():
        return None

    xgb_path = segment_dir / "xgb.pkl"
    lgbm_path = segment_dir / "lgbm.pkl"
    lstm_path = segment_dir / "lstm.pkl"

    models = []
    lstm_model = None

    if xgb_path.exists():
        from finalayze.ml.models.xgboost_model import XGBoostModel  # noqa: PLC0415

        models.append(XGBoostModel.load_from(xgb_path))
        print(f"  Loaded XGBoost model from {xgb_path}")

    if lgbm_path.exists():
        from finalayze.ml.models.lightgbm_model import LightGBMModel  # noqa: PLC0415

        models.append(LightGBMModel.load_from(lgbm_path))
        print(f"  Loaded LightGBM model from {lgbm_path}")

    if lstm_path.exists():
        from finalayze.ml.models.lstm_model import LSTMModel  # noqa: PLC0415

        lstm_model = LSTMModel(segment_id=segment)
        lstm_model.load(lstm_path)
        print(f"  Loaded LSTM model from {lstm_path}")

    if not models and lstm_model is None:
        return None

    ensemble = EnsembleModel(models=models, lstm_model=lstm_model)
    registry = MLModelRegistry()
    registry.register(segment, ensemble)
    return MLStrategy(registry)


def _build_strategies(
    segment: str,
    fetcher: YFinanceFetcher,
    start: datetime,
    end: datetime,
    models_dir: Path | None,
) -> list[BaseStrategy]:
    """Build the full strategy list for evaluation."""
    strategies: list[BaseStrategy] = [MomentumStrategy(), MeanReversionStrategy()]

    pairs = _setup_pairs_strategy(segment, fetcher, start, end)
    if pairs is not None:
        strategies.append(pairs)
        print(f"  PairsStrategy enabled for {segment}")

    if models_dir is not None:
        ml = _setup_ml_strategy(segment, models_dir)
        if ml is not None:
            strategies.append(ml)
            print(f"  MLStrategy enabled for {segment}")
        else:
            print(f"  MLStrategy: no models found in {models_dir / segment}")

    return strategies


def _print_results(
    symbol: str,
    segment: str,
    result: BacktestResult | None,
    journal: DecisionJournal,
) -> None:
    """Print human-readable evaluation results to stdout."""
    print()
    print("=" * 60)
    print(f"  EVALUATION RESULTS: {symbol} ({segment})")
    print("=" * 60)
    if result:
        print(f"  Sharpe Ratio:   {result.sharpe:.4f}")
        print(f"  Max Drawdown:   {result.max_drawdown:.4f}")
        print(f"  Win Rate:       {result.win_rate:.4f}")
        print(f"  Profit Factor:  {result.profit_factor:.4f}")
        print(f"  Total Return:   {result.total_return:.4f}")
        print(f"  Total Trades:   {result.total_trades}")
    else:
        print("  No trades executed.")
    print()

    summary = journal.summary()
    print("  Decision Summary:")
    for action, count in summary["action_counts"].items():
        print(f"    {action}: {count}")
    if summary["top_skip_reasons"]:
        print("  Top Skip Reasons:")
        for reason, count in summary["top_skip_reasons"].items():
            print(f"    {reason}: {count}")
    print("=" * 60)


def _write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    candle_count: int,
    trades: list[TradeResult],
    result: BacktestResult | None,
    journal: DecisionJournal,
) -> None:
    """Write performance summary JSON to disk."""
    perf_data: dict[str, Any] = {
        "symbol": args.symbol,
        "segment": args.segment,
        "start": args.start,
        "end": args.end,
        "initial_cash": str(args.cash),
        "total_candles": candle_count,
        "total_trades": len(trades),
        "metrics": result.model_dump(mode="json") if result else None,
        "journal_summary": journal.summary(),
    }
    summary_path.write_text(json.dumps(perf_data, indent=2, default=str))
    print(f"Performance summary written to {summary_path}")


def main() -> None:
    """Run evaluation backtest and produce decision journal + performance summary."""
    parser = argparse.ArgumentParser(description="Run evaluation backtest with decision journal")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--segment", required=True, help="Segment ID (e.g., us_tech)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=Decimal, default=Decimal(100_000), help="Initial cash")
    parser.add_argument("--output", default="results/evaluation", help="Output directory")
    parser.add_argument("--models-dir", default=None, help="Directory with trained ML models")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch candles
    print(f"Fetching {args.symbol} data from {args.start} to {args.end}...")
    fetcher = YFinanceFetcher(market_id="us")
    candles = fetcher.fetch_candles(args.symbol, start, end)
    if not candles:
        print("No candles fetched. Check symbol and date range.")
        sys.exit(1)
    print(f"Fetched {len(candles)} candles.")

    # Set up strategies (Momentum, MeanReversion, optionally Pairs + ML)
    models_dir = Path(args.models_dir) if args.models_dir else None
    strategies = _build_strategies(args.segment, fetcher, start, end, models_dir)

    # Set up journaling combiner + decision journal
    combiner = JournalingStrategyCombiner(strategies=strategies)
    journal_path = output_dir / "decision_journal.jsonl"
    journal = DecisionJournal(output_path=journal_path)

    # Run backtest
    print("Running evaluation backtest...")
    engine = BacktestEngine(
        strategy=combiner,
        initial_cash=args.cash,
        decision_journal=journal,
        rolling_kelly=RollingKelly(),
    )
    trades, snapshots = engine.run(
        symbol=args.symbol,
        segment_id=args.segment,
        candles=candles,
    )

    journal.flush()
    print(f"Decision journal written to {journal_path} ({len(journal.records)} records)")

    # Fetch benchmark candles
    bench_symbol = "EWZ" if args.segment.startswith("ru_") else "SPY"
    print(f"Fetching benchmark {bench_symbol}...")
    try:
        bench_candles = fetcher.fetch_candles(bench_symbol, start, end)
        print(f"Fetched {len(bench_candles)} benchmark candles.")
    except Exception:
        bench_candles = None
        print(f"Benchmark fetch failed for {bench_symbol}, continuing without benchmark.")

    # Analyze + write output
    result = PerformanceAnalyzer().analyze(trades, snapshots, benchmark_candles=bench_candles)
    summary_path = output_dir / "performance_summary.json"
    _write_summary(summary_path, args, len(candles), trades, result, journal)
    _print_results(args.symbol, args.segment, result, journal)


if __name__ == "__main__":
    main()
