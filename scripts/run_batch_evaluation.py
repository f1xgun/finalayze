"""Run batch evaluation across multiple symbols and segments.

Usage:
    uv run python scripts/run_batch_evaluation.py --output results/2026-03-01-batch
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
from finalayze.core.schemas import BacktestResult
from finalayze.data.fetchers.base import BaseFetcher
from finalayze.data.fetchers.caching import CachingFetcher
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.registry import MLModelRegistry
from finalayze.risk.kelly import RollingKelly
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.mean_reversion import MeanReversionStrategy
from finalayze.strategies.ml_strategy import MLStrategy
from finalayze.strategies.momentum import MomentumStrategy
from finalayze.strategies.pairs import PairsStrategy
from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

_PRESETS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "finalayze" / "strategies" / "presets"
)

# ── Symbol universe ────────────────────────────────────────────────────────────
UNIVERSE: dict[str, list[str]] = {
    # US Tech (12 symbols)
    "us_tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSM",
        "AVGO",
        "ADBE",
        "CRM",
        "INTC",
        "AMD",
    ],
    # US Broad (10 symbols)
    "us_broad": [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "JNJ",
        "PG",
        "KO",
        "WMT",
        "XOM",
        "CVX",
    ],
    # US Finance (8 symbols)
    "us_finance": [
        "JPM",
        "BAC",
        "GS",
        "MS",
        "V",
        "MA",
        "BRK-B",
        "C",
    ],
    # US Healthcare (8 symbols)
    "us_healthcare": [
        "UNH",
        "LLY",
        "PFE",
        "ABBV",
        "MRK",
        "TMO",
        "ABT",
        "AMGN",
    ],
    # RU Blue Chips — MOEX unavailable on yfinance (sanctions).
    # Using ru_blue_chips preset with Russian-adjacent / EM proxies.
    "ru_blue_chips": [
        "RSX",
        "ERUS",
        "FLRU.L",
        "TUR",
        "EWZ",
        "INDA",
    ],
    # RU Energy — using global energy proxies with ru_energy preset
    "ru_energy": [
        "XLE",
        "BP",
        "SHEL",
        "TTE",
        "ENB",
    ],
}


def _load_preset(segment: str) -> dict[str, Any]:
    """Load YAML preset for a segment, returning empty dict on failure."""
    preset_path = _PRESETS_DIR / f"{segment}.yaml"
    if not preset_path.exists():
        return {}
    with preset_path.open() as f:
        return yaml.safe_load(f) or {}


def _setup_pairs_strategy(
    segment: str,
    fetcher: BaseFetcher,
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
        except Exception:
            continue  # peer fetch failure is non-fatal

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

    if lgbm_path.exists():
        from finalayze.ml.models.lightgbm_model import LightGBMModel  # noqa: PLC0415

        models.append(LightGBMModel.load_from(lgbm_path))

    if lstm_path.exists():
        from finalayze.ml.models.lstm_model import LSTMModel  # noqa: PLC0415

        lstm_model = LSTMModel(segment_id=segment)
        lstm_model.load(lstm_path)

    if not models and lstm_model is None:
        return None

    ensemble = EnsembleModel(models=models, lstm_model=lstm_model)
    registry = MLModelRegistry()
    registry.register(segment, ensemble)
    return MLStrategy(registry)


def _build_strategies(
    segment: str,
    fetcher: BaseFetcher,
    start: datetime,
    end: datetime,
    models_dir: Path | None,
) -> list[BaseStrategy]:
    """Build the full strategy list for evaluation."""
    strategies: list[BaseStrategy] = [
        MomentumStrategy(),
        MeanReversionStrategy(),
        RSI2ConnorsStrategy(),
    ]

    pairs = _setup_pairs_strategy(segment, fetcher, start, end)
    if pairs is not None:
        strategies.append(pairs)

    if models_dir is not None:
        ml = _setup_ml_strategy(segment, models_dir)
        if ml is not None:
            strategies.append(ml)

    return strategies


def _run_single(
    symbol: str,
    segment: str,
    start: datetime,
    end: datetime,
    cash: Decimal,
    output_dir: Path,
    market_id: str,
    benchmark_candles: list[Any] | None = None,
    strategies: list[BaseStrategy] | None = None,
) -> dict[str, Any] | None:
    """Run a single symbol evaluation. Returns summary dict or None on failure."""
    sym_dir = output_dir / segment / symbol.replace(".", "_")
    sym_dir.mkdir(parents=True, exist_ok=True)

    try:
        fetcher = CachingFetcher(YFinanceFetcher(market_id=market_id))
        candles = fetcher.fetch_candles(symbol, start, end)
        if not candles:
            print(f"  {symbol}: no data")
            return None

        strats = (
            strategies if strategies is not None else [MomentumStrategy(), MeanReversionStrategy()]
        )
        combiner = JournalingStrategyCombiner(strategies=strats)
        journal = DecisionJournal(output_path=sym_dir / "decision_journal.jsonl")

        engine = BacktestEngine(
            strategy=combiner,
            initial_cash=cash,
            decision_journal=journal,
            rolling_kelly=RollingKelly(),
        )
        trades, snapshots = engine.run(
            symbol=symbol,
            segment_id=segment,
            candles=candles,
        )
        journal.flush()

        result = PerformanceAnalyzer().analyze(
            trades, snapshots, benchmark_candles=benchmark_candles
        )
        summary = _build_summary(symbol, segment, candles, trades, result, journal)

        (sym_dir / "performance_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )

        _print_one_liner(symbol, segment, len(candles), len(trades), result, journal)
        return summary

    except Exception:
        print(f"  {symbol}: ERROR — {traceback.format_exc().splitlines()[-1]}")
        return None


def _build_summary(
    symbol: str,
    segment: str,
    candles: list[Any],
    trades: list[Any],
    result: BacktestResult | None,
    journal: DecisionJournal,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "segment": segment,
        "total_candles": len(candles),
        "total_trades": len(trades),
        "metrics": result.model_dump(mode="json") if result else None,
        "journal_summary": journal.summary(),
    }


def _print_one_liner(
    symbol: str,
    _segment: str,
    n_candles: int,
    n_trades: int,
    result: BacktestResult | None,
    journal: DecisionJournal,
) -> None:
    j = journal.summary()
    buys = j["action_counts"].get("BUY", 0)
    skips = j["action_counts"].get("SKIP", 0)
    if result:
        print(
            f"  {symbol:12s} | {n_candles:4d} bars | "
            f"{n_trades:3d} trades | "
            f"Sharpe {float(result.sharpe):+7.3f} | "
            f"WR {float(result.win_rate):5.1%} | "
            f"Ret {float(result.total_return):+7.3%} | "
            f"BUY {buys:3d} SKIP {skips:3d}"
        )
    else:
        print(
            f"  {symbol:12s} | {n_candles:4d} bars | "
            f"  0 trades | no metrics | "
            f"BUY {buys:3d} SKIP {skips:3d}"
        )


def main() -> None:
    """Run batch evaluation across all segments and symbols."""
    parser = argparse.ArgumentParser(description="Batch evaluation")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--cash", type=Decimal, default=Decimal(100_000))
    parser.add_argument("--output", default="results/2026-03-01-batch")
    parser.add_argument("--models-dir", default=None, help="Directory with trained ML models")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    output_dir = Path(args.output)

    all_results: list[dict[str, Any]] = []

    # Cache benchmark candles per market to avoid re-fetching
    benchmark_cache: dict[str, list[Any] | None] = {}

    models_dir = Path(args.models_dir) if args.models_dir else None

    for segment, symbols in UNIVERSE.items():
        market_id = "moex" if segment.startswith("ru_") else "us"
        print(f"\n{'=' * 70}")
        print(f"  SEGMENT: {segment} ({len(symbols)} symbols, market={market_id})")
        print(f"{'=' * 70}")

        # Fetch benchmark candles (SPY for US, EWZ for RU)
        bench_symbol = "EWZ" if segment.startswith("ru_") else "SPY"
        if bench_symbol not in benchmark_cache:
            try:
                bench_fetcher = CachingFetcher(YFinanceFetcher(market_id="us"))
                benchmark_cache[bench_symbol] = bench_fetcher.fetch_candles(
                    bench_symbol, start, end
                )
                n_bars = len(benchmark_cache[bench_symbol] or [])
                print(f"  Benchmark: {bench_symbol} ({n_bars} bars)")
            except Exception:
                benchmark_cache[bench_symbol] = None
                print(f"  Benchmark: {bench_symbol} (fetch failed)")
        bench_candles = benchmark_cache[bench_symbol]

        # Build strategies once per segment (Pairs needs peer candle fetching)
        seg_fetcher = CachingFetcher(YFinanceFetcher(market_id=market_id))
        strategies = _build_strategies(segment, seg_fetcher, start, end, models_dir)
        strat_names = [s.name for s in strategies]
        print(f"  Strategies: {', '.join(strat_names)}")

        for symbol in symbols:
            result = _run_single(
                symbol,
                segment,
                start,
                end,
                args.cash,
                output_dir,
                market_id,
                benchmark_candles=bench_candles,
                strategies=strategies,
            )
            if result:
                all_results.append(result)

    # Write consolidated summary
    consolidated_path = output_dir / "consolidated_summary.json"
    consolidated_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n{'=' * 70}")
    print(f"  DONE: {len(all_results)} successful evaluations")
    print(f"  Consolidated: {consolidated_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
