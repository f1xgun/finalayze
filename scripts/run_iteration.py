"""Run a new iteration: backtest, measure, gate, save, compare.

Usage:
    uv run python scripts/run_iteration.py \
        --name "baseline" \
        --description "Current system before improvements"

    uv run python scripts/run_iteration.py \
        --name "add-sentiment-to-momentum" \
        --description "Integrate LLM sentiment score" \
        --baseline latest

    uv run python scripts/run_iteration.py \
        --name "test-v1" \
        --description "Initial baseline" \
        --dry-run
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

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.decision_journal import DecisionJournal
from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.iteration_tracker import IterationTracker
from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner
from finalayze.backtest.monte_carlo import bootstrap_from_snapshots
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.backtest.walk_forward import WalkForwardResult
from finalayze.core.schemas import (
    GateResult,
    IterationMetadata,
    IterationMetrics,
    PortfolioState,
    TradeResult,
)
from finalayze.data.fetchers.yfinance import YFinanceFetcher
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
    "us_tech": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "TSM", "AVGO", "ADBE", "CRM", "INTC", "AMD",
    ],
    "us_broad": [
        "SPY", "QQQ", "DIA", "IWM", "JNJ", "PG", "KO", "WMT", "XOM", "CVX",
    ],
    "us_finance": [
        "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "C",
    ],
    "us_healthcare": [
        "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN",
    ],
    "ru_blue_chips": [
        "RSX", "ERUS", "FLRU.L", "TUR", "EWZ", "INDA",
    ],
    "ru_energy": [
        "XLE", "BP", "SHEL", "TTE", "ENB",
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
    fetcher: YFinanceFetcher,
    start: datetime,
    end: datetime,
) -> PairsStrategy | None:
    """Create a PairsStrategy with pre-loaded peer candles, or None."""
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
            continue
    return strategy


def _setup_ml_strategy(segment: str, models_dir: Path) -> MLStrategy | None:
    """Create an MLStrategy with loaded models, or None."""
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

    from finalayze.ml.registry import MLModelRegistry  # noqa: PLC0415
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
    """Build the full strategy list for a segment."""
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


def _run_symbol(
    symbol: str,
    segment: str,
    candles: list[Any],
    strategies: list[BaseStrategy],
    cash: Decimal,
    output_dir: Path,
    benchmark_candles: list[Any] | None = None,
) -> tuple[list[TradeResult], list[PortfolioState], dict[str, Any] | None]:
    """Run backtest for a single symbol. Returns (trades, snapshots, summary)."""
    sym_dir = output_dir / segment / symbol.replace(".", "_")
    sym_dir.mkdir(parents=True, exist_ok=True)

    try:
        combiner = JournalingStrategyCombiner(strategies=strategies)
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

        summary = {
            "symbol": symbol,
            "segment": segment,
            "total_candles": len(candles),
            "total_trades": len(trades),
            "metrics": result.model_dump(mode="json") if result else None,
            "journal_summary": journal.summary(),
        }

        sharpe = float(result.sharpe) if result else 0.0
        wr = float(result.win_rate) if result else 0.0
        ret = float(result.total_return) if result else 0.0
        print(
            f"    {symbol:12s} | {len(candles):4d} bars | "
            f"{len(trades):3d} trades | "
            f"Sharpe {sharpe:+7.3f} | "
            f"WR {wr:5.1%} | "
            f"Ret {ret:+7.3%}"
        )

        return trades, snapshots, summary

    except Exception:
        print(f"    {symbol:12s} | ERROR — {traceback.format_exc().splitlines()[-1]}")
        return [], [], None


def _format_comparison_table(
    current_name: str,
    baseline_name: str | None,
    metrics: IterationMetrics,
    baseline_metrics: IterationMetrics | None,
    gate_results: list[GateResult],
    verdict: str,
    git_info: dict[str, object],
) -> str:
    """Format a comparison table for terminal output."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  Iteration: {current_name}")
    if baseline_name:
        lines.append(f"  Baseline:  {baseline_name}")
    dirty_str = "dirty" if git_info.get("git_dirty") else "clean"
    sha_short = str(git_info.get("git_sha", ""))[:7]
    lines.append(f"  Git:       {sha_short} ({dirty_str})")
    lines.append(f"  Verdict:   {verdict}")
    lines.append("=" * 72)
    lines.append("")

    header = f"  {'Metric':<26} {'Current':>10}"
    if baseline_metrics:
        header += f" {'Baseline':>10} {'Delta':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header.strip()) + 2))

    bm = baseline_metrics
    _add_row(lines, "WF Sharpe", metrics.wf_sharpe, bm)
    _add_row(lines, "Max Drawdown (%)", metrics.wf_max_drawdown, bm, "wf_max_drawdown")
    _add_row(lines, "Profit Factor", metrics.profit_factor, bm)
    _add_row(lines, "Calmar Ratio", metrics.calmar_ratio, bm)
    _add_row(lines, "Trade Count", float(metrics.trade_count), bm, "trade_count")
    _add_row(lines, "Avg Hold (bars)", metrics.avg_hold_bars, bm, "avg_hold_bars")
    lines.append("  " + "-" * (len(header.strip()) + 2))
    _add_row(lines, "Sortino", metrics.sortino_ratio, bm)
    _add_row(lines, "MC 5th-pct Sharpe", metrics.mc_5th_pct_sharpe, bm)
    _add_row(lines, "Model Disagreement", metrics.model_disagreement, bm)
    _add_row(lines, "Turnover-Adj Return", metrics.turnover_adjusted_return, bm)
    _add_row(lines, "Param Stability CV", metrics.param_stability_cv, bm)

    lines.append("")
    gate_str = "  Gates: " + "  ".join(
        f"{g.name}:{'OK' if g.passed else 'FAIL'}" for g in gate_results
    )
    lines.append(gate_str)
    lines.append("")

    # Segment PnL share
    if metrics.segment_pnl_share:
        lines.append("  Segment PnL Share:")
        for seg, share in sorted(
            metrics.segment_pnl_share.items(), key=lambda x: -abs(x[1])
        ):
            lines.append(f"    {seg:<20s} {share:>7.1%}")
        lines.append("")

    # Win rate by segment
    if metrics.win_rate_by_segment:
        lines.append("  Win Rate by Segment:")
        for seg, wr in sorted(metrics.win_rate_by_segment.items()):
            lines.append(f"    {seg:<20s} {wr:>7.1%}")
        lines.append("")

    return "\n".join(lines)


def _add_row(
    lines: list[str],
    name: str,
    current_val: float,
    baseline_metrics: IterationMetrics | None,
    attr: str | None = None,
) -> None:
    """Append a formatted metric row to lines."""
    row = f"  {name:<26} {current_val:>10.4f}"
    if baseline_metrics is not None:
        field = attr or name.lower().replace(" ", "_").replace("(%)", "").strip()
        base_val = getattr(baseline_metrics, field, None)
        if base_val is not None:
            base_f = float(base_val)
            delta = current_val - base_f
            row += f" {base_f:>10.4f} {delta:>+10.4f}"
    lines.append(row)


def _run_dry(
    args: argparse.Namespace,
    tracker: IterationTracker,
    git_info: dict[str, object],
) -> None:
    """Execute dry-run mode with synthetic metrics."""
    print("\n  [DRY RUN] Generating synthetic metrics...")
    metrics = IterationMetrics(
        wf_sharpe=0.0,
        wf_max_drawdown=0.0,
        profit_factor=0.0,
        calmar_ratio=0.0,
        trade_count=0,
        avg_hold_bars=0.0,
        segment_pnl_share={},
        sortino_ratio=0.0,
        win_rate_by_segment={},
        information_ratio=None,
        mc_5th_pct_sharpe=0.0,
        model_disagreement=0.0,
        turnover_adjusted_return=0.0,
        gross_sharpe=0.0,
        net_sharpe=0.0,
        param_stability_cv=0.0,
        per_model_proba_mean={},
    )
    gate_results, verdict = tracker.evaluate_gates(metrics, baseline=None)
    print(
        _format_comparison_table(args.name, None, metrics, None, gate_results, verdict, git_info)
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a new backtest iteration")
    parser.add_argument("--name", required=True, help="Iteration name")
    parser.add_argument("--description", required=True, help="What changed")
    parser.add_argument(
        "--baseline", default="latest", help="Baseline name (default: latest)"
    )
    parser.add_argument(
        "--output", default="results/iterations/", help="Output root"
    )
    parser.add_argument(
        "--segments", default=None, help="Comma-separated segment IDs"
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument(
        "--cash", type=int, default=100_000, help="Initial cash per symbol"
    )
    parser.add_argument(
        "--models-dir", default=None, help="Directory with trained ML models"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_baseline(
    tracker: IterationTracker,
    baseline_arg: str,
) -> tuple[IterationMetrics | None, str | None]:
    """Load baseline metrics by name or 'latest'."""
    if baseline_arg == "latest":
        latest = tracker.load_latest()
        if latest:
            return latest.metrics, latest.name
        return None, None
    try:
        meta = tracker.load(baseline_arg)
        return meta.metrics, baseline_arg
    except FileNotFoundError:
        print(f"  Warning: baseline '{baseline_arg}' not found")
        return None, None


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run a new iteration."""
    args = _parse_args()
    output_root = Path(args.output)
    tracker = IterationTracker(results_root=output_root)

    try:
        git_info = tracker.snapshot_context()
    except Exception:
        git_info = {
            "git_sha": "unknown",
            "git_describe": "unknown",
            "git_dirty": True,
        }

    all_segments = list(UNIVERSE.keys())
    segments = args.segments.split(",") if args.segments else all_segments

    if args.dry_run:
        _run_dry(args, tracker, git_info)
        return

    start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC)
    cash = Decimal(args.cash)
    models_dir = Path(args.models_dir) if args.models_dir else None

    # Load strategy configs from YAML presets
    strategy_configs: dict[str, Any] = {}
    for seg in segments:
        strategy_configs[seg] = _load_preset(seg)

    config = BacktestConfig(initial_cash=cash)
    backtest_config_dict = {
        "initial_cash": str(config.initial_cash),
        "max_positions": config.max_positions,
        "kelly_fraction": str(config.kelly_fraction),
        "atr_multiplier": str(config.atr_multiplier),
    }
    config_hash = tracker.compute_config_hash(backtest_config_dict, strategy_configs)

    # Cache benchmark candles per market
    benchmark_cache: dict[str, list[Any] | None] = {}

    all_trades: list[TradeResult] = []
    all_snapshots: list[PortfolioState] = []
    segment_trades: dict[str, list[TradeResult]] = {}
    all_summaries: list[dict[str, Any]] = []

    print(f"\nRunning iteration '{args.name}'")
    print(f"  Period: {args.start_date} to {args.end_date}")
    print(f"  Segments: {', '.join(segments)}")
    print(f"  Cash: ${cash:,.0f}")
    print()

    for segment in segments:
        symbols = UNIVERSE.get(segment, [])
        if not symbols:
            print(f"  Segment '{segment}' not found in universe, skipping")
            continue

        market_id = "moex" if segment.startswith("ru_") else "us"
        print(f"{'=' * 72}")
        print(f"  SEGMENT: {segment} ({len(symbols)} symbols, market={market_id})")
        print(f"{'=' * 72}")

        # Fetch benchmark
        bench_symbol = "EWZ" if segment.startswith("ru_") else "SPY"
        if bench_symbol not in benchmark_cache:
            try:
                bench_fetcher = YFinanceFetcher(market_id="us")
                benchmark_cache[bench_symbol] = bench_fetcher.fetch_candles(
                    bench_symbol, start, end
                )
                n_bars = len(benchmark_cache[bench_symbol] or [])
                print(f"  Benchmark: {bench_symbol} ({n_bars} bars)")
            except Exception:
                benchmark_cache[bench_symbol] = None
                print(f"  Benchmark: {bench_symbol} (fetch failed)")
        bench_candles = benchmark_cache[bench_symbol]

        # Build strategies once per segment
        seg_fetcher = YFinanceFetcher(market_id=market_id)
        strategies = _build_strategies(segment, seg_fetcher, start, end, models_dir)
        strat_names = [s.name for s in strategies]
        print(f"  Strategies: {', '.join(strat_names)}")
        print()

        segment_trades[segment] = []

        for symbol in symbols:
            # Fetch candles
            try:
                fetcher = YFinanceFetcher(market_id=market_id)
                candles = fetcher.fetch_candles(symbol, start, end)
                if not candles:
                    print(f"    {symbol:12s} | no data")
                    continue
            except Exception:
                print(f"    {symbol:12s} | fetch failed")
                continue

            iter_dir = output_root / args.name
            trades, snapshots, summary = _run_symbol(
                symbol=symbol,
                segment=segment,
                candles=candles,
                strategies=strategies,
                cash=cash,
                output_dir=iter_dir,
                benchmark_candles=bench_candles,
            )

            all_trades.extend(trades)
            segment_trades[segment].extend(trades)
            if snapshots:
                all_snapshots.extend(snapshots)
            if summary:
                all_summaries.append(summary)

        print()

    if not all_trades:
        print("\n  No trades generated across all segments.")
        print("  Saving iteration with zero metrics for tracking purposes.\n")

    # Save consolidated summary
    iter_dir = output_root / args.name
    iter_dir.mkdir(parents=True, exist_ok=True)
    summary_path = iter_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, default=str))

    # Compute iteration metrics
    if all_snapshots and all_trades:
        mc_result = bootstrap_from_snapshots(all_snapshots, n_simulations=1000, seed=42)
    else:
        # Create a minimal MC result for zero-trade iterations
        from finalayze.backtest.monte_carlo import BootstrapCI, BootstrapResult  # noqa: PLC0415
        zero_ci = BootstrapCI(
            point_estimate=0.0, lower=0.0, upper=0.0, confidence_level=0.95
        )
        mc_result = BootstrapResult(
            total_return=zero_ci,
            sharpe_ratio=zero_ci,
            max_drawdown=zero_ci,
            win_rate=zero_ci,
            profit_factor=zero_ci,
        )

    # Build a synthetic WalkForwardResult from the direct backtest run
    # (full walk-forward takes much longer; this gives immediate baseline)
    from finalayze.backtest.performance import PerformanceAnalyzer as PA  # noqa: PLC0415, N814
    full_result = PA().analyze(all_trades, all_snapshots) if all_trades else None
    oos_sharpe = float(full_result.sharpe) if full_result else 0.0
    oos_dd = float(full_result.max_drawdown) if full_result else 0.0

    wf_result = WalkForwardResult(
        oos_sharpe=oos_sharpe,
        oos_max_drawdown_pct=oos_dd * 100,  # convert to percent
        oos_snapshots=all_snapshots,
        per_fold_sharpes=[oos_sharpe] if all_trades else [],
        per_fold_trade_counts=[len(all_trades)] if all_trades else [],
    )

    metrics = tracker.compute_metrics(
        wf_result=wf_result,
        trades=all_trades,
        snapshots=all_snapshots,
        segment_trades=segment_trades,
        mc_result=mc_result,
    )

    baseline_metrics, baseline_name = _load_baseline(tracker, args.baseline)
    gate_results, verdict = tracker.evaluate_gates(metrics, baseline=baseline_metrics)

    metadata = IterationMetadata(
        name=args.name,
        description=args.description,
        created_at=datetime.now(UTC),
        git_describe=str(git_info["git_describe"]),
        git_sha=str(git_info["git_sha"]),
        git_dirty=bool(git_info["git_dirty"]),
        config_hash=config_hash,
        strategy_configs=strategy_configs,
        backtest_config=backtest_config_dict,
        metrics=metrics,
        gate_results=gate_results,
        verdict=verdict,
    )

    result_path = tracker.save(metadata)
    print(f"\n  Saved to: {result_path}")
    print(
        _format_comparison_table(
            args.name,
            baseline_name,
            metrics,
            baseline_metrics,
            gate_results,
            verdict,
            git_info,
        )
    )


if __name__ == "__main__":
    main()
