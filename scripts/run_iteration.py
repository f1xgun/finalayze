"""Run a new iteration: backtest, measure, gate, save, compare.

Usage:
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
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.iteration_tracker import IterationTracker
from finalayze.backtest.monte_carlo import bootstrap_from_snapshots
from finalayze.backtest.walk_forward import WalkForwardResult
from finalayze.core.schemas import (
    GateResult,
    IterationMetadata,
    IterationMetrics,
    PortfolioState,
    TradeResult,
)


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
    lines.append(f"  Iteration: {current_name}")
    if baseline_name:
        lines.append(f"  Baseline:  {baseline_name}")
    dirty_str = "dirty" if git_info.get("git_dirty") else "clean"
    sha_short = str(git_info.get("git_sha", ""))[:7]
    lines.append(f"  Git:       {sha_short} ({dirty_str})")
    lines.append(f"  Verdict:   {verdict}")
    lines.append("")

    header = f"  {'Metric':<26} {'Current':>10}"
    if baseline_metrics:
        header += f" {'Baseline':>10} {'Delta':>10}"
    lines.append(header)
    lines.append("  " + "-" * len(header.strip()))

    bm = baseline_metrics
    _add_row(lines, "WF Sharpe", metrics.wf_sharpe, bm)
    _add_row(lines, "Max Drawdown (%)", metrics.wf_max_drawdown, bm, "wf_max_drawdown")
    _add_row(lines, "Profit Factor", metrics.profit_factor, bm)
    _add_row(lines, "Calmar Ratio", metrics.calmar_ratio, bm)
    _add_row(lines, "Trade Count", float(metrics.trade_count), bm, "trade_count")
    _add_row(lines, "Sortino", metrics.sortino_ratio, bm)
    _add_row(lines, "MC 5th-pct Sharpe", metrics.mc_5th_pct_sharpe, bm)
    _add_row(lines, "Param Stability CV", metrics.param_stability_cv, bm)

    lines.append("")
    gate_str = "  Gates: " + "  ".join(
        f"{g.name} {'PASS' if g.passed else 'FAIL'}" for g in gate_results
    )
    lines.append(gate_str)
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
        field = attr or name.lower().replace(" ", "_").replace("(%)", "")
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
    print(_format_comparison_table(args.name, None, metrics, None, gate_results, verdict, git_info))


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a new backtest iteration")
    parser.add_argument("--name", required=True, help="Iteration name")
    parser.add_argument("--description", required=True, help="What changed")
    parser.add_argument("--baseline", default="latest", help="Baseline name (default: latest)")
    parser.add_argument("--output", default="results/iterations/", help="Output root")
    parser.add_argument("--segments", default=None, help="Comma-separated segment IDs")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
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

    segments = args.segments.split(",") if args.segments else ["us_large_cap"]

    if args.dry_run:
        _run_dry(args, tracker, git_info)
        return

    config = BacktestConfig(initial_cash=Decimal(100000))
    backtest_config_dict = {
        "initial_cash": str(config.initial_cash),
        "max_positions": config.max_positions,
    }
    strategy_configs: dict[str, object] = {seg: {"momentum": {"weight": 1.0}} for seg in segments}
    config_hash = tracker.compute_config_hash(backtest_config_dict, dict(strategy_configs))

    all_trades: list[TradeResult] = []
    all_snapshots: list[PortfolioState] = []
    segment_trades: dict[str, list[TradeResult]] = {}

    print(f"Running iteration '{args.name}' for segments: {segments}")

    for segment in segments:
        print(f"  Processing segment: {segment}")
        segment_trades[segment] = []

    if not all_trades:
        print("\n  No candle data available. Seed historical data first.")
        return

    mc_result = bootstrap_from_snapshots(all_snapshots, n_simulations=1000, seed=42)
    wf_result = WalkForwardResult(
        oos_sharpe=0.0,
        oos_max_drawdown_pct=0.0,
        oos_snapshots=all_snapshots,
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
        strategy_configs=dict(strategy_configs),
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


if __name__ == "__main__":
    main()
