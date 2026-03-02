"""Compare two iterations side-by-side.

Usage:
    uv run python scripts/compare_iterations.py baseline-name current-name
    uv run python scripts/compare_iterations.py --output results/iterations/ v1 v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.backtest.iteration_tracker import IterationTracker


def _format_delta(val: float) -> str:
    """Format a delta value with sign and 4 decimal places."""
    return f"{val:+.4f}"


def main() -> None:
    """Compare two iterations and print formatted table."""
    parser = argparse.ArgumentParser(description="Compare two backtest iterations")
    parser.add_argument("baseline", help="Baseline iteration name")
    parser.add_argument("current", help="Current iteration name")
    parser.add_argument("--output", default="results/iterations/", help="Iterations root directory")
    args = parser.parse_args()

    output_root = Path(args.output)
    tracker = IterationTracker(results_root=output_root)

    try:
        comparison = tracker.compare(args.current, args.baseline)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    current_meta = tracker.load(args.current)
    baseline_meta = tracker.load(args.baseline)
    cm = current_meta.metrics
    bm = baseline_meta.metrics

    print()
    print(f"  Comparison: {args.current} vs {args.baseline}")
    print(f"  Verdict:    {comparison.verdict}")
    print()

    header = f"  {'Metric':<26} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Flag':>6}"
    print(header)
    print("  " + "-" * (len(header.strip())))

    def _row(
        name: str,
        base_val: float,
        curr_val: float,
        flag: str = "",
    ) -> None:
        delta = curr_val - base_val
        print(f"  {name:<26} {base_val:>10.4f} {curr_val:>10.4f} {delta:>+10.4f} {flag:>6}")

    _row("WF Sharpe", bm.wf_sharpe, cm.wf_sharpe)
    _row("Max Drawdown (%)", bm.wf_max_drawdown, cm.wf_max_drawdown)
    _row("Profit Factor", bm.profit_factor, cm.profit_factor)
    _row("Calmar Ratio", bm.calmar_ratio, cm.calmar_ratio)
    _row("Trade Count", float(bm.trade_count), float(cm.trade_count))
    _row("Avg Hold Bars", bm.avg_hold_bars, cm.avg_hold_bars)
    _row("Sortino", bm.sortino_ratio, cm.sortino_ratio)
    _row("MC 5th-pct Sharpe", bm.mc_5th_pct_sharpe, cm.mc_5th_pct_sharpe)
    _row("Model Disagreement", bm.model_disagreement, cm.model_disagreement)
    _row("Param Stability CV", bm.param_stability_cv, cm.param_stability_cv)
    _row("Gross Sharpe", bm.gross_sharpe, cm.gross_sharpe)
    _row("Net Sharpe", bm.net_sharpe, cm.net_sharpe)

    print()
    gate_str = "  Gates: " + "  ".join(
        f"{g.name} {'PASS' if g.passed else 'FAIL'}" for g in comparison.gate_results
    )
    print(gate_str)
    print()


if __name__ == "__main__":
    main()
