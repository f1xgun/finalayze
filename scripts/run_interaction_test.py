"""Interaction test runner -- executes A-only, B-only, A+B backtests and compares results.

Usage:
    python scripts/run_interaction_test.py \
        --experiment-a 2026-04-08-dual-momentum \
        --experiment-b 2026-04-08-ou-mean-rev \
        --segments ru_blue_chips
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.finalayze.core.experiment_manager import ExperimentManager


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run A/B/AB interaction test")
    parser.add_argument("--experiment-a", required=True, help="Experiment A ID")
    parser.add_argument("--experiment-b", required=True, help="Experiment B ID")
    parser.add_argument(
        "--segments", default=None, help="Comma-separated segment IDs (passthrough)"
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    return parser.parse_args()


def _run_hypothesis(
    hypothesis: str,
    run_name: str,
    name: str,
    description: str,
    extra_args: list[str],
) -> None:
    """Run a single backtest via subprocess with --hypothesis flag."""
    cmd = [
        sys.executable,
        "scripts/run_iteration.py",
        "--name",
        name,
        "--description",
        description,
        "--hypothesis",
        hypothesis,
        "--run-name",
        run_name,
        *extra_args,
    ]
    subprocess.run(cmd, check=True)  # noqa: S603 -- args from argparse, no user free-text


def _load_result(experiment_id: str, run_name: str) -> dict[str, Any]:
    """Load a result JSON from results/experiments/{experiment_id}/{run_name}.json."""
    path = Path("results/experiments") / experiment_id / f"{run_name}.json"
    return json.loads(path.read_text())


def _format_comparison_table(
    a: dict[str, Any],
    b: dict[str, Any],
    ab: dict[str, Any],
) -> str:
    """Format a markdown comparison table from three result dicts.

    Columns: Metric, A-only, B-only, A+B, Delta(A), Delta(B)
    """
    metrics = [
        ("WF Sharpe", "wf_sharpe"),
        ("Profit Factor", "profit_factor"),
        ("Max Drawdown", "wf_max_drawdown"),
        ("Trade Count", "trade_count"),
    ]
    lines = [
        "| Metric | A-only | B-only | A+B | Delta(A) | Delta(B) |",
        "|--------|--------|--------|-----|----------|----------|",
    ]
    for label, key in metrics:
        va = a.get(key, 0)
        vb = b.get(key, 0)
        vab = ab.get(key, 0)
        da = vab - va
        db = vab - vb
        if key == "trade_count":
            lines.append(
                f"| {label} | {va:d} | {vb:d} | {vab:d} | {da:+d} | {db:+d} |"
            )
        else:
            lines.append(
                f"| {label} | {va:.4f} | {vb:.4f} | {vab:.4f} | {da:+.4f} | {db:+.4f} |"
            )
    return "\n".join(lines)


def main() -> None:
    """Run A-only, B-only, A+B backtests and produce comparison table."""
    args = _parse_args()
    mgr = ExperimentManager()
    extra_args: list[str] = []
    if args.segments:
        extra_args.extend(["--segments", args.segments])
    extra_args.extend(["--start-date", args.start_date, "--end-date", args.end_date])

    exp_a = mgr.read_experiment(args.experiment_a)
    exp_b = mgr.read_experiment(args.experiment_b)

    # Create combined A+B experiment file with merged overrides from both
    combined_id = f"interaction-{args.experiment_a}-{args.experiment_b}"
    combined_overrides: dict[str, Any] = {}

    from scripts.run_iteration import _deep_merge  # noqa: PLC0415

    if exp_a.preset_overrides:
        for seg, ov in exp_a.preset_overrides.items():
            combined_overrides[seg] = _deep_merge(combined_overrides.get(seg, {}), ov)
    if exp_b.preset_overrides:
        for seg, ov in exp_b.preset_overrides.items():
            combined_overrides[seg] = _deep_merge(combined_overrides.get(seg, {}), ov)

    # Create temporary combined experiment so ExperimentManager can track it
    mgr.create_experiment(
        experiment_id=combined_id,
        hypothesis=f"Interaction: {exp_a.hypothesis} + {exp_b.hypothesis}",
        success_criteria=exp_a.success_criteria,
        preset_overrides=combined_overrides,
    )

    # Run A-only, B-only, A+B
    print("=== Running A-only ===")
    _run_hypothesis(
        args.experiment_a,
        "A-only",
        f"{args.experiment_a}-A-only",
        f"Interaction A-only: {exp_a.hypothesis}",
        extra_args,
    )
    print("=== Running B-only ===")
    _run_hypothesis(
        args.experiment_b,
        "B-only",
        f"{args.experiment_b}-B-only",
        f"Interaction B-only: {exp_b.hypothesis}",
        extra_args,
    )
    print("=== Running A+B ===")
    _run_hypothesis(
        combined_id,
        "AB",
        f"{combined_id}-AB",
        "Interaction A+B",
        extra_args,
    )

    # Load results and compare
    result_a = _load_result(args.experiment_a, "A-only")
    result_b = _load_result(args.experiment_b, "B-only")
    result_ab = _load_result(combined_id, "AB")

    table = _format_comparison_table(result_a, result_b, result_ab)
    print("\n=== Comparison ===\n")
    print(table)

    # Save comparison
    comp_path = (
        Path("results/experiments")
        / f"comparison-{args.experiment_a}-{args.experiment_b}.md"
    )
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp_path.write_text(
        f"# Interaction Test: {args.experiment_a} vs {args.experiment_b}\n\n{table}\n"
    )

    # Record verdict based on A+B results (primary)
    primary_metric = exp_a.success_criteria.metric
    if primary_metric in result_ab:
        mgr.record_verdict(combined_id, result_ab[primary_metric])
    print(f"\nComparison saved to {comp_path}")


if __name__ == "__main__":
    main()
