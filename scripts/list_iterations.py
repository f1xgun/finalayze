"""List all iterations with verdict and key metrics.

Usage:
    uv run python scripts/list_iterations.py
    uv run python scripts/list_iterations.py --verdict PASS
    uv run python scripts/list_iterations.py --output results/iterations/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.backtest.iteration_tracker import IterationTracker


def main() -> None:
    """List iterations from history.jsonl."""
    parser = argparse.ArgumentParser(description="List backtest iterations")
    parser.add_argument(
        "--verdict",
        choices=["PASS", "WARN", "REJECT"],
        default=None,
        help="Filter by verdict",
    )
    parser.add_argument("--output", default="results/iterations/", help="Iterations root directory")
    args = parser.parse_args()

    output_root = Path(args.output)
    tracker = IterationTracker(results_root=output_root)
    iterations = tracker.list_iterations()

    if not iterations:
        print("No iterations found.")
        return

    # Filter by verdict if specified
    if args.verdict:
        iterations = [it for it in iterations if it.get("verdict") == args.verdict]

    if not iterations:
        print(f"No iterations with verdict '{args.verdict}' found.")
        return

    # Print header
    header = (
        f"  {'Name':<30} {'Date':<22} {'SHA':>8} "
        f"{'Verdict':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>8}"
    )
    print()
    print(header)
    print("  " + "-" * (len(header.strip())))

    for it in iterations:
        name = it.get("name", "?")[:30]
        created = it.get("created_at", "?")[:19]
        sha = it.get("git_sha", "?")[:7]
        verdict = it.get("verdict", "?")
        sharpe = it.get("wf_sharpe", 0.0)
        max_dd = it.get("wf_max_drawdown", 0.0)
        trades = it.get("trade_count", 0)
        print(
            f"  {name:<30} {created:<22} {sha:>8} "
            f"{verdict:>8} {sharpe:>8.4f} {max_dd:>8.2f} {trades:>8}"
        )

    print()
    print(f"  Total: {len(iterations)} iteration(s)")
    print()


if __name__ == "__main__":
    main()
