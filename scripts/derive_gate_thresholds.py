"""Derive gate thresholds from backtest history.

Reads ``results/iterations/history.jsonl`` and computes data-driven thresholds
for the go/no-go gate.  Outputs ``config/gate_thresholds.yaml``.

Usage::

    python scripts/derive_gate_thresholds.py \\
        [--history results/iterations/history.jsonl] \\
        [--output config/gate_thresholds.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_history(path: Path) -> list[dict]:
    """Read history.jsonl, returning list of dicts."""
    entries: list[dict] = []
    with path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


def _derive_thresholds(entries: list[dict]) -> dict:
    """Compute gate thresholds from backtest entries.

    Data-driven (from history.jsonl):
    - max_drawdown_pct: p90 of wf_max_drawdown
    - min_trades_5d: p10 of trade_count / ~6 walk-forward periods, floor at 5

    Engineering defaults for everything else.
    """
    wf_max_drawdowns = [
        e["wf_max_drawdown"]
        for e in entries
        if "wf_max_drawdown" in e and e["wf_max_drawdown"] is not None
    ]
    trade_counts = [
        e["trade_count"] for e in entries if "trade_count" in e and e["trade_count"] is not None
    ]

    # Data-driven thresholds
    if wf_max_drawdowns:
        max_dd_threshold = round(float(np.percentile(wf_max_drawdowns, 90)), 2)
        max_dd_source = "p90 of wf_max_drawdown from history.jsonl"
    else:
        max_dd_threshold = 5.0
        max_dd_source = "engineering default (no history data)"

    walk_forward_periods = 6  # typical WF config: 12mo train + 6mo test over 3yr
    if trade_counts:
        trades_per_period = float(np.percentile(trade_counts, 10)) / walk_forward_periods
        min_trades_threshold = max(5, int(trades_per_period))
        min_trades_source = "p10 of trade_count / walk-forward periods from history.jsonl"
    else:
        min_trades_threshold = 5
        min_trades_source = "engineering default (no history data)"

    return {
        "gate": {
            "min_sandbox_days": 5,
            "criteria": {
                "uptime_pct": {
                    "threshold": 99.0,
                    "critical": True,
                    "source": "engineering default",
                },
                "fill_rate_pct": {
                    "threshold": 95.0,
                    "critical": True,
                    "source": "engineering default",
                },
                "max_drawdown_pct": {
                    "threshold": max_dd_threshold,
                    "critical": True,
                    "source": max_dd_source,
                },
                "min_trades_5d": {
                    "threshold": float(min_trades_threshold),
                    "critical": False,
                    "source": min_trades_source,
                },
                "signal_frequency_per_day": {
                    "threshold": 1.0,
                    "critical": False,
                    "source": "engineering default",
                },
                "critical_errors_pct": {
                    "threshold": 1.0,
                    "critical": True,
                    "source": "engineering default",
                },
                "max_slippage_bps": {
                    "threshold": 50.0,
                    "critical": False,
                    "source": "engineering default",
                },
                "signal_divergence_pct": {
                    "threshold": 50.0,
                    "critical": False,
                    "source": "engineering default",
                },
            },
        }
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Derive go/no-go gate thresholds from backtest history."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("results/iterations/history.jsonl"),
        help="Path to history.jsonl (default: results/iterations/history.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/gate_thresholds.yaml"),
        help="Output YAML path (default: config/gate_thresholds.yaml)",
    )
    args = parser.parse_args(argv)

    if not args.history.exists():
        print(f"ERROR: History file not found: {args.history}", file=sys.stderr)
        sys.exit(1)

    entries = _load_history(args.history)
    print(f"Loaded {len(entries)} entries from {args.history}")

    thresholds = _derive_thresholds(entries)

    # Write YAML
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.dump(thresholds, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Wrote thresholds to {args.output}")

    # Print summary
    criteria = thresholds["gate"]["criteria"]
    print("\n--- Threshold Summary ---")
    print(f"min_sandbox_days: {thresholds['gate']['min_sandbox_days']}")
    for name, cfg in criteria.items():
        derived = "DERIVED" if "history.jsonl" in cfg["source"] else "DEFAULT"
        critical = "critical" if cfg["critical"] else "non-critical"
        print(f"  {name}: {cfg['threshold']} ({critical}) [{derived}]")


if __name__ == "__main__":
    main()
