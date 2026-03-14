"""Generate a markdown validation report from sandbox cycle logs.

Reads structured JSONL entries produced by ValidationLogger during
the 5-day sandbox validation run, computes metrics, and writes a
markdown report with pass/fail assessment.

Usage:
    python scripts/generate_validation_report.py [--log-path X] [--output-path Y]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

# Project convention: scripts need sys.path adjustment for config/src imports
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.finalayze.core.validation_logger import (
    CycleLogEntry,
    ValidationLogger,
)

# Validation criteria thresholds
MIN_TRADING_DAYS = 5
MAX_DRAWDOWN_PCT = 5.0
MIN_ROUND_TRIP_TRADES = 10


class _Metrics(NamedTuple):
    trading_days: int
    total_cycles: int
    total_orders: int
    total_fills: int
    max_drawdown: float
    final_equity: float
    total_errors: int
    first_ts: datetime
    last_ts: datetime
    by_date: dict[str, list[CycleLogEntry]]


def _compute_metrics(entries: list[CycleLogEntry]) -> _Metrics:
    """Compute aggregate metrics from cycle log entries."""
    by_date: dict[str, list[CycleLogEntry]] = defaultdict(list)
    for entry in entries:
        date_str = entry.timestamp.strftime("%Y-%m-%d")
        by_date[date_str].append(entry)

    return _Metrics(
        trading_days=len(by_date),
        total_cycles=len(entries),
        total_orders=sum(e.orders_submitted for e in entries),
        total_fills=sum(e.orders_filled for e in entries),
        max_drawdown=max(e.drawdown_pct for e in entries),
        final_equity=entries[-1].equity_rub,
        total_errors=sum(e.errors_caught for e in entries),
        first_ts=entries[0].timestamp,
        last_ts=entries[-1].timestamp,
        by_date=dict(by_date),
    )


def _build_report_sections(m: _Metrics) -> tuple[list[str], bool]:
    """Build markdown sections and return (lines, all_pass)."""
    days_pass = m.trading_days >= MIN_TRADING_DAYS
    dd_pass = m.max_drawdown < MAX_DRAWDOWN_PCT
    trades_pass = m.total_fills >= MIN_ROUND_TRIP_TRADES
    errors_pass = m.total_errors == 0
    all_pass = days_pass and dd_pass and trades_pass and errors_pass

    uptime = m.last_ts - m.first_ts
    now_str = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "# Sandbox Validation Report",
        "",
        f"Generated: {now_str}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Validation Period | {m.first_ts:%Y-%m-%d %H:%M} -- {m.last_ts:%Y-%m-%d %H:%M} |",
        f"| Uptime | {uptime} |",
        f"| Trading Days | {m.trading_days} |",
        f"| Total Cycles | {m.total_cycles} |",
        f"| Orders Submitted | {m.total_orders} |",
        f"| Orders Filled | {m.total_fills} |",
        f"| Max Drawdown | {m.max_drawdown:.2f}% |",
        f"| Final Equity (RUB) | {m.final_equity:,.2f} |",
        f"| Total Errors | {m.total_errors} |",
        "",
        "## Criteria Assessment",
        "",
        "| Criterion | Required | Actual | Result |",
        "|-----------|----------|--------|--------|",
        f"| Trading Days | >= {MIN_TRADING_DAYS} | {m.trading_days}"
        f" | {'PASS' if days_pass else 'FAIL'} |",
        f"| Max Drawdown | < {MAX_DRAWDOWN_PCT}% | {m.max_drawdown:.2f}%"
        f" | {'PASS' if dd_pass else 'FAIL'} |",
        f"| Round-Trip Trades | >= {MIN_ROUND_TRIP_TRADES} | {m.total_fills}"
        f" | {'PASS' if trades_pass else 'FAIL'} |",
        f"| Critical Errors | 0 | {m.total_errors}"
        f" | {'PASS' if errors_pass else 'FAIL'} |",
        "",
    ]

    # Per-Day Breakdown
    lines.append("## Per-Day Breakdown")
    lines.append("")
    lines.append("| Date | Cycles | Orders | Fills | Equity (RUB) | Max DD% | Errors |")
    lines.append("|------|--------|--------|-------|-------------|---------|--------|")
    for date_str in sorted(m.by_date):
        day = m.by_date[date_str]
        lines.append(
            f"| {date_str} | {len(day)} | {sum(e.orders_submitted for e in day)}"
            f" | {sum(e.orders_filled for e in day)} | {day[-1].equity_rub:,.2f}"
            f" | {max(e.drawdown_pct for e in day):.2f}% | {sum(e.errors_caught for e in day)} |"
        )
    lines.append("")

    # Overall Verdict
    lines.append("## Overall Verdict")
    lines.append("")
    if all_pass:
        lines.append("**PASS** -- All validation criteria met.")
    else:
        failed = _collect_failures(m, days_pass, dd_pass, trades_pass, errors_pass)
        lines.append(f"**FAIL** -- Failed criteria: {', '.join(failed)}")
    lines.append("")

    return lines, all_pass


def _collect_failures(
    m: _Metrics,
    days_pass: bool,
    dd_pass: bool,
    trades_pass: bool,
    errors_pass: bool,
) -> list[str]:
    """Collect human-readable failure descriptions."""
    failed: list[str] = []
    if not days_pass:
        failed.append(f"Trading Days ({m.trading_days} < {MIN_TRADING_DAYS})")
    if not dd_pass:
        failed.append(f"Max Drawdown ({m.max_drawdown:.2f}% >= {MAX_DRAWDOWN_PCT}%)")
    if not trades_pass:
        failed.append(f"Round-Trip Trades ({m.total_fills} < {MIN_ROUND_TRIP_TRADES})")
    if not errors_pass:
        failed.append(f"Critical Errors ({m.total_errors} > 0)")
    return failed


def generate_report(log_path: Path, output_path: Path) -> bool:
    """Generate a markdown validation report.

    Args:
        log_path: Path to the cycles.jsonl file.
        output_path: Path for the output markdown report.

    Returns:
        True if all validation criteria pass, False otherwise.
    """
    logger = ValidationLogger(log_path)
    entries = logger.get_entries()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not entries:
        output_path.write_text(_build_empty_report())
        return False

    metrics = _compute_metrics(entries)
    lines, all_pass = _build_report_sections(metrics)
    output_path.write_text("\n".join(lines))
    return all_pass


def _build_empty_report() -> str:
    """Build a FAIL report when there is no data."""
    now_str = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Sandbox Validation Report",
        "",
        f"Generated: {now_str}",
        "",
        "## Summary",
        "",
        "No data available. The cycle log file is empty or does not exist.",
        "",
        "## Criteria Assessment",
        "",
        "No data to assess.",
        "",
        "## Per-Day Breakdown",
        "",
        "No data.",
        "",
        "## Overall Verdict",
        "",
        "**FAIL** -- No data collected during validation period.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate sandbox validation report from cycle logs."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("results/validation/cycles.jsonl"),
        help="Path to the JSONL cycle log file (default: results/validation/cycles.jsonl)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/validation/VALIDATION-REPORT.md"),
        help="Path for the output report (default: results/validation/VALIDATION-REPORT.md)",
    )
    args = parser.parse_args()

    passed = generate_report(args.log_path, args.output_path)
    status = "PASS" if passed else "FAIL"
    print(f"Validation report generated: {args.output_path}")
    print(f"Overall verdict: {status}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
