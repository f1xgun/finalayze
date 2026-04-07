"""Daily post-market data collection for autonomous analysis pipeline.

Collects trading day data from cycle logs, sandbox metrics, and iteration
history into a structured JSON report for agent analysis.

Usage:
    uv run python scripts/daily_review.py [--date 2026-04-07] [--output results/daily/]
    uv run python scripts/daily_review.py --collect   # today's data
    uv run python scripts/daily_review.py --summary   # print last 7 days summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.finalayze.core.validation_logger import CycleLogEntry, ValidationLogger

CYCLE_LOG_PATH = Path("results/validation/cycles.jsonl")
ITERATION_HISTORY_PATH = Path("results/iterations/history.jsonl")


@dataclass
class DailySnapshot:
    """Aggregated metrics for a single trading day."""

    date: str
    total_cycles: int
    equity_cycles: int
    bond_cycles: int
    signals_generated: int
    orders_submitted: int
    orders_filled: int
    fill_rate_pct: float
    errors_caught: int
    max_drawdown_pct: float
    final_equity: float
    max_circuit_breaker_level: int
    avg_duration_ms: float
    instruments_processed: int


@dataclass
class DailyReport:
    """Full daily review data package for agent analysis."""

    report_date: str
    generated_at: str
    snapshot: DailySnapshot | None
    rolling_7d: dict
    recent_iterations: list[dict]
    anomalies: list[str]
    data_quality: dict


def _load_cycle_entries(log_path: Path) -> list[CycleLogEntry]:
    """Load all cycle log entries from JSONL."""
    logger = ValidationLogger(log_path=log_path)
    return logger.get_entries()


def _filter_by_date(entries: list[CycleLogEntry], target_date: date) -> list[CycleLogEntry]:
    """Filter entries to a specific date."""
    return [e for e in entries if e.timestamp.date() == target_date]


def _compute_snapshot(entries: list[CycleLogEntry], target_date: date) -> DailySnapshot | None:
    """Compute aggregated metrics for a single day."""
    day_entries = _filter_by_date(entries, target_date)
    if not day_entries:
        return None

    equity_entries = [e for e in day_entries if e.cycle_type == "equity"]
    bond_entries = [e for e in day_entries if e.cycle_type == "bond"]

    total_orders = sum(e.orders_submitted for e in day_entries)
    total_fills = sum(e.orders_filled for e in day_entries)
    fill_rate = (total_fills / total_orders * 100) if total_orders > 0 else 0.0

    return DailySnapshot(
        date=target_date.isoformat(),
        total_cycles=len(day_entries),
        equity_cycles=len(equity_entries),
        bond_cycles=len(bond_entries),
        signals_generated=sum(e.signals_generated for e in day_entries),
        orders_submitted=total_orders,
        orders_filled=total_fills,
        fill_rate_pct=round(fill_rate, 1),
        errors_caught=sum(e.errors_caught for e in day_entries),
        max_drawdown_pct=round(max(e.drawdown_pct for e in day_entries), 4),
        final_equity=day_entries[-1].equity_rub,
        max_circuit_breaker_level=max(e.circuit_breaker_level for e in day_entries),
        avg_duration_ms=round(sum(e.duration_ms for e in day_entries) / len(day_entries), 1),
        instruments_processed=sum(e.instruments_processed for e in day_entries),
    )


def _compute_rolling_7d(entries: list[CycleLogEntry], end_date: date) -> dict:
    """Compute 7-day rolling metrics ending on end_date."""
    start_date = end_date - timedelta(days=6)
    window_entries = [e for e in entries if start_date <= e.timestamp.date() <= end_date]

    if not window_entries:
        return {"days_with_data": 0, "message": "No data in 7-day window"}

    by_date: dict[str, list[CycleLogEntry]] = defaultdict(list)
    for e in window_entries:
        by_date[e.timestamp.date().isoformat()].append(e)

    daily_drawdowns = []
    daily_signals = []
    daily_errors = []
    for day_entries in by_date.values():
        daily_drawdowns.append(max(e.drawdown_pct for e in day_entries))
        daily_signals.append(sum(e.signals_generated for e in day_entries))
        daily_errors.append(sum(e.errors_caught for e in day_entries))

    total_orders = sum(e.orders_submitted for e in window_entries)
    total_fills = sum(e.orders_filled for e in window_entries)

    return {
        "days_with_data": len(by_date),
        "total_cycles": len(window_entries),
        "total_signals": sum(daily_signals),
        "avg_signals_per_day": round(sum(daily_signals) / len(by_date), 1),
        "total_orders": total_orders,
        "total_fills": total_fills,
        "fill_rate_pct": round(total_fills / total_orders * 100, 1) if total_orders > 0 else 0.0,
        "max_drawdown_pct": round(max(daily_drawdowns), 4),
        "avg_daily_drawdown_pct": round(sum(daily_drawdowns) / len(daily_drawdowns), 4),
        "total_errors": sum(daily_errors),
        "avg_errors_per_day": round(sum(daily_errors) / len(by_date), 1),
    }


def _load_recent_iterations(history_path: Path, days: int = 7) -> list[dict]:
    """Load iterations from the last N days."""
    if not history_path.exists():
        return []

    cutoff = datetime.now(tz=UTC) - timedelta(days=days)
    recent = []
    for line in history_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        created = datetime.fromisoformat(entry.get("created_at", "2000-01-01"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if created >= cutoff:
            recent.append(entry)
    return recent


# -- Anomaly detection thresholds --
_DRAWDOWN_WARNING_PCT = 2.0
_FILL_RATE_MIN_PCT = 80.0
_MIN_CYCLES = 2
_MIN_ROLLING_DAYS = 3
_GAP_SECONDS_THRESHOLD = 7200  # 2 hours


def _detect_anomalies(snapshot: DailySnapshot | None, rolling: dict) -> list[str]:
    """Flag anomalous conditions."""
    anomalies = []

    if snapshot is None:
        anomalies.append("NO_DATA: No trading cycles recorded today")
        return anomalies

    if snapshot.errors_caught > 0:
        anomalies.append(f"ERRORS: {snapshot.errors_caught} errors caught during trading")

    if snapshot.max_circuit_breaker_level > 0:
        anomalies.append(f"CIRCUIT_BREAKER: Level {snapshot.max_circuit_breaker_level} triggered")

    if snapshot.max_drawdown_pct > _DRAWDOWN_WARNING_PCT:
        anomalies.append(
            f"HIGH_DRAWDOWN: {snapshot.max_drawdown_pct:.2f}% exceeds 2% warning threshold"
        )

    if snapshot.signals_generated == 0:
        anomalies.append("NO_SIGNALS: Zero signals generated — possible data or strategy issue")

    if snapshot.fill_rate_pct < _FILL_RATE_MIN_PCT and snapshot.orders_submitted > 0:
        anomalies.append(
            f"LOW_FILL_RATE: {snapshot.fill_rate_pct:.1f}% fill rate below 80% threshold"
        )

    if snapshot.total_cycles < _MIN_CYCLES:
        anomalies.append(
            f"LOW_CYCLES: Only {snapshot.total_cycles} cycles — possible scheduling issue"
        )

    # Rolling comparison
    days_with_data = rolling.get("days_with_data", 0)
    if days_with_data >= _MIN_ROLLING_DAYS and snapshot.signals_generated > 0:
        avg_signals = rolling.get("avg_signals_per_day", 0)
        if avg_signals > 0 and snapshot.signals_generated < avg_signals * 0.3:
            anomalies.append(
                f"SIGNAL_DROP: {snapshot.signals_generated} signals vs "
                f"{avg_signals:.0f} 7d avg — 70%+ decrease"
            )

    return anomalies


def _check_data_quality(entries: list[CycleLogEntry], target_date: date) -> dict:
    """Check data quality indicators."""
    day_entries = _filter_by_date(entries, target_date)

    if not day_entries:
        return {"status": "NO_DATA", "details": "No cycle entries for this date"}

    timestamps = [e.timestamp for e in day_entries]
    gaps = []
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if delta > _GAP_SECONDS_THRESHOLD:
            gaps.append(
                {
                    "from": timestamps[i - 1].isoformat(),
                    "to": timestamps[i].isoformat(),
                    "gap_seconds": int(delta),
                }
            )

    return {
        "status": "OK" if not gaps else "GAPS_DETECTED",
        "cycle_count": len(day_entries),
        "first_cycle": timestamps[0].isoformat(),
        "last_cycle": timestamps[-1].isoformat(),
        "time_gaps": gaps,
    }


def collect(target_date: date, output_root: Path) -> DailyReport:
    """Collect all daily review data and save to output directory."""
    entries = _load_cycle_entries(CYCLE_LOG_PATH)
    snapshot = _compute_snapshot(entries, target_date)
    rolling = _compute_rolling_7d(entries, target_date)
    iterations = _load_recent_iterations(ITERATION_HISTORY_PATH)
    anomalies = _detect_anomalies(snapshot, rolling)
    data_quality = _check_data_quality(entries, target_date)

    report = DailyReport(
        report_date=target_date.isoformat(),
        generated_at=datetime.now(tz=UTC).isoformat(),
        snapshot=snapshot,
        rolling_7d=rolling,
        recent_iterations=iterations,
        anomalies=anomalies,
        data_quality=data_quality,
    )

    # Save
    day_dir = output_root / target_date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    report_path = day_dir / "raw_data.json"

    report_dict = {
        "report_date": report.report_date,
        "generated_at": report.generated_at,
        "snapshot": asdict(report.snapshot) if report.snapshot else None,
        "rolling_7d": report.rolling_7d,
        "recent_iterations": report.recent_iterations,
        "anomalies": report.anomalies,
        "data_quality": report.data_quality,
    }
    report_path.write_text(json.dumps(report_dict, indent=2, default=str))
    print(f"Report saved to {report_path}")

    return report


def print_summary(output_root: Path, days: int = 7) -> None:
    """Print summary of last N days of reports."""
    today = datetime.now(tz=UTC).date()
    print(f"\n{'=' * 60}")
    print(f"  Daily Review Summary (last {days} days)")
    print(f"{'=' * 60}\n")

    for i in range(days):
        d = today - timedelta(days=i)
        report_path = output_root / d.isoformat() / "raw_data.json"
        if report_path.exists():
            data = json.loads(report_path.read_text())
            snap = data.get("snapshot")
            anomalies = data.get("anomalies", [])
            if snap:
                status = "!!" if anomalies else "OK"
                print(
                    f"  {d}  [{status}]  "
                    f"cycles={snap['total_cycles']}  "
                    f"signals={snap['signals_generated']}  "
                    f"fills={snap['orders_filled']}  "
                    f"dd={snap['max_drawdown_pct']:.2f}%  "
                    f"errors={snap['errors_caught']}"
                )
                for a in anomalies:
                    print(f"           ^ {a}")
            else:
                print(f"  {d}  [--]  No trading data")
        else:
            print(f"  {d}  [--]  No report")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily post-market review data collection")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/daily/",
        help="Output root directory",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        default=True,
        help="Collect today's data (default action)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print last 7 days summary instead of collecting",
    )
    args = parser.parse_args()

    output_root = Path(args.output)

    if args.summary:
        print_summary(output_root)
        return

    target = date.fromisoformat(args.date) if args.date else datetime.now(tz=UTC).date()
    report = collect(target, output_root)

    # Print quick summary
    if report.snapshot:
        s = report.snapshot
        print(f"\n  Date: {s.date}")
        print(f"  Cycles: {s.total_cycles} (equity={s.equity_cycles}, bond={s.bond_cycles})")
        print(
            f"  Signals: {s.signals_generated}  Orders: {s.orders_submitted}"
            f"  Fills: {s.orders_filled}"
        )
        print(f"  Fill Rate: {s.fill_rate_pct}%  Max DD: {s.max_drawdown_pct:.2f}%")
        print(f"  Errors: {s.errors_caught}  CB Level: {s.max_circuit_breaker_level}")
    else:
        print(f"\n  No trading data for {target}")

    if report.anomalies:
        print(f"\n  Anomalies ({len(report.anomalies)}):")
        for a in report.anomalies:
            print(f"    - {a}")

    print()


if __name__ == "__main__":
    main()
