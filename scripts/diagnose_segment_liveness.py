"""Read-only per-segment LIVENESS classifier (UNIV-01 / D-01).

Names the root-cause bucket for each suspect MOEX segment so Phase-68 Waves 2-3
act on measured data, not assumptions (Pitfall 3). Distinguishes the three
buckets D-03 keys on, plus ``alive``:

- ``no_symbols``  -- the selector returns an empty set (e.g. ru_utilities: its
  sole liquid name IRAO is sanctioned). Structural -> disable.
- ``no_candles``  -- symbols ARE selected but every one is below the 60-bar
  as-of eligibility gate (young/newly-listed names). Defer (revival is a future
  phase) unless a documented bounded min-bars override is justified.
- ``no_signals``  -- candles present (>= 60 bars) but zero trades, because the
  preset-less default strategy thresholds never clear (a potentially cheap fix).
- ``alive``       -- candles present and at least one trade fired.

This is a PURE FILE READER (Layer 7, T-68-02): it imports ONLY stdlib, the
Layer-2 ``select_segment_symbols`` selector (which reads the committed snapshot
OFFLINE -- no token), and the existing ``diagnose_ru_finance.load_trades_jsonl``
trade reader. It performs NO network or secret access whatsoever -- there is no
data-source/order-routing client import and no credential env var read here (the
acceptance grep for those symbols is 0).

Usage (``--run`` is REQUIRED -- no default, so a forgotten flag errors loudly
rather than reading a prior phase's stale sidecar):
    uv run python scripts/diagnose_segment_liveness.py --run phase68-baseline
    uv run python scripts/diagnose_segment_liveness.py --run <run> \
        --segments ru_metals,ru_consumer --output results/iterations/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.diagnose_ru_finance import load_trades_jsonl

from finalayze.markets.liquidity import _MIN_BARS_FOR_LIQUIDITY, select_segment_symbols

# The four bucket strings (named constants -- no bare literals in the classifier).
_BUCKET_NO_SYMBOLS = "no_symbols"
_BUCKET_NO_CANDLES = "no_candles"
_BUCKET_NO_SIGNALS = "no_signals"
_BUCKET_ALIVE = "alive"

# The verified suspect-segment list (the Phase-68 RESEARCH defect set): the
# ru_blue_chips collapse + the six zero-trade sectors + ru_utilities.
_DEFAULT_SUSPECT_SEGMENTS = (
    "ru_blue_chips",
    "ru_metals",
    "ru_consumer",
    "ru_telecom",
    "ru_transport",
    "ru_chemicals",
    "ru_construction",
    "ru_utilities",
)


@dataclass(frozen=True)
class SegmentLiveness:
    """Per-segment liveness verdict (the UNIV-01 deliverable row)."""

    segment_id: str
    bucket: str
    selected_count: int
    max_bar_count: int
    trade_count: int
    signal_count: int
    selected_symbols: list[str]


def classify_segment_liveness(
    *,
    selected_count: int,
    max_bar_count: int,
    trade_count: int,
    signal_count: int,  # noqa: ARG001 -- part of the documented contract; recorded in the
    # verdict table for the human reader but does NOT gate the bucket (a no-trade segment is
    # no_signals whether or not raw signals fired -- both are the "fixable threshold" bucket).
) -> str:
    """Classify a segment into exactly one liveness bucket (pure).

    Order matters: no_symbols dominates (nothing to trade), then the candle gate
    (no_candles when the best symbol is below the min-history floor), then trades
    (alive) vs no trades (no_signals). A boundary symbol exactly at the floor with
    signals but no trade is a fixable no_signals case, NOT no_candles.
    """
    if selected_count <= 0:
        return _BUCKET_NO_SYMBOLS
    if max_bar_count < _MIN_BARS_FOR_LIQUIDITY:
        return _BUCKET_NO_CANDLES
    if trade_count > 0:
        return _BUCKET_ALIVE
    return _BUCKET_NO_SIGNALS


def _read_segment_artifacts(output_root: Path, run: str, segment: str) -> tuple[int, int]:
    """Read (max_bar_count, signal_count) from a run's per-segment artifacts.

    Reads the consolidated ``summary.json`` (a list of per-segment summary dicts
    written by run_iteration) when present, and the segment's ``summary.json``
    sidecar if one exists. Degrades GRACEFULLY: a never-run segment (no artifacts)
    yields ``(0, 0)`` -- which classifies as no_candles, an honest "no data" verdict.

    ``total_candles`` in the summary is the SUM of bars across the segment's
    symbols, which is a conservative upper bound on any single symbol's bar count;
    a segment with a non-zero total has at least one symbol with bars, so we use it
    as the ``max_bar_count`` proxy (>= the true max only when one symbol dominates,
    but for the bucket decision the relevant question is "did ANY symbol clear the
    60-bar gate" and a non-trivial total answers that conservatively in favour of
    no_signals over no_candles -- the diagnosis then leans on the trade count).
    """
    max_bar_count = 0
    signal_count = 0

    consolidated = output_root / run / "summary.json"
    if consolidated.exists():
        try:
            data = json.loads(consolidated.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = []
        rows = data if isinstance(data, list) else []
        for row in rows:
            if isinstance(row, dict) and row.get("segment") == segment:
                max_bar_count = max(max_bar_count, int(row.get("total_candles", 0) or 0))
                signal_count = max(signal_count, int(row.get("combined_above_threshold", 0) or 0))

    seg_summary = output_root / run / segment / "summary.json"
    if seg_summary.exists():
        try:
            row = json.loads(seg_summary.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            row = {}
        if isinstance(row, dict):
            max_bar_count = max(max_bar_count, int(row.get("total_candles", 0) or 0))
            signal_count = max(signal_count, int(row.get("combined_above_threshold", 0) or 0))

    return max_bar_count, signal_count


def diagnose_segment(output_root: Path, run: str, segment: str) -> SegmentLiveness:
    """Classify one segment from offline selection + persisted run artifacts."""
    selected = select_segment_symbols(segment)
    trades = load_trades_jsonl(output_root / run / segment / "trades.jsonl")
    max_bar_count, signal_count = _read_segment_artifacts(output_root, run, segment)
    bucket = classify_segment_liveness(
        selected_count=len(selected),
        max_bar_count=max_bar_count,
        trade_count=len(trades),
        signal_count=signal_count,
    )
    return SegmentLiveness(
        segment_id=segment,
        bucket=bucket,
        selected_count=len(selected),
        max_bar_count=max_bar_count,
        trade_count=len(trades),
        signal_count=signal_count,
        selected_symbols=selected,
    )


def _print_table(rows: list[SegmentLiveness], *, run: str) -> None:
    print()
    print(f"  Segment liveness verdict @ run={run}")
    print("  " + "-" * 78)
    header = (
        f"  {'segment':<16} {'bucket':<12} {'selected':>8} "
        f"{'max_bars':>8} {'trades':>7} {'signals':>8}"
    )
    print(header)
    print("  " + "-" * 78)
    for r in rows:
        print(
            f"  {r.segment_id:<16} {r.bucket:<12} {r.selected_count:>8} "
            f"{r.max_bar_count:>8} {r.trade_count:>7} {r.signal_count:>8}"
        )
    print()
    for r in rows:
        print(f"  {r.segment_id:<16} -> {r.selected_symbols}")
    print()


def main() -> None:
    """CLI entry point: classify each suspect segment and print the verdict table."""
    parser = argparse.ArgumentParser(description="Per-segment liveness classifier (UNIV-01)")
    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Iteration run name under --output (e.g. phase68-baseline). REQUIRED: there "
            "is no default so a forgotten flag errors loudly instead of silently reading "
            "a prior phase's stale artifacts."
        ),
    )
    parser.add_argument(
        "--segments",
        default=",".join(_DEFAULT_SUSPECT_SEGMENTS),
        help="Comma-separated segment ids to classify (default: the Phase-68 suspect set).",
    )
    parser.add_argument("--output", default="results/iterations/", help="Iterations root directory")
    args = parser.parse_args()

    output_root = Path(args.output)
    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    rows = [diagnose_segment(output_root, args.run, seg) for seg in segments]
    _print_table(rows, run=args.run)


if __name__ == "__main__":
    main()
