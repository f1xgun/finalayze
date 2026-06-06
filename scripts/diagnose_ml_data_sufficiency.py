"""H1 data/sample-sufficiency reporter (MLDIAG-01) -- token-free, read-only.

Measures, per MOEX segment in {ru_blue_chips, ru_energy}, the quantities the
walk-forward ML gate actually consumes:

* raw bar count + date span (usable history),
* triple-barrier sample count + label class balance,
* walk-forward fold count, and per-fold ``n_test`` / ``n_effective``.

It reuses the EXACT production labelling + fold code paths
(``build_triple_barrier_dataset`` + ``generate_walk_forward_folds``). FIDELITY
SCOPE -- two granularities are reported, only one of which mirrors the binding
gate:

* PER SYMBOL (``report_for_candles``): raw-bar / sample / class-balance counts
  match the gate's symbol-level path (``_build_dataset_triple_barrier``).
  ``fold_count`` and ``n_effective`` here are PER SYMBOL over that symbol's
  narrower date span and DO NOT equal what the gate sees.
* PER SEGMENT (``report_for_segment``): pools all non-skipped symbols'
  timestamps + hold-bars into a single date-sorted dataset and runs folds ONCE
  per segment -- exactly as ``_build_dataset_triple_barrier`` ->
  ``generate_walk_forward_folds`` does. This ``pooled`` block IS the binding
  gate's view (segment ``fold_count`` >= any per-symbol number), and is the
  number a re-enablement decision should read.

It is DB-first (token-free); it only escalates to Tinkoff gRPC for symbols the
DB misses, reading the token via ``os.environ`` (never logged/printed --
T-70-01).

HONESTY GUARDRAIL (Phase-70 D-05): this reporter trains NO model and ships NO
model -- it never writes any artifact under ``models/`` and never bypasses the
walk-forward gate. It is pure measurement.

Usage::

    # Token-free DB-first run (default segments ru_blue_chips + ru_energy):
    python scripts/diagnose_ml_data_sufficiency.py

    # Escalate to Tinkoff for DB-missed symbols (operator run, worktree recipe):
    #   export FINALAYZE_TINKOFF_TOKEN=...   (token-only, NOT `source .env`)
    #   export GRPC_DNS_RESOLVER=native      (+ certs/grpc_roots.pem symlinked)
    python scripts/diagnose_ml_data_sufficiency.py \\
        --segments ru_blue_chips ru_energy \\
        --output results/iterations/ml_data_sufficiency_phase70.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure src/ and project root are importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))  # for config.settings

from scripts.training.data_loader import (
    fetch_from_db,
    fetch_tinkoff_candles,
    is_moex_segment,
)
from scripts.training.dataset_builder import (
    _MIN_HISTORY_DAYS,
    _WINDOW_SIZE,
    get_triple_barrier_params,
)
from scripts.training.quality import compute_n_eff
from scripts.training.walk_forward import generate_walk_forward_folds

from finalayze.ml.training.labeling import build_triple_barrier_dataset

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import Candle

# Default segments -- the two data-richest MOEX segments (Phase-70 D-03).
_DEFAULT_SEGMENTS = ["ru_blue_chips", "ru_energy"]
_DEFAULT_OUTPUT = Path("results/iterations/ml_data_sufficiency_phase70.json")


def _folds_from_tuples(
    fold_tuples: list[tuple[list[int], list[int], list[int]]],
    hold_bars: list[int],
) -> list[dict[str, object]]:
    """Build per-fold {n_test, n_effective} entries, mirroring the gate.

    For each fold, ``avg_hold`` is computed from that fold's OWN test slice
    (``hold_bars[i] for i in test_idx``), exactly as production does in
    ``walk_forward.py:392-394`` -- with the empty-test fallback of ``1.0`` (NOT
    ``max_hold``) so the per-fold ``n_effective`` equals the value the gate
    feeds ``compute_n_eff``. ``hold_bars`` is index-aligned with the timestamps
    passed to ``generate_walk_forward_folds``, so ``test_idx`` indexes it
    directly.
    """
    folds: list[dict[str, object]] = []
    for _train_idx, _cal_idx, test_idx in fold_tuples:
        n_test = len(test_idx)
        test_hb = [hold_bars[i] for i in test_idx if i < len(hold_bars)]
        fold_avg_hold = (sum(test_hb) / len(test_hb)) if test_hb else 1.0
        n_effective = float(compute_n_eff(n_test, fold_avg_hold))
        folds.append({"n_test": n_test, "n_effective": n_effective})
    return folds


def report_for_candles(
    segment_id: str,
    symbol: str,
    candles: list[Candle],
) -> dict[str, object]:
    """Compute the H1 sufficiency report for ONE symbol's candles.

    PURE / token-free by construction: takes already-fetched candles as an
    argument and makes NO network call and reads NO token. Reuses the exact
    production labelling path (``build_triple_barrier_dataset``), so the
    per-symbol raw-bar / sample / class-balance counts match the gate's
    symbol-level path (``_build_dataset_triple_barrier``).

    FIDELITY NOTE: ``fold_count`` and ``n_effective`` returned here are PER
    SYMBOL over this symbol's narrower date span. The binding gate pools ALL
    symbols' timestamps into a single date-sorted dataset and runs folds ONCE
    per segment, so the segment-level ``fold_count`` (see ``report_for_segment``)
    is >= the per-symbol number reported here. Read the segment ``pooled`` block
    for the gate's actual view.

    Returns a dict with:
        raw_bar_count, date_span (start, end), sample_count, class_balance
        (label -> count), fold_count, folds (list of {n_test, n_effective}),
        skipped (bool -- True when below the production min-history gate).
    """
    raw_bar_count = len(candles)
    if candles:
        # This local sort exists ONLY to extract date_span (first/last timestamp);
        # build_triple_barrier_dataset re-sorts internally (labeling.py:290), so
        # the sorted order passed below is NOT a load-bearing invariant. The
        # redundant double sort is correctness-neutral and out of v1 perf scope
        # (IN-04).
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        date_span: tuple[datetime, datetime] | None = (
            sorted_candles[0].timestamp,
            sorted_candles[-1].timestamp,
        )
    else:
        sorted_candles = []
        date_span = None

    tb_params = get_triple_barrier_params(segment_id)
    max_hold = int(tb_params["max_hold"])
    min_candles_tb = _WINDOW_SIZE + max_hold + 1

    # Mirror the production gate in _build_dataset_triple_barrier: a symbol with
    # < _MIN_HISTORY_DAYS or < min_candles_tb is SKIPPED (contributes 0 samples).
    # The `< min_candles_tb` (101) clause is subsumed by `< _MIN_HISTORY_DAYS`
    # (500) and can never be the deciding factor; it is kept VERBATIM to mirror
    # the two-stage production gate exactly (do NOT "simplify" it away) -- IN-02.
    skipped = raw_bar_count < _MIN_HISTORY_DAYS or raw_bar_count < min_candles_tb
    if skipped:
        return {
            "symbol": symbol,
            "raw_bar_count": raw_bar_count,
            "date_span": date_span,
            "sample_count": 0,
            "class_balance": {},
            "fold_count": 0,
            "folds": [],
            "skipped": True,
        }

    # Production labelling path (identical params to the trainer). Returns
    # (features, labels, weights, timestamps, hold_bars).
    _features, labels, _weights, timestamps, hold_bars = build_triple_barrier_dataset(
        sorted_candles,
        window_size=_WINDOW_SIZE,
        upper_atr_mult=float(tb_params["upper_atr_mult"]),
        lower_atr_mult=float(tb_params["lower_atr_mult"]),
        max_hold=max_hold,
        atr_period=int(tb_params["atr_period"]),
        atr_scale=bool(tb_params["atr_scale"]),
    )

    sample_count = len(labels)
    class_balance: dict[int, int] = {}
    for lbl in labels:
        class_balance[lbl] = class_balance.get(lbl, 0) + 1

    # Production fold generator (MOEX windows when segment_id starts with "ru_").
    fold_tuples = generate_walk_forward_folds(timestamps, segment_id)
    # Per-fold avg_hold from the fold's OWN test slice, mirroring the gate
    # (walk_forward.py:392-394). hold_bars is index-aligned with timestamps, so
    # test_idx indexes into it directly. Empty-test fallback is 1.0 to match
    # production exactly (NOT max_hold), so per-fold n_effective equals the gate.
    folds = _folds_from_tuples(fold_tuples, hold_bars)

    return {
        "symbol": symbol,
        "raw_bar_count": raw_bar_count,
        "date_span": date_span,
        "sample_count": sample_count,
        "class_balance": class_balance,
        "fold_count": len(folds),
        "folds": folds,
        "skipped": False,
    }


def report_for_segment(
    segment_id: str,
    candles_by_symbol: dict[str, list[Candle]],
) -> dict[str, object]:
    """Compute the SEGMENT-level pooled-fold report -- the gate's actual view.

    PURE / token-free by construction (takes already-fetched candles, makes NO
    network call, reads NO token). Mirrors the production gate's
    ``_build_dataset_triple_barrier`` -> ``generate_walk_forward_folds`` path:
    each non-skipped symbol is labelled with ``build_triple_barrier_dataset``,
    every symbol's ``(timestamp, hold_bar)`` rows are accumulated into one list,
    that pooled list is date-sorted, and ``generate_walk_forward_folds`` runs
    ONCE over the union-of-all-symbols timestamp range. This is what the binding
    WF gate consumes, so segment ``fold_count`` here is the number a
    re-enablement decision should read (>= any per-symbol ``fold_count``).

    Returns a dict with:
        pooled_sample_count, pooled_date_span (start, end) | None,
        contributing_symbols (list -- non-skipped symbols that fed the pool),
        fold_count, folds (list of {n_test, n_effective}).
    """
    tb_params = get_triple_barrier_params(segment_id)
    max_hold = int(tb_params["max_hold"])
    min_candles_tb = _WINDOW_SIZE + max_hold + 1

    # rows: (timestamp, hold_bar) accumulated across symbols, mirroring
    # _build_dataset_triple_barrier (dataset_builder.py:404-412).
    rows: list[tuple[datetime, int]] = []
    contributing: list[str] = []
    for symbol, candles in candles_by_symbol.items():
        raw_bar_count = len(candles)
        # Same two-stage skip gate as report_for_candles / production. The
        # `< min_candles_tb` (101) clause is subsumed by `< _MIN_HISTORY_DAYS`
        # (500); kept VERBATIM to mirror the production gate exactly (IN-02).
        if raw_bar_count < _MIN_HISTORY_DAYS or raw_bar_count < min_candles_tb:
            continue
        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        _features, _labels, _weights, timestamps, hold_bars = build_triple_barrier_dataset(
            sorted_candles,
            window_size=_WINDOW_SIZE,
            upper_atr_mult=float(tb_params["upper_atr_mult"]),
            lower_atr_mult=float(tb_params["lower_atr_mult"]),
            max_hold=max_hold,
            atr_period=int(tb_params["atr_period"]),
            atr_scale=bool(tb_params["atr_scale"]),
        )
        if not timestamps:
            continue
        contributing.append(symbol)
        rows.extend(zip(timestamps, hold_bars, strict=True))

    # Date-sort the pooled union (mirrors dataset_builder.py:407 rows.sort).
    rows.sort(key=lambda r: r[0])
    pooled_timestamps = [r[0] for r in rows]
    pooled_hold_bars = [r[1] for r in rows]
    pooled_sample_count = len(pooled_timestamps)
    pooled_date_span: tuple[datetime, datetime] | None = (
        (pooled_timestamps[0], pooled_timestamps[-1]) if pooled_timestamps else None
    )

    # ONE fold generation over the pooled, segment-wide timestamp range.
    fold_tuples = generate_walk_forward_folds(pooled_timestamps, segment_id)
    folds = _folds_from_tuples(fold_tuples, pooled_hold_bars)

    return {
        "pooled_sample_count": pooled_sample_count,
        "pooled_date_span": pooled_date_span,
        "contributing_symbols": contributing,
        "fold_count": len(folds),
        "folds": folds,
    }


def fetch_segment_candles(
    segment_id: str,
    symbols: list[str],
) -> tuple[dict[str, list[Candle]], list[str]]:
    """Fetch candles per symbol DB-first; escalate to Tinkoff only on DB miss.

    Returns ``(candles_by_symbol, db_missed)``. DB-first keeps the run
    token-free when candles are cached. ``db_missed`` lists ONLY symbols still
    pending after escalation -- a symbol the DB lacked but Tinkoff back-filled
    this run is NOT included (WR-03), so the "needs one-time Tinkoff backfill"
    summary stays accurate. The Tinkoff escalation reads the token via
    ``os.environ`` (NOT ``Settings`` / ``source .env``) and the token is NEVER
    logged or printed (T-70-01). The yfinance fallback in the production loader
    is deliberately bypassed because it returns EMPTY for MOEX.
    """
    import asyncio  # noqa: PLC0415

    from config.settings import Settings  # noqa: PLC0415

    settings = Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]
    candles_by_symbol: dict[str, list[Candle]] = {}
    db_missed: list[str] = []

    has_token = bool(os.environ.get("FINALAYZE_TINKOFF_TOKEN"))

    for symbol in symbols:
        candles = asyncio.run(fetch_from_db(symbol, market_id, settings))
        if candles:
            candles_by_symbol[symbol] = candles
            continue
        # DB miss. Escalate to Tinkoff ONLY for MOEX and ONLY when a token is
        # present (keeps the default run token-free). Record the symbol as
        # still-pending ONLY if the escalation did not produce candles -- a
        # symbol back-filled this run is NOT a pending backfill (WR-03).
        db_miss = True
        if is_moex_segment(segment_id) and has_token:
            tinkoff_candles = fetch_tinkoff_candles(symbol)
            if tinkoff_candles:
                candles_by_symbol[symbol] = tinkoff_candles
                db_miss = False  # back-filled this run; not pending
        if db_miss:
            db_missed.append(symbol)

    return candles_by_symbol, db_missed


def _segment_symbols(segment_id: str) -> list[str]:
    """Resolve a segment's symbols via the production SEGMENT_SYMBOLS map."""
    from scripts.training.cli import SEGMENT_SYMBOLS  # noqa: PLC0415

    return SEGMENT_SYMBOLS.get(segment_id, [])


def _print_summary(report: dict[str, object]) -> None:
    """Print a human-readable per-symbol summary table to stdout."""
    print("\n=== ML data-sufficiency summary (H1) ===")
    header = (
        f"{'segment':<16} {'symbol':<8} {'raw_bars':>9} "
        f"{'samples':>8} {'folds':>6} {'n_eff/fold':>24}"
    )
    print(header)
    print("-" * len(header))
    segments = report["segments"]
    assert isinstance(segments, dict)
    for segment_id, seg in segments.items():
        assert isinstance(seg, dict)
        per_symbol = seg["symbols"]
        assert isinstance(per_symbol, dict)
        db_missed = seg.get("db_missed", [])
        assert isinstance(db_missed, list)
        for symbol, sym in per_symbol.items():
            assert isinstance(sym, dict)
            n_eff = [round(float(f["n_effective"]), 1) for f in sym["folds"]]
            flags = []
            if sym.get("skipped"):
                flags.append("SKIPPED<500d")
            if symbol in db_missed:
                flags.append("DB-MISSED->needs-Tinkoff-backfill")
            flag_str = ("  " + ", ".join(flags)) if flags else ""
            print(
                f"{segment_id:<16} {symbol:<8} {sym['raw_bar_count']:>9} "
                f"{sym['sample_count']:>8} {sym['fold_count']:>6} {n_eff!s:>24}{flag_str}"
            )
        # Segment-level pooled folds = the binding gate's actual view (WR-01).
        pooled = seg.get("pooled")
        if isinstance(pooled, dict):
            pooled_n_eff = [round(float(f["n_effective"]), 1) for f in pooled["folds"]]
            print(
                f"  [{segment_id}] POOLED (gate view): "
                f"samples={pooled['pooled_sample_count']} "
                f"folds={pooled['fold_count']} "
                f"n_eff/fold={pooled_n_eff} "
                f"contributing={pooled['contributing_symbols']}"
            )
        if db_missed:
            print(
                f"  [{segment_id}] DB-missed symbols (need one-time Tinkoff backfill): {db_missed}"
            )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point -- token-free, read-only H1 reporter."""
    parser = argparse.ArgumentParser(
        description=(
            "H1 data/sample-sufficiency reporter (MLDIAG-01): per-segment/symbol "
            "triple-barrier sample, WF fold, and usable-history counts. Token-free "
            "DB-first; escalates to Tinkoff only for DB-missed symbols. Trains/ships "
            "NO model; never bypasses the walk-forward gate."
        )
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=list(_DEFAULT_SEGMENTS),
        help=f"Segments to report (default: {' '.join(_DEFAULT_SEGMENTS)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    report: dict[str, object] = {"segments": {}}
    segments_out = report["segments"]
    assert isinstance(segments_out, dict)

    for segment_id in args.segments:
        symbols = _segment_symbols(segment_id)
        if not symbols:
            print(f"  [{segment_id}] no symbols resolved -- skipping.", file=sys.stderr)
        candles_by_symbol, db_missed = fetch_segment_candles(segment_id, symbols)
        per_symbol: dict[str, object] = {}
        for symbol in symbols:
            candles = candles_by_symbol.get(symbol, [])
            per_symbol[symbol] = report_for_candles(segment_id, symbol, candles)
        # Segment-level pooled-fold report -- THE binding gate's view (WR-01).
        pooled = report_for_segment(segment_id, candles_by_symbol)
        segments_out[segment_id] = {
            "symbols": per_symbol,
            "pooled": pooled,
            "db_missed": db_missed,
        }

    # Write the aggregate JSON (date_span datetimes serialized via default=str).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote sufficiency report to {args.output}")

    _print_summary(report)


if __name__ == "__main__":
    main()
