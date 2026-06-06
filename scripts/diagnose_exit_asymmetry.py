"""Segment-agnostic exit-path loss-asymmetry diagnostic (EXITDIAG-01/03/05, D-06).

Promoted from ``scripts/diagnose_ru_finance.py``: drops the ru_finance-specific
naming, batch-runs across every enabled MOEX segment off each segment's
``trades.jsonl`` sidecar, and emits ONE consolidated, severity-ranked
cross-segment report that names the implicated lever per segment.

Per-segment it reports the loss-asymmetry attribution (avg-win / avg-loss +
payoff ratio, exit-reason share, per-strategy / per-symbol realised PnL) and a
named LEVER VERDICT via a TWO-BRANCH dispatch (D-04):

* EQUITY: the Phase-67 payoff-driven binary -- "chandelier stop multiplier"
  (losers run too far) vs "min_exit_confidence" (winners cut early). The frozen
  ``_LOSS_DOMINANCE_FACTOR = Decimal("1.0")`` break-even is preserved verbatim.
* BOND: bonds run ``bond_duration_rotation`` / ``bond_carry`` and have no ATR
  chandelier / exit-confidence levers, so the bond branch NEVER emits a
  chandelier verdict -- it names a bond-relevant lever keyed on the exit-reason
  mix (yield_stop / max_hold / rebalance / max_positions).

Segments below ``_THIN_TRADE_FLOOR`` closed trades are flagged
"low-confidence -- informational only" (D-05) and are diagnosed but never tuned.

This is a PURE FILE READER (Layer 7, T-69-04): it imports ONLY stdlib + the
frozen ``TradeResult`` schema (and ``config.segments`` for the default segment
list). It MUST NOT import any broker / data fetcher or reference an API token
env var -- it never touches the network or any secret (the acceptance grep for
token symbols is 0).

Usage (``--run`` is REQUIRED -- no default, so a forgotten flag errors loudly
rather than reading a prior phase's stale sidecar):
    uv run python scripts/diagnose_exit_asymmetry.py --run phase69-diagnostic-baseline
    uv run python scripts/diagnose_exit_asymmetry.py --run <run> \
        --segments ru_energy,ru_tech,ru_finance --output results/iterations/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from finalayze.core.schemas import ExitReason, TradeResult

# A loss is materially asymmetric when avg-loss magnitude exceeds avg-win by
# this factor -- implicating the stop side (chandelier) rather than the entry.
#
# Pinned at exactly 1.0 ON PURPOSE: the gate is a bare ``loss_mag > avg_win``,
# i.e. ANY net loss-magnitude dominance names the chandelier. This is the
# intended boundary -- in this binary diagnostic the only two outcomes are
# "losers run too far" (stop/chandelier) vs "winners cut early"
# (min_exit_confidence), so the natural break-even is the 1:1 payoff line, not
# an arbitrary margin. The ru_finance case clears it with room to spare
# (avg-loss 1022.83 vs avg-win 566.52, ratio ~1.8), and the D-04 ACCEPT was
# decided on this exact bare-`>` verdict. Raising the factor (e.g. 1.2) would
# silently re-decide marginal future runs -- do not change without re-running
# the diagnostic and re-accepting the verdict.
_LOSS_DOMINANCE_FACTOR = Decimal("1.0")

# Minimum closed-trade count for a verdict to be statistically actionable.
# Below this floor a segment is flagged "low-confidence -- informational only"
# (D-05) and is diagnosed but NEVER tuned (anti-curve-fit on noise). Pinned at
# 25: consistent with the small-sample discipline of ``_HRP_MIN_HISTORY = 20``
# and Phase 67's treatment (~9 trades meaningless, ~68 usable) -- a defensible
# 20-30 mid-point. The 4 newly-activated sector segments and both OFZ bond
# segments are expected to land below it.
_THIN_TRADE_FLOOR = 25


@dataclass(frozen=True)
class Attribution:
    """Computed loss-asymmetry attribution for one segment (equity OR bond)."""

    total_trades: int
    win_count: int
    loss_count: int
    avg_win: Decimal
    avg_loss: Decimal
    payoff_ratio: Decimal
    win_rate: float
    exit_reason_counts: dict[str, int]
    exit_reason_pnl: dict[str, Decimal]
    # Diagnostic-only context (printed, NOT part of the lever verdict -- see
    # the _name_lever_* functions, which key on the avg-win/avg-loss payoff
    # asymmetry [equity] or the exit-reason mix [bond]).
    stop_hit_share: float
    signal_exit_share: float
    per_strategy_pnl: dict[str, Decimal] = field(default_factory=dict)
    per_symbol_pnl: dict[str, Decimal] = field(default_factory=dict)
    lever_verdict: str = ""


@dataclass(frozen=True)
class SegmentDiagnosis:
    """One segment's diagnosis: its attribution + the thin-sample flag (D-05)."""

    segment: str
    instrument_type: str
    attribution: Attribution
    thin_sample: bool

    def report_row(self) -> str:
        """A single consolidated-report markdown table row for this segment."""
        attr = self.attribution
        confidence = "low-confidence -- informational only" if self.thin_sample else "actionable"
        return (
            f"| {self.segment} | {self.instrument_type} | {attr.total_trades} "
            f"| {attr.win_rate:.1%} | {attr.payoff_ratio:.3f} "
            f"| {attr.lever_verdict} | {confidence} |"
        )


def load_trades_jsonl(path: Path) -> list[TradeResult]:
    """Read closed trades from a ``trades.jsonl`` sidecar.

    Inverse of the run_iteration writer: one ``TradeResult`` per JSON line.
    Returns ``[]`` when the file is absent (a run predating instrumentation).
    """
    if not path.exists():
        return []
    trades: list[TradeResult] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            trades.append(TradeResult.model_validate(json.loads(stripped)))
    return trades


def _name_lever_equity(segment: str, avg_win: Decimal, avg_loss: Decimal) -> str:
    """Name the equity lever the asymmetry data implicates (D-01 deliverable).

    The verdict is intentionally PAYOFF-DRIVEN -- it keys ONLY on avg-win vs
    avg-loss magnitude (the frozen ``_LOSS_DOMINANCE_FACTOR`` break-even). The
    ``stop_hit_share`` / ``signal_exit_share`` fields are printed DIAGNOSTIC-ONLY
    context; they do NOT gate this verdict (D-04 ACCEPT was decided on this
    payoff-driven verdict -- do not fold the shares in without re-accepting).
    """
    loss_mag = abs(avg_loss)
    if loss_mag > avg_win * _LOSS_DOMINANCE_FACTOR:
        return (
            f"LEVER VERDICT: {segment} chandelier stop multiplier "
            "(avg-loss magnitude dominates avg-win -- losers run too far)"
        )
    return (
        f"LEVER VERDICT: {segment} min_exit_confidence (avg-win too small -- winners are cut early)"
    )


def _name_lever_bond(segment: str, attr: Attribution) -> str:
    """Name a bond-relevant lever from the exit-reason mix (D-04, WR-01).

    NEVER emits a chandelier verdict: bonds run ``bond_duration_rotation`` /
    ``bond_carry`` and have no ATR chandelier / exit-confidence levers, so the
    equity payoff verdict does not map. Keys on the dominant exit reason and
    names ONLY honestly-wired bond levers (RESEARCH bond table):

      * FORCE_CLOSE share dominant -> max_hold bars / rebalance cadence
        (positions held to end -- exits dominated by force-close at last bar)
      * STOP share dominant        -> yield_stop_bps (the bond analogue of the stop)
      * TIME share dominant        -> max_hold bars
      * SIGNAL share dominant      -> rebalance_interval_bars / maturity-rotation cadence
      * dominant == 0 (no trades)  -> max_positions / duration-carry timing (informational)

    ``force_close`` MUST be in the dominance set (WR-01): ``bond_engine.run()``
    force-closes every position still open at the last bar with
    ``ExitReason.FORCE_CLOSE``, so in a single-pass OFZ backtest the majority of
    closes are typically ``force_close``. Excluding it mis-attributed the lever
    (a lone ``stop`` falsely "dominating", or a clear force-close majority
    reported as "no dominant exit reason").
    """
    counts = attr.exit_reason_counts
    stop = counts.get(ExitReason.STOP.value, 0)
    time_ct = counts.get(ExitReason.TIME.value, 0)
    signal = counts.get(ExitReason.SIGNAL.value, 0)
    force = counts.get(ExitReason.FORCE_CLOSE.value, 0)
    dominant = max(stop, time_ct, signal, force)

    prefix = f"LEVER VERDICT: {segment} bond"
    if dominant == 0:
        return f"{prefix} max_positions / duration-carry exit timing (no trades -- informational)"
    if force == dominant:
        return (
            f"{prefix} max_hold bars / rebalance cadence "
            "(positions held to end -- exits dominated by force-close at last bar)"
        )
    if stop == dominant:
        return f"{prefix} yield_stop_bps (yield-stop exits dominate)"
    if time_ct == dominant:
        return f"{prefix} max_hold bars (time exits dominate)"
    return f"{prefix} rebalance_interval_bars / maturity-rotation cadence (signal exits dominate)"


def compute_attribution(trades: list[TradeResult]) -> Attribution:
    """Compute the loss-asymmetry attribution from closed trades.

    Produces the equity ``lever_verdict`` by default (backwards-compatible with
    the Phase-67 single-segment contract). The bond branch is selected via
    ``diagnose_attribution``, which overrides ``lever_verdict`` after dispatch.
    """
    total = len(trades)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    win_count = len(wins)
    loss_count = len(losses)

    avg_win = sum(wins, Decimal(0)) / win_count if win_count else Decimal(0)
    avg_loss = sum(losses, Decimal(0)) / loss_count if loss_count else Decimal(0)
    payoff = avg_win / abs(avg_loss) if avg_loss != 0 else Decimal(0)
    win_rate = win_count / total if total else 0.0

    reason_counts: Counter[str] = Counter()
    reason_pnl: dict[str, Decimal] = {}
    strat_pnl: dict[str, Decimal] = {}
    sym_pnl: dict[str, Decimal] = {}
    for t in trades:
        reason = t.exit_reason or "unknown"
        reason_counts[reason] += 1
        reason_pnl[reason] = reason_pnl.get(reason, Decimal(0)) + t.pnl
        strat = t.entry_strategy or "unknown"
        strat_pnl[strat] = strat_pnl.get(strat, Decimal(0)) + t.pnl
        sym_pnl[t.symbol] = sym_pnl.get(t.symbol, Decimal(0)) + t.pnl

    stop_share = reason_counts.get("stop", 0) / total if total else 0.0
    signal_share = reason_counts.get("signal", 0) / total if total else 0.0

    return Attribution(
        total_trades=total,
        win_count=win_count,
        loss_count=loss_count,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=payoff,
        win_rate=win_rate,
        exit_reason_counts=dict(reason_counts),
        exit_reason_pnl=reason_pnl,
        stop_hit_share=stop_share,
        signal_exit_share=signal_share,
        per_strategy_pnl=strat_pnl,
        per_symbol_pnl=sym_pnl,
        # Default equity verdict; overridden by diagnose_attribution dispatch.
        lever_verdict=_name_lever_equity("segment", avg_win, avg_loss),
    )


def diagnose_attribution(
    segment: str, trades: list[TradeResult], *, instrument_type: str = "stock"
) -> Attribution:
    """Compute attribution and dispatch the verdict on instrument type (D-04).

    The equity branch names chandelier vs min_exit_confidence (payoff-driven);
    the bond branch names a bond-relevant lever and NEVER a chandelier. Empty
    trades -> a "no trades" sentinel verdict.
    """
    attr = compute_attribution(trades)
    if attr.total_trades == 0:
        verdict = f"LEVER VERDICT: {segment} no trades (no sidecar / empty run)"
    elif instrument_type == "bond":
        verdict = _name_lever_bond(segment, attr)
    else:
        verdict = _name_lever_equity(segment, attr.avg_win, attr.avg_loss)
    # Attribution is frozen -- rebuild with the dispatched verdict.
    return Attribution(
        total_trades=attr.total_trades,
        win_count=attr.win_count,
        loss_count=attr.loss_count,
        avg_win=attr.avg_win,
        avg_loss=attr.avg_loss,
        payoff_ratio=attr.payoff_ratio,
        win_rate=attr.win_rate,
        exit_reason_counts=attr.exit_reason_counts,
        exit_reason_pnl=attr.exit_reason_pnl,
        stop_hit_share=attr.stop_hit_share,
        signal_exit_share=attr.signal_exit_share,
        per_strategy_pnl=attr.per_strategy_pnl,
        per_symbol_pnl=attr.per_symbol_pnl,
        lever_verdict=verdict,
    )


def diagnose_segment(
    run: str,
    segment: str,
    output_root: Path,
    *,
    instrument_type: str = "stock",
    trade_floor: int = _THIN_TRADE_FLOOR,
) -> SegmentDiagnosis:
    """Load a segment's sidecar, compute attribution, dispatch the verdict.

    Builds the ``<output>/<run>/<segment>/trades.jsonl`` path exactly as the
    single-segment Phase-67 ``main`` did, loads it (missing -> ``[]``), computes
    the dispatched attribution, and sets the thin-sample flag (D-05).
    """
    path = output_root / run / segment / "trades.jsonl"
    trades = load_trades_jsonl(path)
    attr = diagnose_attribution(segment, trades, instrument_type=instrument_type)
    return SegmentDiagnosis(
        segment=segment,
        instrument_type=instrument_type,
        attribution=attr,
        thin_sample=attr.total_trades < trade_floor,
    )


def _print_report(attr: Attribution, *, run: str, segment: str) -> None:
    def _row(name: str, value: str) -> None:
        print(f"  {name:<28} {value:>14}")

    print()
    print(f"  exit-asymmetry attribution: {segment} @ {run}")
    print("  " + "-" * 44)
    _row("Trades", str(attr.total_trades))
    _row("Win / Loss count", f"{attr.win_count} / {attr.loss_count}")
    _row("Win rate", f"{attr.win_rate:.1%}")
    _row("Avg win", f"{attr.avg_win:+.2f}")
    _row("Avg loss", f"{attr.avg_loss:+.2f}")
    _row("Payoff (win/|loss|)", f"{attr.payoff_ratio:.3f}")
    print()
    print("  Exit-reason share (count | summed PnL):")
    for reason in ("stop", "profit_target", "time", "signal", "force_close", "unknown"):
        if reason in attr.exit_reason_counts:
            count = attr.exit_reason_counts[reason]
            pnl = attr.exit_reason_pnl.get(reason, Decimal(0))
            _row(f"  {reason}", f"{count} | {pnl:+.2f}")
    print()
    print("  Per-strategy PnL:")
    for strat, pnl in sorted(attr.per_strategy_pnl.items(), key=lambda kv: kv[1]):
        _row(f"  {strat}", f"{pnl:+.2f}")
    print()
    print("  Per-symbol PnL:")
    for sym, pnl in sorted(attr.per_symbol_pnl.items(), key=lambda kv: kv[1]):
        _row(f"  {sym}", f"{pnl:+.2f}")
    print()
    print(f"  {attr.lever_verdict}")
    print()


def _severity_key(d: SegmentDiagnosis) -> tuple[int, Decimal]:
    """Severity sort key: genuinely-asymmetric segments first (WR-02).

    ``compute_attribution`` collapses two NON-asymmetric cases to
    ``payoff_ratio == Decimal(0)``: a segment with zero trades, and a segment
    with no losing trades (``avg_loss == 0``). A naive ascending payoff sort
    would float both to the TOP ("most asymmetric"), inverting the report's
    purpose. We bucket those two cases LAST (bucket 1) and rank the genuinely
    asymmetric segments (real losses, payoff < 1) most-asymmetric-first by
    payoff ascending within bucket 0.
    """
    a = d.attribution
    bucket = 1 if (a.total_trades == 0 or a.loss_count == 0) else 0
    return (bucket, a.payoff_ratio)


def build_consolidated_report(diagnoses: dict[str, SegmentDiagnosis]) -> str:
    """Build the severity-ranked consolidated cross-segment report (D-06, WR-02).

    Ranks segments by asymmetry severity (payoff ratio ascending = MOST
    asymmetric first), one markdown row per segment naming the lever and tagging
    thin segments "low-confidence -- informational only". Returns the markdown
    table body (the runbook file itself is written in Plan 04).

    A no-loss segment (``loss_count == 0``) or a zero-trade segment
    (``total_trades == 0``) is NOT asymmetric, so it sorts to the BOTTOM via
    ``_severity_key`` rather than the top (WR-02).
    """
    ordered = sorted(diagnoses.values(), key=_severity_key)
    lines = [
        "| Segment | Type | Trades | Win rate | Payoff | Lever verdict | Confidence |",
        "|---------|------|--------|----------|--------|---------------|------------|",
    ]
    lines.extend(d.report_row() for d in ordered)
    return "\n".join(lines)


def _resolve_type(seg: str, known: dict[str, str]) -> str:
    """Resolve a segment's instrument type with a bond-safe fallback (WR-04).

    Defaulting an unknown ``--segments`` id to ``"stock"`` would route a bond
    segment (disabled, or absent from the enabled-MOEX defaults) through the
    EQUITY branch and emit a ``chandelier`` verdict -- exactly what the bond
    branch (D-04) must NEVER emit for bonds. So we infer ``"bond"`` from the
    ``ru_ofz`` id prefix as a safety net, and warn loudly on any other unknown
    id instead of silently mislabelling it.
    """
    if seg in known:
        return known[seg]
    if seg.startswith("ru_ofz"):
        return "bond"
    print(f"  WARNING: '{seg}' not in enabled MOEX defaults; assuming stock.")
    return "stock"


def _default_moex_segments() -> list[tuple[str, str]]:
    """Enabled post-68 MOEX segments as ``(segment_id, instrument_type)`` pairs.

    Derived from ``config.segments.DEFAULT_SEGMENTS`` filtered to
    ``market == "moex" and enabled`` (pure config read, no network). This is the
    ONLY non-schema finalayze import and is optional context for the default
    segment list -- never a token/fetcher import.
    """
    # Deferred local import: keep this module's top-level imports pure-stdlib +
    # TradeResult only (pure-reader, T-69-04) and avoid eager config side effects.
    from config.segments import DEFAULT_SEGMENTS  # noqa: PLC0415

    return [
        (s.segment_id, s.instrument_type)
        for s in DEFAULT_SEGMENTS
        if s.market == "moex" and s.enabled
    ]


def main() -> None:
    """CLI entry point: batch-read per-segment sidecars and print reports."""
    parser = argparse.ArgumentParser(description="exit-path loss-asymmetry attribution")
    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Iteration run name under --output (e.g. phase69-...). REQUIRED: there is "
            "no default so a forgotten flag errors loudly instead of silently analyzing "
            "a prior phase's stale sidecar."
        ),
    )
    parser.add_argument(
        "--segments",
        default=None,
        help=(
            "Comma-separated segment ids to analyze. If omitted, defaults to every "
            "enabled MOEX segment from config.segments.DEFAULT_SEGMENTS."
        ),
    )
    parser.add_argument("--output", default="results/iterations/", help="Iterations root directory")
    args = parser.parse_args()

    instrument_by_segment = dict(_default_moex_segments())
    if args.segments:
        requested = [s.strip() for s in args.segments.split(",") if s.strip()]
        # WR-04: _resolve_type infers "bond" for ru_ofz* ids absent from the
        # enabled-MOEX defaults and warns on any other unknown id, instead of
        # silently routing a bond through the equity/chandelier branch.
        segments = [(s, _resolve_type(s, instrument_by_segment)) for s in requested]
    else:
        segments = list(instrument_by_segment.items())

    output_root = Path(args.output)
    diagnoses: dict[str, SegmentDiagnosis] = {}
    for segment, instrument_type in segments:
        diag = diagnose_segment(args.run, segment, output_root, instrument_type=instrument_type)
        diagnoses[segment] = diag
        if diag.attribution.total_trades == 0:
            print(
                f"  No trades.jsonl for {segment} @ {args.run} "
                "(re-run run_iteration to generate the instrumented sidecar)."
            )
            continue
        _print_report(diag.attribution, run=args.run, segment=segment)

    if diagnoses:
        print()
        print("  Consolidated severity-ranked report (most-asymmetric first):")
        print()
        print(build_consolidated_report(diagnoses))
        print()


if __name__ == "__main__":
    main()
