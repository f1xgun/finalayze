"""Read-only per-trade attribution for ru_finance (RUFIN-01 / D-01).

Reads the per-segment ``trades.jsonl`` sidecar written by ``run_iteration``
(one ``TradeResult.model_dump_json()`` per line) and reports the loss-asymmetry
attribution that D-01 gates RUFIN-02 on: avg-win / avg-loss + payoff ratio,
exit-reason share (stop / profit_target / time / signal / force_close), and
per-strategy / per-symbol realised PnL. Ends with a named LEVER VERDICT.

This is a PURE FILE READER (Layer 7, T-67-token): it imports ONLY stdlib +
the frozen ``TradeResult`` schema. It MUST NOT import any broker/data fetcher
or reference an API token env var -- it never touches the network or any
secret (the acceptance grep for token symbols is 0).

Usage:
    uv run python scripts/diagnose_ru_finance.py
    uv run python scripts/diagnose_ru_finance.py --run phase66-liquidity-expanded-v2
    uv run python scripts/diagnose_ru_finance.py --segment ru_finance --output results/iterations/
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

from finalayze.core.schemas import TradeResult

# A loss is materially asymmetric when avg-loss magnitude exceeds avg-win by
# this factor -- implicating the stop side (chandelier) rather than the entry.
_LOSS_DOMINANCE_FACTOR = Decimal("1.0")


@dataclass(frozen=True)
class Attribution:
    """Computed ru_finance loss-asymmetry attribution."""

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
    # _name_lever, which keys solely on the avg-win/avg-loss payoff asymmetry).
    stop_hit_share: float
    signal_exit_share: float
    per_strategy_pnl: dict[str, Decimal] = field(default_factory=dict)
    per_symbol_pnl: dict[str, Decimal] = field(default_factory=dict)
    lever_verdict: str = ""


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


def _name_lever(avg_win: Decimal, avg_loss: Decimal) -> str:
    """Name the lever the asymmetry data implicates (the D-01 deliverable).

    NOTE: the verdict is intentionally PAYOFF-DRIVEN -- it keys ONLY on avg-win
    vs avg-loss magnitude. The ``stop_hit_share`` / ``signal_exit_share`` fields
    computed in ``compute_attribution`` and printed by ``_print_report`` are
    DIAGNOSTIC-ONLY context for the human reader; they do NOT gate this verdict.
    (D-04 ACCEPT was decided on this payoff-driven verdict -- do not fold the
    shares into the decision without re-running the diagnostic + re-accepting.)
    """
    loss_mag = abs(avg_loss)
    if loss_mag > avg_win * _LOSS_DOMINANCE_FACTOR:
        return (
            "LEVER VERDICT: ru_finance chandelier stop multiplier "
            "(avg-loss magnitude dominates avg-win -- losers run too far)"
        )
    return "LEVER VERDICT: min_exit_confidence (avg-win too small -- winners are cut early)"


def compute_attribution(trades: list[TradeResult]) -> Attribution:
    """Compute the loss-asymmetry attribution from closed trades."""
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
        lever_verdict=_name_lever(avg_win, avg_loss),
    )


def _print_report(attr: Attribution, *, run: str, segment: str) -> None:
    def _row(name: str, value: str) -> None:
        print(f"  {name:<28} {value:>14}")

    print()
    print(f"  ru_finance attribution: {segment} @ {run}")
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


def main() -> None:
    """CLI entry point: read trades.jsonl and print the attribution."""
    parser = argparse.ArgumentParser(description="ru_finance loss-asymmetry attribution")
    parser.add_argument("--run", default="phase66-liquidity-expanded-v2", help="Iteration run name")
    parser.add_argument("--segment", default="ru_finance", help="Segment to analyze")
    parser.add_argument("--output", default="results/iterations/", help="Iterations root directory")
    args = parser.parse_args()

    path = Path(args.output) / args.run / args.segment / "trades.jsonl"
    trades = load_trades_jsonl(path)
    if not trades:
        print(
            f"  No trades.jsonl at {path} "
            "(re-run run_iteration to generate the instrumented sidecar)."
        )
        return

    attr = compute_attribution(trades)
    _print_report(attr, run=args.run, segment=args.segment)


if __name__ == "__main__":
    main()
