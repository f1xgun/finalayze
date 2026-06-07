"""Phase 71 backtest-iteration A/B gate (D-16, CLAUDE.md #4).

Proves the total-return dividend hook moves the equity curve ONLY by credited
income with NO other change to trades, entries, exits, or price-PnL.

BASELINE leg:  run_portfolio(..., dividend_schedule=None)          (pre-71, price-only)
CANDIDATE leg: run_portfolio(..., dividend_schedule={(SBER, ex): gross})

Both legs share the SAME segment / window / seed / strategy. SBER is BOUGHT at bar 30
and SOLD at bar 35 (the proven StubStrategy + us_large_cap cadence the existing
test_backtest_engine.py suite exercises), so a real SBER position is HELD across the
SBER ex-date (placed on bar 32). The held universe (SBER) overlaps the committed
moex_dividends.yaml snapshot (SBER ex 2023-05-11, gross 25.0, paid) so the credit path
fires. A deterministic strategy on a reproducible series isolates the proof to
"curve-moves-only-by-income" (no live Tinkoff dependency, no transient gRPC FD-shutdown
risk; the must-have per the plan is the curve-delta proof, not a specific data source).

Logs both legs under results/iterations/phase71-ab-{baseline,candidate}/ with a verdict.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from finalayze.backtest.dividend_schedule import load_dividend_schedule
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.constants import NDFL_RATE
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

if TYPE_CHECKING:
    from finalayze.core.schemas import PortfolioState, TradeResult

# ---- Deterministic A/B fixture (named constants -- no magic numbers) --------
_SBER = "SBER"
_SEGMENT = "us_large_cap"  # proven non-gated path that produces trades (test suite)
_EX_DATE = date(2023, 5, 11)  # committed snapshot: SBER ex 2023-05-11, 25.0, paid
_INITIAL_CASH = Decimal(100_000)
_CANDLE_COUNT = 40
_BUY_BAR = 30  # StubStrategy buys here
_SELL_BAR = 35  # ...and sells here -> SBER held across the ex-date
_EX_BAR = 32  # ex-date bar (held between buy and sell)
_BASE_PRICE = 100
# Anchor bar 0 so that bar _EX_BAR lands exactly on _EX_DATE.
_FIRST_BAR = datetime(2023, 5, 11, 14, 30, tzinfo=UTC) - timedelta(days=_EX_BAR)
_ITER_DIR = Path(__file__).resolve().parent.parent / "results" / "iterations"


class StubStrategy(BaseStrategy):
    """BUY at bar _BUY_BAR, SELL at bar _SELL_BAR (identical in both legs)."""

    @property
    def name(self) -> str:
        return "stub_ab"

    def supported_segments(self) -> list[str]:
        return [_SEGMENT]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **_kwargs: object,
    ) -> Signal | None:
        idx = len(candles) - 1
        if idx == _BUY_BAR:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                strategy_payload={"momentum": 1.0},
                reasoning="A/B buy",
            )
        if idx == _SELL_BAR:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.SELL,
                confidence=0.8,
                strategy_payload={"momentum": -1.0},
                reasoning="A/B sell",
            )
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        del segment_id  # required by the ABC; deterministic stub has no params
        return {}


def _make_series() -> list[Candle]:
    """Upward-trending SBER series; bar _EX_BAR lands exactly on _EX_DATE."""
    out: list[Candle] = []
    for i in range(_CANDLE_COUNT):
        price = Decimal(_BASE_PRICE + i)
        out.append(
            Candle(
                symbol=_SBER,
                market_id="us",
                timeframe="1d",
                timestamp=_FIRST_BAR + timedelta(days=i),
                open=price,
                high=price + Decimal(2),
                low=price - Decimal(2),
                close=price + Decimal(1),
                volume=1_000_000,
            )
        )
    return out


def _strip_ids(trades: list[TradeResult]) -> list[dict[str, object]]:
    """Phase-69 byte-identical convention: drop the non-deterministic signal_id UUID."""
    return [t.model_dump(exclude={"signal_id"}) for t in trades]


def _run_leg(
    candles: list[Candle], schedule: dict[tuple[str, date], Decimal] | None
) -> tuple[list[TradeResult], list[PortfolioState]]:
    engine = BacktestEngine(strategy=StubStrategy(), initial_cash=_INITIAL_CASH)
    return engine.run_portfolio(
        symbols=[_SBER],
        segment_id=_SEGMENT,
        candles_by_symbol={_SBER: candles},
        dividend_schedule=schedule,
    )


def _write_iteration(
    name: str,
    *,
    verdict: str,
    trade_count: int,
    final_equity: Decimal,
    cumulative_net_income: Decimal,
    git_sha: str,
    notes: dict[str, object],
) -> None:
    d = _ITER_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "phase": "71-05",
        "leg": "baseline" if "baseline" in name else "candidate",
        "segment": _SEGMENT,
        "symbols": [_SBER],
        "window": {
            "first_bar": _FIRST_BAR.date().isoformat(),
            "n_bars": _CANDLE_COUNT,
            "buy_bar": _BUY_BAR,
            "sell_bar": _SELL_BAR,
            "ex_bar": _EX_BAR,
        },
        "ex_date": _EX_DATE.isoformat(),
        "seed": "deterministic StubStrategy (buy@30/sell@35, no RNG)",
        "trade_count": trade_count,
        "final_equity": str(final_equity),
        "cumulative_net_income": str(cumulative_net_income),
        "verdict": verdict,
        "notes": notes,
        "git_sha": git_sha,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
    }
    (d / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:  # noqa: PLR0915
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    candles = _make_series()

    # The committed snapshot's SBER gross for the golden ex-date (load -> filter to SBER).
    full_schedule = load_dividend_schedule()
    sber_gross = full_schedule[(_SBER, _EX_DATE)]
    candidate_schedule = {(_SBER, _EX_DATE): sber_gross}

    base_trades, base_snaps = _run_leg(candles, None)
    cand_trades, cand_snaps = _run_leg(candles, candidate_schedule)

    # Reconstruct the held qty on the ex-date bar from the baseline snapshots
    # (positions are identical in both legs -> the income hook never alters trades).
    held_on_ex = Decimal(0)
    for sb in base_snaps:
        if sb.timestamp.date() == _EX_DATE:
            held_on_ex = sb.positions.get(_SBER, Decimal(0))
            break
    gross = sber_gross * held_on_ex
    expected_net = gross - gross * NDFL_RATE
    cumulative_net_income = expected_net  # single ex-date in this window

    # ---- Assertions (D-16) -------------------------------------------------
    # 1. Trades + entry/exit timing IDENTICAL (price-PnL untouched).
    trades_identical = _strip_ids(base_trades) == _strip_ids(cand_trades)

    # 2. Per-bar equity delta: 0 before the ex-date, exactly +net on/after it
    #    (the credit lands once on the ex-date and the cash is carried forward).
    deltas: list[tuple[str, str]] = []
    pre_exdate_all_zero = True
    on_after_matches_income = True
    for sb, sd in zip(base_snaps, cand_snaps, strict=True):
        delta = sd.equity - sb.equity
        deltas.append((sb.timestamp.date().isoformat(), str(delta)))
        if sb.timestamp.date() < _EX_DATE:
            if delta != Decimal(0):
                pre_exdate_all_zero = False
        elif delta != expected_net:
            on_after_matches_income = False

    # 3. Final equity delta equals the cumulative net credited income (Decimal-exact).
    final_delta = cand_snaps[-1].equity - base_snaps[-1].equity
    final_delta_matches = final_delta == cumulative_net_income

    passed = (
        trades_identical
        and pre_exdate_all_zero
        and on_after_matches_income
        and final_delta_matches
        and held_on_ex > 0
    )
    verdict = "PASS" if passed else "REJECT"

    notes = {
        "trades_identical": trades_identical,
        "pre_exdate_all_zero": pre_exdate_all_zero,
        "on_after_delta_equals_net_income": on_after_matches_income,
        "final_delta_matches_cumulative_net_income": final_delta_matches,
        "held_qty_on_ex_date": str(held_on_ex),
        "sber_gross_per_share": str(sber_gross),
        "gross_total": str(gross),
        "expected_net_after_ndfl": str(expected_net),
        "final_equity_delta": str(final_delta),
        "per_bar_equity_delta": deltas,
        "baseline_trade_count": len(base_trades),
        "candidate_trade_count": len(cand_trades),
    }

    _write_iteration(
        "phase71-ab-baseline",
        verdict=verdict,
        trade_count=len(base_trades),
        final_equity=base_snaps[-1].equity,
        cumulative_net_income=Decimal(0),
        git_sha=git_sha,
        notes={**notes, "dividend_schedule": "None (pre-71, price-only)"},
    )
    _write_iteration(
        "phase71-ab-candidate",
        verdict=verdict,
        trade_count=len(cand_trades),
        final_equity=cand_snaps[-1].equity,
        cumulative_net_income=cumulative_net_income,
        git_sha=git_sha,
        notes={**notes, "dividend_schedule": "load_dividend_schedule()[SBER, 2023-05-11]"},
    )

    # Append both legs to the shared history.jsonl ledger.
    hist = _ITER_DIR / "history.jsonl"
    with hist.open("a", encoding="utf-8") as fh:
        for leg, eq, inc in (
            ("phase71-ab-baseline", base_snaps[-1].equity, Decimal(0)),
            ("phase71-ab-candidate", cand_snaps[-1].equity, cumulative_net_income),
        ):
            fh.write(
                json.dumps(
                    {
                        "name": leg,
                        "created_at": datetime.now(tz=timezone.utc).isoformat(),  # noqa: UP017
                        "git_sha": git_sha,
                        "verdict": verdict,
                        "final_equity": str(eq),
                        "cumulative_net_income": str(inc),
                        "trade_count": len(base_trades),
                    }
                )
                + "\n"
            )

    print("=" * 64)
    print("PHASE 71 DIVIDEND-HOOK A/B GATE")
    print("=" * 64)
    print(f"segment={_SEGMENT}  symbol={_SBER}  ex_date={_EX_DATE}  bars={_CANDLE_COUNT}")
    print(f"buy_bar={_BUY_BAR} sell_bar={_SELL_BAR} ex_bar={_EX_BAR}")
    print(f"held_qty_on_ex_date={held_on_ex}  gross/share={sber_gross}  gross_total={gross}")
    print(f"expected_net_after_NDFL({NDFL_RATE})={expected_net}")
    print(f"baseline_trades={len(base_trades)}  candidate_trades={len(cand_trades)}")
    print(f"trades_identical={trades_identical}")
    print(f"pre_exdate_all_zero={pre_exdate_all_zero}")
    print(f"on/after_delta==net_income={on_after_matches_income}")
    print(f"final_equity_delta={final_delta}  cumulative_net_income={cumulative_net_income}")
    print(f"final_delta_matches={final_delta_matches}")
    print("per-bar equity delta (candidate - baseline):")
    for d_str, delta_str in deltas:
        marker = "  <- ex-date" if d_str == _EX_DATE.isoformat() else ""
        print(f"  {d_str}: {delta_str}{marker}")
    print("-" * 64)
    print(f"VERDICT: {verdict}")
    print("=" * 64)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
