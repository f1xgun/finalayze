"""Unit tests for scripts/diagnose_ru_finance.py attribution (RUFIN-01 / D-01).

Drives the importable ``compute_attribution`` function with in-memory closed
trades and round-trips a temp ``trades.jsonl`` written one ``model_dump_json``
per line (inverse of the run_iteration sidecar writer).
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from finalayze.core.schemas import TradeResult

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.diagnose_ru_finance import (  # noqa: E402
    compute_attribution,
    load_trades_jsonl,
)

# ---------------------------------------------------------------------------
# Constants (no magic numbers -- ruff PLR2004)
# ---------------------------------------------------------------------------
WIN_A = Decimal(100)
WIN_B = Decimal(50)
LOSS_BIG = Decimal(-400)
LOSS_SMALL = Decimal(-20)
ENTRY = Decimal("100.00")
QTY = Decimal(1)

EXPECTED_TRADES = 4
EXPECTED_WINS = 2
EXPECTED_LOSSES = 2
EXPECTED_AVG_WIN = (WIN_A + WIN_B) / EXPECTED_WINS  # 75
EXPECTED_AVG_LOSS = (LOSS_BIG + LOSS_SMALL) / EXPECTED_LOSSES  # -210
EXPECTED_STOP_COUNT = 2
EXPECTED_SIGNAL_COUNT = 1
EXPECTED_FORCE_COUNT = 1


def _trade(pnl: Decimal, *, exit_reason: str, entry_strategy: str) -> TradeResult:
    exit_price = ENTRY + pnl  # qty == 1 so pnl == exit - entry
    return TradeResult(
        signal_id=uuid4(),
        symbol="SBER",
        side="SELL",
        quantity=QTY,
        entry_price=ENTRY,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl / ENTRY,
        hold_bars=5,
        exit_reason=exit_reason,
        entry_strategy=entry_strategy,
    )


def _sample_trades() -> list[TradeResult]:
    return [
        _trade(WIN_A, exit_reason="signal", entry_strategy="momentum"),
        _trade(WIN_B, exit_reason="profit_target", entry_strategy="rsi2_connors"),
        _trade(LOSS_BIG, exit_reason="stop", entry_strategy="momentum"),
        _trade(LOSS_SMALL, exit_reason="stop", entry_strategy="rsi2_connors"),
    ]


class TestComputeAttribution:
    """The compute function is importable and returns the asymmetry split."""

    def test_win_loss_split(self) -> None:
        attr = compute_attribution(_sample_trades())
        assert attr.total_trades == EXPECTED_TRADES
        assert attr.win_count == EXPECTED_WINS
        assert attr.loss_count == EXPECTED_LOSSES
        assert attr.avg_win == EXPECTED_AVG_WIN
        assert attr.avg_loss == EXPECTED_AVG_LOSS

    def test_payoff_ratio(self) -> None:
        attr = compute_attribution(_sample_trades())
        # payoff = avg_win / |avg_loss| = 75 / 210
        assert attr.payoff_ratio == EXPECTED_AVG_WIN / abs(EXPECTED_AVG_LOSS)

    def test_exit_reason_share(self) -> None:
        attr = compute_attribution(_sample_trades())
        assert attr.exit_reason_counts["stop"] == EXPECTED_STOP_COUNT
        assert attr.exit_reason_counts["signal"] == EXPECTED_SIGNAL_COUNT
        # stop-hit share = 2 / 4
        assert attr.stop_hit_share == EXPECTED_STOP_COUNT / EXPECTED_TRADES
        assert attr.signal_exit_share == EXPECTED_SIGNAL_COUNT / EXPECTED_TRADES

    def test_per_strategy_pnl(self) -> None:
        attr = compute_attribution(_sample_trades())
        # momentum: +100 (signal) + -400 (stop) = -300
        assert attr.per_strategy_pnl["momentum"] == WIN_A + LOSS_BIG
        # rsi2_connors: +50 (profit) + -20 (stop) = +30
        assert attr.per_strategy_pnl["rsi2_connors"] == WIN_B + LOSS_SMALL

    def test_per_symbol_pnl(self) -> None:
        attr = compute_attribution(_sample_trades())
        assert attr.per_symbol_pnl["SBER"] == WIN_A + WIN_B + LOSS_BIG + LOSS_SMALL

    def test_lever_verdict_names_chandelier_when_losses_dominate(self) -> None:
        # avg loss magnitude (210) >> avg win (75) -> chandelier stop implicated
        attr = compute_attribution(_sample_trades())
        assert "chandelier" in attr.lever_verdict.lower()


class TestRoundTripFromJsonl:
    """A temp trades.jsonl is read back and produces identical attribution."""

    def test_round_trip_matches_in_memory(self, tmp_path: Path) -> None:
        trades = _sample_trades()
        path = tmp_path / "trades.jsonl"
        path.write_text("".join(t.model_dump_json() + "\n" for t in trades))

        loaded = load_trades_jsonl(path)
        assert len(loaded) == EXPECTED_TRADES

        from_mem = compute_attribution(trades)
        from_disk = compute_attribution(loaded)
        assert from_disk.avg_win == from_mem.avg_win
        assert from_disk.avg_loss == from_mem.avg_loss
        assert from_disk.exit_reason_counts == from_mem.exit_reason_counts
        assert from_disk.per_strategy_pnl == from_mem.per_strategy_pnl

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        loaded = load_trades_jsonl(tmp_path / "absent.jsonl")
        assert loaded == []
