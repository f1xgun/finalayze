"""Tests for scripts/run_bond_iteration.py.

TDD tests for the bond iteration harness. Currently covers the per-segment
``trades.jsonl`` sidecar writer (EXITDIAG-02 / D-04): the bond harness must emit
a one-``TradeResult``-per-line JSONL alongside ``summary.json`` so the
cross-segment exit-asymmetry diagnostic (Plan 02) can read OFZ trades the same
way it reads equity trades. RED until ``run_bond_iteration._write_trades_jsonl``
exists.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

# config/ lives at the project root, not under src/ -- make it importable so
# `import scripts.run_bond_iteration` resolves the same way the harness runs.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_bond_iteration import _write_trades_jsonl  # noqa: E402

from finalayze.core.schemas import ExitReason, TradeResult  # noqa: E402

_EXPECTED_TRADE_COUNT = 3
_QTY = Decimal(100)
_CARRY_STRATEGY = "bond_carry"
_DURATION_STRATEGY = "bond_duration_rotation"


def _make_bond_trade(
    *,
    symbol: str,
    pnl: Decimal,
    exit_reason: str | None,
    entry_strategy: str | None,
) -> TradeResult:
    """Build a synthetic bond TradeResult for round-trip testing."""
    return TradeResult(
        signal_id=uuid4(),
        symbol=symbol,
        side="SELL",
        quantity=_QTY,
        entry_price=Decimal("850.0"),
        exit_price=Decimal("840.0"),
        pnl=pnl,
        pnl_pct=pnl / Decimal("85000.0"),
        hold_bars=10,
        coupon_income=Decimal("355.0"),
        instrument_type="bond",
        exit_reason=exit_reason,
        entry_strategy=entry_strategy,
    )


class TestBondTradesJsonlWriter:
    """The bond harness writer emits round-trippable trades.jsonl lines."""

    def _sample_trades(self) -> list[TradeResult]:
        return [
            _make_bond_trade(
                symbol="SU26243RMFS4",
                pnl=Decimal("-1200.0"),
                exit_reason=ExitReason.STOP.value,
                entry_strategy=_DURATION_STRATEGY,
            ),
            _make_bond_trade(
                symbol="SU26244RMFS2",
                pnl=Decimal("800.0"),
                exit_reason=ExitReason.TIME.value,
                entry_strategy=_DURATION_STRATEGY,
            ),
            _make_bond_trade(
                symbol="SU29007RMFS0",
                pnl=Decimal("0.0"),
                exit_reason=ExitReason.FORCE_CLOSE.value,
                entry_strategy=_CARRY_STRATEGY,
            ),
        ]

    def test_writes_one_line_per_trade(self, tmp_path: Path) -> None:
        """One model_dump_json line is written per trade."""
        out = tmp_path / "ru_ofz_pd" / "trades.jsonl"
        _write_trades_jsonl(out, self._sample_trades())

        assert out.exists()
        lines = out.read_text().splitlines()
        assert len(lines) == _EXPECTED_TRADE_COUNT

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """The writer mkdirs the parent segment directory."""
        out = tmp_path / "nested" / "seg" / "trades.jsonl"
        _write_trades_jsonl(out, self._sample_trades())
        assert out.exists()

    def test_round_trips_via_model_validate(self, tmp_path: Path) -> None:
        """Each line round-trips back to a TradeResult, metadata intact."""
        out = tmp_path / "ru_ofz_pd" / "trades.jsonl"
        trades = self._sample_trades()
        _write_trades_jsonl(out, trades)

        loaded = [
            TradeResult.model_validate(json.loads(line)) for line in out.read_text().splitlines()
        ]
        assert len(loaded) == _EXPECTED_TRADE_COUNT
        assert [t.exit_reason for t in loaded] == [
            ExitReason.STOP.value,
            ExitReason.TIME.value,
            ExitReason.FORCE_CLOSE.value,
        ]
        assert [t.entry_strategy for t in loaded] == [
            _DURATION_STRATEGY,
            _DURATION_STRATEGY,
            _CARRY_STRATEGY,
        ]
        assert all(t.instrument_type == "bond" for t in loaded)

    def test_empty_trades_writes_empty_file(self, tmp_path: Path) -> None:
        """An empty trade list writes a zero-line sidecar (no crash)."""
        out = tmp_path / "ru_ofz_pk" / "trades.jsonl"
        _write_trades_jsonl(out, [])
        assert out.exists()
        assert out.read_text() == ""
