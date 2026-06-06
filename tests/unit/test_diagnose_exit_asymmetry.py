"""Unit tests for scripts/diagnose_exit_asymmetry.py (EXITDIAG-01/03/05).

Generalizes ``test_diagnose_ru_finance.py`` from the single ru_finance segment to
the segment-agnostic ``diagnose_exit_asymmetry`` diagnostic: the equity
chandelier-vs-min_exit_confidence verdict (parameterized on segment id), a
SEPARATE bond verdict branch that NEVER emits a chandelier verdict (D-04), a
thin-sample flag at a documented trade floor (D-05), and a severity-ranked
consolidated cross-segment report (D-06). The equity ``compute_attribution`` /
``load_trades_jsonl`` behavior is unchanged, so those tests are reused verbatim.
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

from scripts.diagnose_exit_asymmetry import (  # noqa: E402
    _LOSS_DOMINANCE_FACTOR,
    _THIN_TRADE_FLOOR,
    SegmentDiagnosis,
    build_consolidated_report,
    compute_attribution,
    diagnose_attribution,
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

# Equity "winners cut early" case: avg-win magnitude >= avg-loss magnitude.
WIN_FAT = Decimal(300)
LOSS_TINY = Decimal(-10)

# Thin-sample handling (D-05): a count below the floor and one at-or-above it.
COUNT_BELOW_FLOOR = _THIN_TRADE_FLOOR - 1
COUNT_AT_FLOOR = _THIN_TRADE_FLOOR

# Consolidated-report severity ordering: a clearly more-asymmetric segment vs a
# milder one (payoff ratio ascending = most-asymmetric first).
SEVERE_LOSS = Decimal(-500)
MILD_LOSS = Decimal(-120)

SEGMENT_A = "ru_finance"
SEGMENT_B = "ru_tech"
SEGMENT_BOND = "ru_ofz_pd"


def _trade(
    pnl: Decimal,
    *,
    exit_reason: str,
    entry_strategy: str,
    instrument_type: str = "stock",
    symbol: str = "SBER",
) -> TradeResult:
    exit_price = ENTRY + pnl  # qty == 1 so pnl == exit - entry
    return TradeResult(
        signal_id=uuid4(),
        symbol=symbol,
        side="SELL",
        quantity=QTY,
        entry_price=ENTRY,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl / ENTRY,
        hold_bars=5,
        instrument_type=instrument_type,
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


def _winners_cut_early_trades() -> list[TradeResult]:
    # avg-win magnitude (300) > avg-loss magnitude (10) -> min_exit_confidence.
    return [
        _trade(WIN_FAT, exit_reason="signal", entry_strategy="momentum"),
        _trade(WIN_FAT, exit_reason="signal", entry_strategy="momentum"),
        _trade(LOSS_TINY, exit_reason="stop", entry_strategy="momentum"),
    ]


def _bond_trades() -> list[TradeResult]:
    # Bond exits dominated by yield_stop (STOP) so the bond branch names a STOP lever.
    return [
        _trade(
            WIN_A,
            exit_reason="stop",
            entry_strategy="bond_duration_rotation",
            instrument_type="bond",
            symbol="SU26244RMFS2",
        ),
        _trade(
            LOSS_SMALL,
            exit_reason="stop",
            entry_strategy="bond_duration_rotation",
            instrument_type="bond",
            symbol="SU26241RMFS8",
        ),
        _trade(
            LOSS_BIG,
            exit_reason="stop",
            entry_strategy="bond_duration_rotation",
            instrument_type="bond",
            symbol="SU26244RMFS2",
        ),
        _trade(
            WIN_B,
            exit_reason="time",
            entry_strategy="bond_duration_rotation",
            instrument_type="bond",
            symbol="SU26241RMFS8",
        ),
    ]


# ---------------------------------------------------------------------------
# Reused verbatim: equity compute / round-trip behavior is unchanged
# ---------------------------------------------------------------------------
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
        assert attr.payoff_ratio == EXPECTED_AVG_WIN / abs(EXPECTED_AVG_LOSS)

    def test_exit_reason_share(self) -> None:
        attr = compute_attribution(_sample_trades())
        assert attr.exit_reason_counts["stop"] == EXPECTED_STOP_COUNT
        assert attr.exit_reason_counts["signal"] == EXPECTED_SIGNAL_COUNT
        assert attr.stop_hit_share == EXPECTED_STOP_COUNT / EXPECTED_TRADES
        assert attr.signal_exit_share == EXPECTED_SIGNAL_COUNT / EXPECTED_TRADES

    def test_per_strategy_pnl(self) -> None:
        attr = compute_attribution(_sample_trades())
        assert attr.per_strategy_pnl["momentum"] == WIN_A + LOSS_BIG
        assert attr.per_strategy_pnl["rsi2_connors"] == WIN_B + LOSS_SMALL

    def test_per_symbol_pnl(self) -> None:
        attr = compute_attribution(_sample_trades())
        assert attr.per_symbol_pnl["SBER"] == WIN_A + WIN_B + LOSS_BIG + LOSS_SMALL


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


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    loaded = load_trades_jsonl(tmp_path / "absent.jsonl")
    assert loaded == []


# ---------------------------------------------------------------------------
# New: equity verdict is parameterized on segment id (no hardcoded ru_finance)
# ---------------------------------------------------------------------------
class TestEquityVerdict:
    """The equity verdict names chandelier vs min_exit_confidence + the segment id."""

    def test_chandelier_when_losses_dominate(self) -> None:
        # avg-loss magnitude (210) >> avg-win (75) -> chandelier stop implicated.
        attr = diagnose_attribution(SEGMENT_A, _sample_trades(), instrument_type="stock")
        verdict = attr.lever_verdict.lower()
        assert "chandelier" in verdict
        assert SEGMENT_A in attr.lever_verdict

    def test_min_exit_confidence_when_winners_cut_early(self) -> None:
        # avg-win magnitude (300) >> avg-loss magnitude (10) -> winners cut early.
        attr = diagnose_attribution(SEGMENT_B, _winners_cut_early_trades(), instrument_type="stock")
        verdict = attr.lever_verdict.lower()
        assert "min_exit_confidence" in verdict
        assert "chandelier" not in verdict
        assert SEGMENT_B in attr.lever_verdict


# ---------------------------------------------------------------------------
# New: bond verdict NEVER names a chandelier (D-04) and names a bond lever
# ---------------------------------------------------------------------------
class TestBondVerdict:
    """Bond verdict must not contain 'chandelier' and must name a bond lever."""

    _BOND_LEVERS = (
        "yield_stop",
        "max_hold",
        "rebalance",
        "max_positions",
        "duration",
        "carry",
    )

    def test_bond_verdict_never_chandelier(self) -> None:
        attr = diagnose_attribution(SEGMENT_BOND, _bond_trades(), instrument_type="bond")
        assert "chandelier" not in attr.lever_verdict.lower()

    def test_bond_verdict_names_a_bond_lever(self) -> None:
        attr = diagnose_attribution(SEGMENT_BOND, _bond_trades(), instrument_type="bond")
        verdict = attr.lever_verdict.lower()
        assert any(lever in verdict for lever in self._BOND_LEVERS)
        assert SEGMENT_BOND in attr.lever_verdict


# ---------------------------------------------------------------------------
# New: thin-sample flag (D-05) at a documented floor
# ---------------------------------------------------------------------------
class TestThinSampleFlag:
    """A segment below the trade floor is flagged low-confidence / informational."""

    def test_below_floor_flagged_thin(self) -> None:
        trades = [
            _trade(WIN_A, exit_reason="stop", entry_strategy="momentum")
            for _ in range(COUNT_BELOW_FLOOR)
        ]
        attr = compute_attribution(trades)
        diag = SegmentDiagnosis(
            segment=SEGMENT_A,
            instrument_type="stock",
            attribution=attr,
            thin_sample=attr.total_trades < _THIN_TRADE_FLOOR,
        )
        assert diag.thin_sample is True
        row = diag.report_row()
        assert "low-confidence" in row.lower() or "informational" in row.lower()

    def test_at_floor_not_flagged_thin(self) -> None:
        trades = [
            _trade(WIN_A, exit_reason="stop", entry_strategy="momentum")
            for _ in range(COUNT_AT_FLOOR)
        ]
        attr = compute_attribution(trades)
        diag = SegmentDiagnosis(
            segment=SEGMENT_A,
            instrument_type="stock",
            attribution=attr,
            thin_sample=attr.total_trades < _THIN_TRADE_FLOOR,
        )
        assert diag.thin_sample is False
        row = diag.report_row()
        assert "low-confidence" not in row.lower()


# ---------------------------------------------------------------------------
# New: consolidated severity-ranked report (D-06)
# ---------------------------------------------------------------------------
class TestConsolidatedReport:
    """The report ranks segments most-asymmetric-first and names each lever."""

    def _severe_trades(self) -> list[TradeResult]:
        # payoff = 75 / 500 = 0.15 (very asymmetric).
        return [
            _trade(WIN_A, exit_reason="signal", entry_strategy="momentum"),
            _trade(WIN_B, exit_reason="signal", entry_strategy="momentum"),
            _trade(SEVERE_LOSS, exit_reason="stop", entry_strategy="momentum"),
        ]

    def _mild_trades(self) -> list[TradeResult]:
        # payoff = 75 / 120 = 0.625 (milder asymmetry).
        return [
            _trade(WIN_A, exit_reason="signal", entry_strategy="momentum"),
            _trade(WIN_B, exit_reason="signal", entry_strategy="momentum"),
            _trade(MILD_LOSS, exit_reason="stop", entry_strategy="momentum"),
        ]

    def _diag(self, segment: str, trades: list[TradeResult]) -> SegmentDiagnosis:
        attr = diagnose_attribution(segment, trades, instrument_type="stock")
        return SegmentDiagnosis(
            segment=segment,
            instrument_type="stock",
            attribution=attr,
            thin_sample=attr.total_trades < _THIN_TRADE_FLOOR,
        )

    def test_report_orders_most_asymmetric_first(self) -> None:
        diagnoses = {
            SEGMENT_B: self._diag(SEGMENT_B, self._mild_trades()),
            SEGMENT_A: self._diag(SEGMENT_A, self._severe_trades()),
        }
        report = build_consolidated_report(diagnoses)
        # The most-asymmetric segment (SEGMENT_A, payoff 0.15) appears before
        # the milder one (SEGMENT_B, payoff 0.625).
        assert report.index(SEGMENT_A) < report.index(SEGMENT_B)

    def test_report_includes_each_lever(self) -> None:
        diagnoses = {
            SEGMENT_A: self._diag(SEGMENT_A, self._severe_trades()),
            SEGMENT_B: self._diag(SEGMENT_B, self._mild_trades()),
        }
        report = build_consolidated_report(diagnoses)
        assert "chandelier" in report.lower()
        assert SEGMENT_A in report
        assert SEGMENT_B in report


# ---------------------------------------------------------------------------
# New: frozen factor pin (Pitfall 4 / T-69-05)
# ---------------------------------------------------------------------------
class TestFrozenFactor:
    """The break-even factor is pinned at exactly 1.0 (D-04 ACCEPT boundary)."""

    def test_loss_dominance_factor_is_one(self) -> None:
        assert Decimal("1.0") == _LOSS_DOMINANCE_FACTOR
