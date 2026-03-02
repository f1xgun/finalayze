"""Unit tests for DecisionJournal and BacktestEngine journal integration."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from finalayze.backtest.decision_journal import (
    CandleSnapshot,
    DecisionJournal,
    DecisionRecord,
    FinalAction,
    StrategySignalRecord,
)
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

# ── Constants ──────────────────────────────────────────────────────────────────
INITIAL_CASH = Decimal(100_000)
CANDLE_COUNT = 40
TRADE_DAY_BUY = 30
TRADE_DAY_SELL = 35


# ── Helpers ────────────────────────────────────────────────────────────────────
def _make_candle_series(count: int = CANDLE_COUNT) -> list[Candle]:
    """Create an upward-trending candle series."""
    base_price = Decimal(100)
    candles: list[Candle] = []
    for i in range(count):
        price = base_price + Decimal(i)
        candles.append(
            Candle(
                symbol="TEST",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1, 14, 30, tzinfo=UTC) + timedelta(days=i),
                open=price,
                high=price + Decimal(2),
                low=price - Decimal(2),
                close=price + Decimal(1),
                volume=1_000_000,
            )
        )
    return candles


class StubStrategy(BaseStrategy):
    """Emits BUY at candle index TRADE_DAY_BUY, SELL at TRADE_DAY_SELL."""

    @property
    def name(self) -> str:
        return "stub"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        **kwargs: object,
    ) -> Signal | None:
        idx = len(candles) - 1
        if idx == TRADE_DAY_BUY:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={"momentum": 1.0},
                reasoning="Test buy signal",
            )
        if idx == TRADE_DAY_SELL:
            return Signal(
                strategy_name=self.name,
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.SELL,
                confidence=0.8,
                features={"momentum": -1.0},
                reasoning="Test sell signal",
            )
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


class SilentStrategy(BaseStrategy):
    """Always returns None — no signals."""

    @property
    def name(self) -> str:
        return "silent"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        **kwargs: object,
    ) -> Signal | None:
        return None

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


class AlwaysBuyStrategy(BaseStrategy):
    """Always emits a BUY signal."""

    @property
    def name(self) -> str:
        return "always_buy"

    def supported_segments(self) -> list[str]:
        return ["us_large_cap"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        **kwargs: object,
    ) -> Signal | None:
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id="us",
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=0.9,
            features={"momentum": 1.0},
            reasoning="Always buy",
        )

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


# ── Journal unit tests ─────────────────────────────────────────────────────────
class TestJournalRecordAndAccess:
    def test_journal_record_and_access(self) -> None:
        journal = DecisionJournal()
        record = DecisionJournal.make_record(
            timestamp=datetime(2024, 6, 1, 14, 30, tzinfo=UTC),
            symbol="AAPL",
            segment_id="us_large_cap",
            final_action=FinalAction.BUY,
        )
        journal.record(record)

        assert len(journal.records) == 1
        assert journal.records[0].symbol == "AAPL"
        assert journal.records[0].final_action == FinalAction.BUY


class TestJournalFlushJsonl:
    def test_journal_flush_jsonl(self, tmp_path: Path) -> None:
        output = tmp_path / "journal.jsonl"
        journal = DecisionJournal(output_path=output)
        for i in range(3):
            journal.record(
                DecisionJournal.make_record(
                    timestamp=datetime(2024, 6, 1, 14, 30, tzinfo=UTC) + timedelta(days=i),
                    symbol="AAPL",
                    segment_id="us_large_cap",
                    final_action=FinalAction.SKIP,
                    skip_reason="no_signal",
                )
            )
        journal.flush()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3
        # Each line must be valid JSON
        for line in lines:
            data = json.loads(line)
            assert data["symbol"] == "AAPL"
            assert data["final_action"] == "SKIP"


class TestJournalFlushNoPath:
    def test_journal_flush_no_path(self) -> None:
        journal = DecisionJournal(output_path=None)
        journal.record(
            DecisionJournal.make_record(
                timestamp=datetime(2024, 6, 1, 14, 30, tzinfo=UTC),
                symbol="AAPL",
                segment_id="us_large_cap",
                final_action=FinalAction.BUY,
            )
        )
        # Should not raise
        journal.flush()
        assert len(journal.records) == 1


class TestDecisionRecordSerializable:
    def test_decision_record_serializable(self) -> None:
        record = DecisionRecord(
            record_id=uuid4(),
            timestamp=datetime(2024, 6, 1, 14, 30, tzinfo=UTC),
            symbol="AAPL",
            segment_id="us_large_cap",
            final_action=FinalAction.SELL,
            portfolio_equity=Decimal(100000),
            portfolio_cash=Decimal(50000),
            strategy_signals=[
                StrategySignalRecord(
                    strategy_name="momentum",
                    direction="BUY",
                    confidence=0.7,
                    weight=Decimal("1.0"),
                    contribution=Decimal("0.7"),
                )
            ],
            recent_candles=[
                CandleSnapshot(
                    timestamp=datetime(2024, 6, 1, 14, 30, tzinfo=UTC),
                    open=Decimal("150.00"),
                    high=Decimal("152.00"),
                    low=Decimal("148.00"),
                    close=Decimal("151.00"),
                    volume=1_000_000,
                )
            ],
        )
        json_str = record.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "AAPL"
        assert parsed["final_action"] == "SELL"
        assert len(parsed["strategy_signals"]) == 1
        assert len(parsed["recent_candles"]) == 1


class TestJournalSummaryCounts:
    def test_journal_summary_counts(self) -> None:
        journal = DecisionJournal()
        ts = datetime(2024, 6, 1, 14, 30, tzinfo=UTC)
        for action, reason in [
            (FinalAction.BUY, None),
            (FinalAction.BUY, None),
            (FinalAction.SELL, None),
            (FinalAction.SKIP, "no_signal"),
            (FinalAction.SKIP, "no_signal"),
            (FinalAction.SKIP, "pre_trade_check_failed"),
        ]:
            journal.record(
                DecisionJournal.make_record(
                    timestamp=ts,
                    symbol="AAPL",
                    segment_id="us_large_cap",
                    final_action=action,
                    skip_reason=reason,
                )
            )

        summary = journal.summary()
        assert summary["total_decisions"] == 6
        assert summary["action_counts"]["BUY"] == 2
        assert summary["action_counts"]["SELL"] == 1
        assert summary["action_counts"]["SKIP"] == 3
        assert summary["top_skip_reasons"]["no_signal"] == 2
        assert summary["top_skip_reasons"]["pre_trade_check_failed"] == 1


# ── Engine integration tests ──────────────────────────────────────────────────
class TestEngineWithJournalRecordsBuy:
    def test_engine_with_journal_records_buy(self) -> None:
        journal = DecisionJournal()
        engine = BacktestEngine(
            strategy=StubStrategy(),
            initial_cash=INITIAL_CASH,
            decision_journal=journal,
        )
        candles = _make_candle_series()
        trades, _snapshots = engine.run(
            symbol="TEST",
            segment_id="us_large_cap",
            candles=candles,
        )

        assert len(trades) >= 1
        buy_records = [r for r in journal.records if r.final_action == FinalAction.BUY]
        assert len(buy_records) >= 1
        assert buy_records[0].symbol == "TEST"
        assert buy_records[0].fill_price is not None
        assert buy_records[0].pre_trade_passed is True


class TestEngineWithJournalSkipNoSignal:
    def test_engine_with_journal_skip_no_signal(self) -> None:
        journal = DecisionJournal()
        engine = BacktestEngine(
            strategy=SilentStrategy(),
            initial_cash=INITIAL_CASH,
            decision_journal=journal,
        )
        candles = _make_candle_series()
        trades, _snapshots = engine.run(
            symbol="TEST",
            segment_id="us_large_cap",
            candles=candles,
        )

        assert len(trades) == 0
        # All records should be SKIP with reason "no_signal"
        assert len(journal.records) > 0
        for rec in journal.records:
            assert rec.final_action == FinalAction.SKIP
            assert rec.skip_reason == "no_signal"


class TestEngineWithJournalSkipPreTradeFail:
    def test_engine_with_journal_skip_pretrade_fail(self) -> None:
        """With very low cash, pre-trade check should fail on BUY signals."""
        journal = DecisionJournal()
        engine = BacktestEngine(
            strategy=AlwaysBuyStrategy(),
            initial_cash=Decimal(1),  # Very low cash
            decision_journal=journal,
        )
        candles = _make_candle_series()
        trades, _snapshots = engine.run(
            symbol="TEST",
            segment_id="us_large_cap",
            candles=candles,
        )

        # No trades should execute with $1 cash
        assert len(trades) == 0
        # Should have SKIP records — either no_signal, position_value_zero, or pre_trade failures
        skip_records = [r for r in journal.records if r.final_action == FinalAction.SKIP]
        assert len(skip_records) > 0


class TestEngineWithoutJournalUnchanged:
    def test_engine_without_journal_unchanged(self) -> None:
        """Regression: None journal doesn't break anything."""
        engine = BacktestEngine(
            strategy=StubStrategy(),
            initial_cash=INITIAL_CASH,
            decision_journal=None,
        )
        candles = _make_candle_series()
        trades, snapshots = engine.run(
            symbol="TEST",
            segment_id="us_large_cap",
            candles=candles,
        )

        assert len(snapshots) == CANDLE_COUNT
        assert len(trades) >= 1


class TestJournalingCombinerCapturesSignals:
    def test_journaling_combiner_captures_signals(self, tmp_path: Path) -> None:
        from finalayze.backtest.journaling_combiner import JournalingStrategyCombiner

        combiner = JournalingStrategyCombiner(
            strategies=[StubStrategy(), SilentStrategy()],
        )

        candles = _make_candle_series()
        # At TRADE_DAY_BUY, StubStrategy fires BUY; SilentStrategy returns None
        history = candles[: TRADE_DAY_BUY + 1]

        # Write a temporary preset YAML so combiner can load config
        presets_dir = tmp_path / "presets"
        presets_dir.mkdir()
        preset_file = presets_dir / "us_large_cap.yaml"
        preset_file.write_text(
            "strategies:\n"
            "  stub:\n"
            "    enabled: true\n"
            "    weight: 1.0\n"
            "  silent:\n"
            "    enabled: true\n"
            "    weight: 0.5\n"
        )
        combiner._presets_dir = presets_dir

        combiner.generate_signal("TEST", history, "us_large_cap")

        # StubStrategy should have fired a BUY
        assert "stub" in combiner.last_signals
        stub_sig = combiner.last_signals["stub"]
        assert stub_sig is not None
        assert stub_sig.direction == SignalDirection.BUY

        # SilentStrategy should be None
        assert "silent" in combiner.last_signals
        assert combiner.last_signals["silent"] is None

        # Weights should be recorded
        assert combiner.last_weights["stub"] == Decimal("1.0")
        assert combiner.last_weights["silent"] == Decimal("0.5")

        # Net score should be recorded
        assert combiner.last_net_score is not None
