"""Decision journal — structured records at every backtest decision point.

Captures per-strategy signals, risk check results, and final actions
so that evaluation agents can analyze decision quality post-hoc.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime  # noqa: TC003 — used in Pydantic model fields
from decimal import Decimal
from enum import StrEnum
from pathlib import Path  # noqa: TC003 — used at runtime in __init__
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict


class FinalAction(StrEnum):
    """Outcome of a decision point."""

    BUY = "BUY"
    SELL = "SELL"
    SKIP = "SKIP"


class CandleSnapshot(BaseModel):
    """Lightweight candle representation for journal context."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class StrategySignalRecord(BaseModel):
    """Record of a single strategy's contribution to the combined signal."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    direction: str | None  # "BUY" / "SELL" / None if strategy didn't fire
    confidence: float | None
    weight: Decimal
    contribution: Decimal  # direction_score * confidence * weight


class DecisionRecord(BaseModel):
    """Full record of a single decision point in a backtest."""

    model_config = ConfigDict(frozen=True)

    record_id: UUID
    timestamp: datetime
    symbol: str
    segment_id: str

    # Strategy signals
    strategy_signals: list[StrategySignalRecord] = []
    combined_direction: str | None = None
    combined_confidence: float | None = None
    net_weighted_score: float | None = None

    # Context
    sentiment_score: float = 0.0

    # Risk
    pre_trade_passed: bool | None = None
    pre_trade_violations: list[str] = []
    position_value: Decimal | None = None
    quantity: Decimal | None = None
    fill_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    circuit_breaker_level: str = "normal"
    portfolio_equity: Decimal = Decimal(0)
    portfolio_cash: Decimal = Decimal(0)
    open_position_count: int = 0

    # Attribution
    dominant_strategy: str | None = None

    # Outcome
    final_action: FinalAction = FinalAction.SKIP
    skip_reason: str | None = None

    # Last N candles OHLCV
    recent_candles: list[CandleSnapshot] = []

    # Decision logging enrichment (PR-2)
    strategy_features: dict[str, float] | None = None
    model_probas: dict[str, float] | None = None


class DecisionJournal:
    """Collects DecisionRecord entries and optionally flushes to JSONL."""

    def __init__(self, output_path: Path | None = None) -> None:
        self._output_path = output_path
        self._records: list[DecisionRecord] = []

    def record(self, entry: DecisionRecord) -> None:
        """Append a decision record."""
        self._records.append(entry)

    @property
    def records(self) -> list[DecisionRecord]:
        """Return all recorded entries."""
        return list(self._records)

    def flush(self) -> None:
        """Write all records to JSONL file at output_path."""
        if self._output_path is None:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("w") as f:
            for rec in self._records:
                f.write(rec.model_dump_json() + "\n")

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics: counts by action, top skip reasons."""
        action_counts = Counter(r.final_action.value for r in self._records)
        skip_reasons = Counter(r.skip_reason for r in self._records if r.skip_reason is not None)
        return {
            "total_decisions": len(self._records),
            "action_counts": dict(action_counts),
            "top_skip_reasons": dict(skip_reasons.most_common(10)),
        }

    @staticmethod
    def make_record(
        *,
        timestamp: datetime,
        symbol: str,
        segment_id: str,
        final_action: FinalAction,
        skip_reason: str | None = None,
        **kwargs: Any,
    ) -> DecisionRecord:
        """Convenience factory for DecisionRecord with auto-generated UUID."""
        return DecisionRecord(
            record_id=uuid4(),
            timestamp=timestamp,
            symbol=symbol,
            segment_id=segment_id,
            final_action=final_action,
            skip_reason=skip_reason,
            **kwargs,
        )
