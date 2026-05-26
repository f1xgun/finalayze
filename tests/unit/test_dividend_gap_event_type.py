"""S5.2 — DividendGapStrategy must stamp metadata.event_type = DIVIDEND.

The combiner's ``_dedup_event_signals`` collapses same-cycle signals on a
ticker when they share an ``EventType`` in ``_DEDUP_EVENT_TYPES = {CBR,
DIVIDEND}``. DividendGapStrategy historically left ``metadata`` at the
default ``EventType.NONE``, so its signals never participated in dedup —
allowing duplicate entries when another strategy (e.g. event_driven) also
fired on the same ex-div date.

Contract:
  DG-EVT-01: BUY signal on ex-div day carries event_type = DIVIDEND.
  DG-EVT-02: SELL on gap closure carries event_type = DIVIDEND.
  DG-EVT-03: SELL on max-hold also carries event_type = DIVIDEND.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, EventType, SignalDirection
from finalayze.strategies.dividend_gap import DividendEntry, DividendGapStrategy


def _candle(*, close: float, market_id: str = "moex", ts: datetime) -> Candle:
    return Candle(
        symbol="SBER",
        market_id=market_id,
        timestamp=ts,
        timeframe="1d",
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=Decimal(1_000_000),
    )


@pytest.fixture
def strategy() -> DividendGapStrategy:
    s = DividendGapStrategy()
    ex_date = datetime(2024, 5, 10, tzinfo=UTC)
    s.add_dividend(
        "SBER",
        DividendEntry(ex_date=ex_date, amount=15.0, status="paid"),
    )
    return s


def _trigger_buy(strategy: DividendGapStrategy):
    ex_date = datetime(2024, 5, 10, tzinfo=UTC)
    candles = [
        _candle(close=300.0, ts=ex_date - timedelta(days=1)),
        _candle(close=285.0, ts=ex_date),  # ~5% gap
    ]
    return strategy.generate_signal(
        symbol="SBER",
        candles=candles,
        segment_id="ru_finance",
        has_open_position=False,
    )


# ─── DG-EVT-01 ───────────────────────────────────────────────────────────────
def test_buy_signal_carries_event_type_dividend(strategy: DividendGapStrategy) -> None:
    sig = _trigger_buy(strategy)
    assert sig is not None
    assert sig.direction == SignalDirection.BUY
    assert sig.metadata.event_type == EventType.DIVIDEND


# ─── DG-EVT-02 ───────────────────────────────────────────────────────────────
def test_sell_on_gap_closure_carries_event_type(strategy: DividendGapStrategy) -> None:
    # First trigger the BUY so the tracker is in _active_gaps
    _trigger_buy(strategy)

    ex_date = datetime(2024, 5, 10, tzinfo=UTC)
    # Next bar where price recovers above pre-exdiv close
    candles = [
        _candle(close=300.0, ts=ex_date - timedelta(days=1)),
        _candle(close=285.0, ts=ex_date),
        _candle(close=302.0, ts=ex_date + timedelta(days=2)),  # gap closed
    ]
    sell = strategy.generate_signal(
        symbol="SBER",
        candles=candles,
        segment_id="ru_finance",
        has_open_position=True,
    )
    assert sell is not None
    assert sell.direction == SignalDirection.SELL
    assert sell.metadata.event_type == EventType.DIVIDEND


# ─── DG-EVT-03 ───────────────────────────────────────────────────────────────
def test_sell_on_max_hold_carries_event_type(strategy: DividendGapStrategy) -> None:
    _trigger_buy(strategy)
    # Force expiry: walk forward enough bars that bars_since >= max_hold_bars.
    # Empirically max_hold_bars is yield-driven; pad with 60 daily candles
    # (well above any realistic yield-based hold cap).
    ex_date = datetime(2024, 5, 10, tzinfo=UTC)
    candles = [
        _candle(close=300.0, ts=ex_date - timedelta(days=1)),
        _candle(close=285.0, ts=ex_date),
    ]
    # Stay below pre-exdiv close so gap-closure exit doesn't fire first.
    candles.extend(_candle(close=287.0, ts=ex_date + timedelta(days=i)) for i in range(1, 61))

    sell = strategy.generate_signal(
        symbol="SBER",
        candles=candles,
        segment_id="ru_finance",
        has_open_position=True,
    )
    assert sell is not None
    assert sell.direction == SignalDirection.SELL
    assert sell.metadata.event_type == EventType.DIVIDEND
