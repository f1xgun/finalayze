"""Sprint 8: BUY quantity round-up at the 0->1 share boundary.

Audit #16 follow-up. The sizing pipeline produces a position_value (already
>= min_position_size) that, for expensive MOEX shares (e.g. AKRN ~19000 RUB),
is smaller than one whole share, so floor(position_value / price) == 0 and the
BUY was silently dropped with skip_reason="quantity_zero". This left several
MOEX sector segments unable to trade at all (verified end-to-end).

The fix rounds to the *nearest* whole share at the 0->1 boundary: when the sized
allocation is at least half a share AND one share fits inside the hard position
cap and available cash, take the minimum tradeable size of 1 share instead of
skipping. Floor behaviour for quantity >= 1 is unchanged (risk-conservative).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.risk.pre_trade_check import PreTradeResult

_EXPENSIVE_PRICE = Decimal(19000)  # AKRN-like RUB share price
_SEGMENT = "ru_chemicals"


def _make_candle(price: Decimal) -> Candle:
    return Candle(
        symbol="AKRN",
        market_id="moex",
        timeframe="1d",
        timestamp=datetime(2024, 6, 3, 7, 0, tzinfo=UTC),
        open=price,
        high=price * Decimal("1.02"),
        low=price * Decimal("0.98"),
        close=price,
        volume=1_000_000,
    )


def _make_signal() -> Signal:
    return Signal(
        direction=SignalDirection.BUY,
        confidence=0.8,
        strategy_name="momentum",
        symbol="AKRN",
        market_id="moex",
        segment_id=_SEGMENT,
        strategy_payload={},
        reasoning="test",
    )


def _run_buy(
    *,
    position_value: Decimal,
    price: Decimal,
    cash: Decimal,
    equity: Decimal = Decimal(1_000_000),
) -> MagicMock:
    """Run engine._handle_buy with a stubbed sizing pipeline; return the broker mock."""
    config = BacktestConfig(initial_cash=equity)
    engine = BacktestEngine(strategy=MagicMock(), config=config)
    engine._sizing_pipeline = MagicMock()
    engine._sizing_pipeline.compute.return_value = position_value

    broker = MagicMock()
    broker.has_position.return_value = False
    portfolio = MagicMock()
    portfolio.equity = equity
    portfolio.cash = cash
    portfolio.positions = {}
    broker.get_portfolio.return_value = portfolio
    broker.get_positions.return_value = {}

    checker = MagicMock()
    result_mock = MagicMock(spec=PreTradeResult)
    result_mock.passed = True
    result_mock.violations = []
    checker.check.return_value = result_mock

    engine._handle_buy(
        broker=broker,
        checker=checker,
        fill_candle=_make_candle(price),
        symbol="AKRN",
        history=[_make_candle(price) for _ in range(60)],
        entry_prices={},
        segment_id=_SEGMENT,
        signal=_make_signal(),
        entry_bars={},
        bar_index=5,
    )
    return broker


def test_expensive_share_rounds_up_to_one() -> None:
    """position_value >= half a share, fits cap+cash -> buy exactly 1 share.

    Note: handle_buy multiplies position_value by a confidence scale
    (0.5 + confidence*0.5 = 0.9 here), so 12000 -> 10800 post-scale; 10800/19000
    = 0.57 share -> rounds up to 1.
    """
    broker = _run_buy(
        position_value=Decimal(12_000), price=_EXPENSIVE_PRICE, cash=Decimal(1_000_000)
    )

    broker.submit_order.assert_called_once()
    order = broker.submit_order.call_args[0][0]
    assert order.side == "BUY"
    assert order.quantity == Decimal(1)


def test_tiny_allocation_below_half_share_still_skips() -> None:
    """position_value < half a share -> do NOT force a position (no over-allocation)."""
    # 5000 / 19000 = 0.26 shares -> below 0.5 -> skip
    broker = _run_buy(
        position_value=Decimal(5_000), price=_EXPENSIVE_PRICE, cash=Decimal(1_000_000)
    )
    broker.submit_order.assert_not_called()


def test_unaffordable_single_share_skips() -> None:
    """Even a >=0.5-share allocation skips if one share exceeds available cash."""
    # ratio 0.95 (>=0.5) but cash 1000 < 19000 one-share cost -> skip
    broker = _run_buy(position_value=Decimal(18_000), price=_EXPENSIVE_PRICE, cash=Decimal(1_000))
    broker.submit_order.assert_not_called()


def test_single_share_exceeding_position_cap_skips() -> None:
    """One share above the hard position cap (max_position_pct*equity) skips."""
    # equity 50000, cap 20% = 10000 < 19000 one-share cost -> skip
    broker = _run_buy(
        position_value=Decimal(10_000),
        price=_EXPENSIVE_PRICE,
        cash=Decimal(1_000_000),
        equity=Decimal(50_000),
    )
    broker.submit_order.assert_not_called()
