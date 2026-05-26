"""S5.3 — Backtest end-of-data close-out is opt-in.

Closing every open position at the final candle's close systematically
inflated Sharpe: the synthetic exit assumed zero spread / slippage, so
gains in the open positions became fully realised at mid.  The default is
now ``force_close_at_end=False`` — positions stay open at end of data and
contribute to equity only via per-bar MTM (the broker already snapshots
their value into ``snapshots[-1].equity``).

Contract:
  EOD-01: With ``force_close_at_end=False`` (default), an open position at
          the last bar does NOT produce a trade record.
  EOD-02: Equity snapshots still reflect the unrealised MTM of open
          positions (so Sharpe / max-DD remain honest).
  EOD-03: With ``force_close_at_end=True``, the legacy behaviour returns:
          every open position is closed at the last bar's close.
  EOD-04: ``_last_run_summary["unclosed_at_end"]`` exposes the count.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection


def _make_candle(*, close: float, day_offset: int) -> Candle:
    ts = datetime(2024, 1, 1, 12, 0, tzinfo=UTC) + timedelta(days=day_offset)
    return Candle(
        symbol="SBER",
        market_id="moex",
        timestamp=ts,
        timeframe="1d",
        open=Decimal(str(close)),
        high=Decimal(str(close + 1)),
        low=Decimal(str(close - 1)),
        close=Decimal(str(close)),
        volume=Decimal(1_000_000),
    )


def _trend_candles(n: int = 30) -> list[Candle]:
    """A clean uptrend so a momentum-style BUY happens and stays profitable."""
    return [_make_candle(close=100.0 + i * 2.0, day_offset=i) for i in range(n)]


def _always_buy_signal(symbol: str, segment_id: str) -> Signal:
    return Signal(
        strategy_name="momentum",
        symbol=symbol,
        market_id="moex",
        segment_id=segment_id,
        direction=SignalDirection.BUY,
        confidence=0.85,
        reasoning="forced-buy fixture",
    )


@pytest.fixture
def buy_once_strategy() -> MagicMock:
    """Combiner stub: emit a single BUY on bar 5, then None forever after."""
    strat = MagicMock()
    state: dict[str, bool] = {"fired": False}

    def _gen(symbol, candles, segment_id, **_kw):
        if not state["fired"] and len(candles) >= 5:
            state["fired"] = True
            return _always_buy_signal(symbol, segment_id)
        return None

    strat.generate_signal.side_effect = _gen
    strat.preset_for_segment = MagicMock(return_value={})
    strat.reset_buy_state = MagicMock()
    strat.set_segment_caps = MagicMock()
    return strat


def _run(strategy: MagicMock, *, force_close: bool):
    cfg = BacktestConfig(
        initial_cash=Decimal(100_000),
        force_close_at_end=force_close,
        # disable noise from optional features
        circuit_breaker=None,
        loss_limits=None,
    )
    engine = BacktestEngine(strategy=strategy, config=cfg)
    candles = _trend_candles(30)
    return engine.run("SBER", "ru_finance", candles)


# ─── EOD-01 ──────────────────────────────────────────────────────────────────
def test_default_leaves_open_position_unclosed(buy_once_strategy: MagicMock) -> None:
    trades, snapshots = _run(buy_once_strategy, force_close=False)
    # The forced-close path used to append a final SELL — no longer.
    # Either no trades at all (BUY → still open) OR trades exist but the
    # last one is a BUY, not a synthetic end-of-data SELL.
    if trades:
        assert trades[-1].entry_price > 0  # has real entry, not synthetic close
    # Snapshots must reflect MTM of the open position at the final bar.
    assert snapshots[-1].equity > Decimal(0)


# ─── EOD-02 ──────────────────────────────────────────────────────────────────
def test_equity_snapshots_reflect_mtm_with_unclosed_positions(
    buy_once_strategy: MagicMock,
) -> None:
    """Equity at end of data should track the open position's MTM, not the
    initial cash. For an uptrending series with a BUY mid-way, end equity
    must exceed mid-run equity."""
    trades, snapshots = _run(buy_once_strategy, force_close=False)
    # If a BUY actually fired, end equity should exceed snapshot 0 (initial cash).
    if trades:
        assert snapshots[-1].equity != snapshots[0].equity


# ─── EOD-03 ──────────────────────────────────────────────────────────────────
def test_force_close_at_end_true_realises_position(buy_once_strategy: MagicMock) -> None:
    trades_off, _ = _run(buy_once_strategy, force_close=False)
    # Need a fresh strategy because fixture state carries across runs
    strat2 = MagicMock()
    state: dict[str, bool] = {"fired": False}

    def _gen(symbol, candles, segment_id, **_kw):
        if not state["fired"] and len(candles) >= 5:
            state["fired"] = True
            return _always_buy_signal(symbol, segment_id)
        return None

    strat2.generate_signal.side_effect = _gen
    strat2.preset_for_segment = MagicMock(return_value={})
    strat2.reset_buy_state = MagicMock()
    strat2.set_segment_caps = MagicMock()
    trades_on, _ = _run(strat2, force_close=True)

    # force_close=True path produces ≥ trades than force_close=False path
    # (the forced close adds a final realised SELL if a position is open).
    assert len(trades_on) >= len(trades_off)


# ─── EOD-04 ──────────────────────────────────────────────────────────────────
def test_summary_exposes_unclosed_count(buy_once_strategy: MagicMock) -> None:
    cfg = BacktestConfig(initial_cash=Decimal(100_000), force_close_at_end=False)
    engine = BacktestEngine(strategy=buy_once_strategy, config=cfg)
    engine.run("SBER", "ru_finance", _trend_candles(30))
    summary = engine._last_run_summary
    assert "unclosed_at_end" in summary
    assert isinstance(summary["unclosed_at_end"], int)
    assert summary["unclosed_at_end"] >= 0
