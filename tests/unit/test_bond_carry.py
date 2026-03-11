"""Unit tests for BondCarryStrategy."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.bond_carry import (
    _MATURITY_ROTATION_MONTHS,
    _REBALANCE_INTERVAL_BARS,
    BondCarryStrategy,
)

# ── Test constants ──────────────────────────────────────────────────────────

_BOND_A = "SU29006RMFS2"
_BOND_B = "SU29014RMFS6"
_BOND_C = "SU29024RMFS5"

_CONFIDENCE_BUY = 0.8
_CONFIDENCE_SELL = 0.9
_DEFAULT_REBALANCE_INTERVAL = 63


def _make_candles(
    symbol: str,
    n: int,
    *,
    start: datetime | None = None,
    price: float = 100.0,
) -> list[Candle]:
    """Create n candles for a bond symbol."""
    base = start or datetime(2025, 1, 10, 10, 0, tzinfo=UTC)
    p = Decimal(str(price))
    return [
        Candle(
            symbol=symbol,
            market_id="moex",
            timeframe="1d",
            timestamp=base + timedelta(days=i),
            open=p,
            high=p + Decimal(1),
            low=p - Decimal(1),
            close=p,
            volume=1000,
        )
        for i in range(n)
    ]


def _far_maturity() -> date:
    """Maturity date well in the future (2 years from now)."""
    return date(2027, 6, 1)


def _near_maturity(candle_date: date) -> date:
    """Maturity date within 6 months of the given date."""
    return candle_date + timedelta(days=90)  # ~3 months


# ── 1. Initial buy signals ────────────────────────────────────────────────


class TestInitialBuySignals:
    """On first bar, generates BUY for all non-held bonds."""

    def test_buy_signal_for_unheld_bond(self) -> None:
        symbols = [_BOND_A, _BOND_B]
        maturity_dates = {
            _BOND_A: _far_maturity(),
            _BOND_B: _far_maturity(),
        }
        strategy = BondCarryStrategy(
            symbols=symbols,
            maturity_dates=maturity_dates,
        )
        candles = _make_candles(_BOND_A, 1)
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={},
            bar_idx=0,
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.symbol == _BOND_A

    def test_buy_signal_for_each_unheld_bond(self) -> None:
        symbols = [_BOND_A, _BOND_B, _BOND_C]
        maturity_dates = {s: _far_maturity() for s in symbols}
        strategy = BondCarryStrategy(
            symbols=symbols,
            maturity_dates=maturity_dates,
        )
        signals = []
        for sym in symbols:
            candles = _make_candles(sym, 1)
            sig = strategy.generate_signal(
                symbol=sym,
                candles=candles,
                open_positions={},
                bar_idx=0,
            )
            signals.append(sig)

        assert all(s is not None for s in signals)
        assert all(s.direction == SignalDirection.BUY for s in signals if s)


# ── 2. No signal when fully invested ──────────────────────────────────────


class TestNoSignalWhenFullyInvested:
    """Returns None when all bonds held and no rebalance is due."""

    def test_returns_none_mid_cycle(self) -> None:
        symbols = [_BOND_A, _BOND_B]
        maturity_dates = {s: _far_maturity() for s in symbols}
        strategy = BondCarryStrategy(
            symbols=symbols,
            maturity_dates=maturity_dates,
        )
        # Simulate: all bonds held, not at rebalance interval
        open_positions = {_BOND_A: {"qty": 10}, _BOND_B: {"qty": 10}}
        candles = _make_candles(_BOND_A, 5)
        # Set last rebalance to current bar so rebalance is not due
        strategy._last_rebalance_bar = 30
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions=open_positions,
            bar_idx=35,
        )
        assert signal is None


# ── 3. Maturity rotation SELL ─────────────────────────────────────────────


class TestMaturityRotation:
    """Generates SELL when bond < 6 months from maturity."""

    def test_sell_signal_near_maturity(self) -> None:
        candles = _make_candles(_BOND_A, 5)
        current_date = candles[-1].timestamp.date()
        near_mat = _near_maturity(current_date)

        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: near_mat},
        )
        open_positions = {_BOND_A: {"qty": 10}}
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions=open_positions,
            bar_idx=10,
        )
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence == _CONFIDENCE_SELL

    def test_sell_includes_months_to_maturity_feature(self) -> None:
        candles = _make_candles(_BOND_A, 5)
        current_date = candles[-1].timestamp.date()
        near_mat = _near_maturity(current_date)

        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: near_mat},
        )
        open_positions = {_BOND_A: {"qty": 10}}
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions=open_positions,
            bar_idx=10,
        )
        assert signal is not None
        assert "months_to_maturity" in signal.features
        assert signal.features["months_to_maturity"] < _MATURITY_ROTATION_MONTHS

    def test_no_sell_if_not_held(self) -> None:
        """Even if near maturity, no SELL if the bond is not in open_positions."""
        candles = _make_candles(_BOND_A, 5)
        current_date = candles[-1].timestamp.date()
        near_mat = _near_maturity(current_date)

        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: near_mat},
        )
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={},
            bar_idx=10,
        )
        # Should not SELL a bond we do not hold
        assert signal is None or signal.direction != SignalDirection.SELL


# ── 4. Rebalance BUY ─────────────────────────────────────────────────────


class TestRebalanceBuy:
    """After rebalance_interval bars, generates BUY for missing positions."""

    def test_buy_at_rebalance(self) -> None:
        symbols = [_BOND_A, _BOND_B]
        maturity_dates = {s: _far_maturity() for s in symbols}
        strategy = BondCarryStrategy(
            symbols=symbols,
            maturity_dates=maturity_dates,
        )
        # Initially trigger rebalance (first bar)
        strategy._last_rebalance_bar = 0

        candles = _make_candles(_BOND_A, 70)
        # BOND_B is held, BOND_A is not
        open_positions = {_BOND_B: {"qty": 10}}
        bar_idx = _DEFAULT_REBALANCE_INTERVAL  # exactly at rebalance boundary

        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions=open_positions,
            bar_idx=bar_idx,
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == _CONFIDENCE_BUY

    def test_no_buy_before_rebalance(self) -> None:
        symbols = [_BOND_A, _BOND_B]
        maturity_dates = {s: _far_maturity() for s in symbols}
        strategy = BondCarryStrategy(
            symbols=symbols,
            maturity_dates=maturity_dates,
        )
        # Set last rebalance to bar 10
        strategy._last_rebalance_bar = 10

        candles = _make_candles(_BOND_A, 40)
        open_positions = {_BOND_B: {"qty": 10}}
        # Not yet at rebalance interval (10 + 63 = 73, we are at 40)
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions=open_positions,
            bar_idx=40,
        )
        assert signal is None


# ── 5. Skip near-maturity on buy ──────────────────────────────────────────


class TestSkipNearMaturityOnBuy:
    """Does not generate BUY for bonds near maturity."""

    def test_no_buy_near_maturity(self) -> None:
        candles = _make_candles(_BOND_A, 5)
        current_date = candles[-1].timestamp.date()
        near_mat = _near_maturity(current_date)

        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: near_mat},
        )
        # Empty positions and first bar -> would normally trigger BUY
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={},
            bar_idx=0,
        )
        # Should NOT buy a bond near maturity
        assert signal is None

    def test_no_buy_near_maturity_at_rebalance(self) -> None:
        candles = _make_candles(_BOND_A, 70)
        current_date = candles[-1].timestamp.date()
        near_mat = _near_maturity(current_date)

        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: near_mat},
        )
        strategy._last_rebalance_bar = 0
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={},
            bar_idx=_DEFAULT_REBALANCE_INTERVAL,
        )
        assert signal is None


# ── 6. Signal has correct fields ──────────────────────────────────────────


class TestSignalFields:
    """Verify strategy_name, instrument_type, market_id, segment_id."""

    def test_buy_signal_fields(self) -> None:
        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: _far_maturity()},
        )
        candles = _make_candles(_BOND_A, 1)
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={},
            bar_idx=0,
        )
        assert signal is not None
        assert signal.strategy_name == "bond_carry"
        assert signal.instrument_type == "bond"
        assert signal.market_id == "moex"
        assert signal.segment_id == "ru_ofz_pk"

    def test_sell_signal_fields(self) -> None:
        candles = _make_candles(_BOND_A, 5)
        current_date = candles[-1].timestamp.date()
        near_mat = _near_maturity(current_date)

        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: near_mat},
        )
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={_BOND_A: {"qty": 10}},
            bar_idx=10,
        )
        assert signal is not None
        assert signal.strategy_name == "bond_carry"
        assert signal.instrument_type == "bond"
        assert signal.market_id == "moex"
        assert signal.segment_id == "ru_ofz_pk"

    def test_confidence_in_valid_range(self) -> None:
        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: _far_maturity()},
        )
        candles = _make_candles(_BOND_A, 1)
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={},
            bar_idx=0,
        )
        assert signal is not None
        min_confidence = 0.0
        max_confidence = 1.0
        assert min_confidence <= signal.confidence <= max_confidence

    def test_empty_candles_returns_none(self) -> None:
        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: _far_maturity()},
        )
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=[],
            open_positions={},
            bar_idx=0,
        )
        assert signal is None

    def test_custom_rebalance_interval(self) -> None:
        custom_interval = 21
        strategy = BondCarryStrategy(
            symbols=[_BOND_A, _BOND_B],
            maturity_dates={_BOND_A: _far_maturity(), _BOND_B: _far_maturity()},
            rebalance_interval=custom_interval,
        )
        strategy._last_rebalance_bar = 0
        candles = _make_candles(_BOND_A, 25)
        open_positions = {_BOND_B: {"qty": 10}}

        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions=open_positions,
            bar_idx=custom_interval,
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY

    def test_reasoning_present(self) -> None:
        strategy = BondCarryStrategy(
            symbols=[_BOND_A],
            maturity_dates={_BOND_A: _far_maturity()},
        )
        candles = _make_candles(_BOND_A, 1)
        signal = strategy.generate_signal(
            symbol=_BOND_A,
            candles=candles,
            open_positions={},
            bar_idx=0,
        )
        assert signal is not None
        assert len(signal.reasoning) > 0
        assert _BOND_A in signal.reasoning
