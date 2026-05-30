"""Phase-60 look-ahead test suite for the wired earnings SUE path (INTG-01).

Phase 59 computed ``compute_sue_proxy`` / ``EarningsSurprise`` but left them
with zero callers. Phase 60 wires the SUE proxy into the live decision path by
extending the existing ``event_driven`` strategy with a *self-resolving*
earnings event type (D-02): the strategy resolves the active surprise from its
own registered calendar using ``candles[-1].timestamp``, so neither the engine
nor the combiner signature changes. The earnings path MUST NOT read
``sentiment_score`` (the backtest engine passes ``0.0`` — that path is dead in
backtest).

This module is the point-in-time correctness gate for that new path. Every test
is named ``test_lookahead_earnings_*`` so both ``-k lookahead`` and
``-k earnings`` collect the whole suite. Mirrors ``test_lookahead_phase59.py``:
``from __future__ import annotations``, named constants (ruff PLR2004), no live
data / token required.

Covered look-ahead invariants:
  - In-window event (``announcement_date <= bar_date``, within drift window,
    ``|sue| >= threshold``) FIRES a labelled BUY/SELL signal.
  - A FUTURE event (``announcement_date > bar_date``) is SILENT (the core
    look-ahead guard — a future announcement must never move an earlier bar).
  - An OUT-OF-WINDOW event (announced many bars before D, ``bars_since >
    drift_window``) is SILENT.
  - The emitted Signal carries ``EventType.EARNINGS`` and ``is_proxy`` /
    ``sue_score`` on ``strategy_payload`` (Phase-59 D-01 proxy-label carry).
  - A sub-threshold ``|sue|`` produces NO earnings signal.
  - The ``event_driven`` earnings path is independent of ``sentiment_score``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from finalayze.core.schemas import Candle, EventType, SignalDirection
from finalayze.strategies.event_driven import EventDrivenStrategy
from finalayze.strategies.pead import EarningsSurprise, compute_sue_proxy

# Ensure project root is importable so the run_iteration loader can be exercised.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

with patch("sys.argv", ["run_iteration.py", "--name", "test", "--description", "test"]):
    import scripts.run_iteration as ri  # noqa: E402

# ── Shared constants (ruff PLR2004: no magic numbers) ────────────────────────
_SYMBOL = "LKOH"
_MARKET_ID = "moex"
_SEGMENT = "ru_energy"
_TIMEFRAME = "1d"

_ONE_DAY = timedelta(days=1)
# Bar window anchor: the most recent candle timestamp the strategy resolves to.
_D = datetime(2024, 3, 1, tzinfo=UTC)
# Number of daily candles to build ending at D.
_N_CANDLES = 90
# Drift window the extended event_driven uses for earnings resolution.
_DRIFT_WINDOW_BARS = 60

# SUE values relative to the strategy's gate.
_ABOVE_THRESHOLD_POS = 2.0
_ABOVE_THRESHOLD_NEG = -2.0
_BELOW_THRESHOLD = 0.1

# A surprise announced right at the start of the candle window is in-window.
_ANN_IN_WINDOW = _D - timedelta(days=10)
# A surprise dated AFTER the last bar is a future event (look-ahead guard).
_ANN_FUTURE = _D + timedelta(days=10)
# A surprise announced far before D so bars_since exceeds the drift window.
_ANN_STALE = _D - timedelta(days=85)

# Synthetic eps_ttm series for the seeded-proxy test: a step-up that yields a
# positive, non-zero, labelled SUE when resolved as-of the announcement date.
_EPS_SERIES_DAYS = (730, 365, 0)  # days before announcement
_EPS_SERIES_VALUES = (100.0, 105.0, 140.0)


def _build_candles(n: int, end: datetime) -> list[Candle]:
    """Build ``n`` daily candles ending at ``end`` (oldest first)."""
    candles: list[Candle] = []
    for i in range(n):
        ts = end - _ONE_DAY * (n - 1 - i)
        candles.append(
            Candle(
                symbol=_SYMBOL,
                market_id=_MARKET_ID,
                timeframe=_TIMEFRAME,
                timestamp=ts,
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1000,
            ),
        )
    return candles


def _strategy_with_surprise(
    sue_score: float,
    announcement_date: datetime,
    *,
    is_proxy: bool = True,
) -> EventDrivenStrategy:
    strategy = EventDrivenStrategy()
    strategy.add_earnings_surprise(
        EarningsSurprise(
            symbol=_SYMBOL,
            announcement_date=announcement_date,
            sue_score=sue_score,
            actual_eps=140.0,
            expected_eps=100.0,
            is_proxy=is_proxy,
        ),
    )
    return strategy


def test_lookahead_earnings_in_window_fires_buy() -> None:
    """In-window positive surprise fires a BUY with sentiment_score=0.0."""
    strategy = _strategy_with_surprise(_ABOVE_THRESHOLD_POS, _ANN_IN_WINDOW)
    candles = _build_candles(_N_CANDLES, _D)

    signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=0.0)

    assert signal is not None
    assert signal.direction == SignalDirection.BUY
    assert signal.metadata.event_type == EventType.EARNINGS


def test_lookahead_earnings_in_window_fires_sell() -> None:
    """In-window negative surprise fires a SELL with sentiment_score=0.0."""
    strategy = _strategy_with_surprise(_ABOVE_THRESHOLD_NEG, _ANN_IN_WINDOW)
    candles = _build_candles(_N_CANDLES, _D)

    signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=0.0)

    assert signal is not None
    assert signal.direction == SignalDirection.SELL
    assert signal.metadata.event_type == EventType.EARNINGS


def test_lookahead_earnings_future_silent() -> None:
    """A future announcement (date > bar date) must produce NO signal."""
    strategy = _strategy_with_surprise(_ABOVE_THRESHOLD_POS, _ANN_FUTURE)
    candles = _build_candles(_N_CANDLES, _D)

    signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=0.0)

    assert signal is None


def test_lookahead_earnings_out_of_window_silent() -> None:
    """An event older than the drift window must produce NO signal."""
    strategy = _strategy_with_surprise(_ABOVE_THRESHOLD_POS, _ANN_STALE)
    candles = _build_candles(_N_CANDLES, _D)

    signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=0.0)

    assert signal is None


def test_lookahead_earnings_sub_threshold_silent() -> None:
    """A sub-threshold |sue_score| produces NO earnings signal."""
    strategy = _strategy_with_surprise(_BELOW_THRESHOLD, _ANN_IN_WINDOW)
    candles = _build_candles(_N_CANDLES, _D)

    signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=0.0)

    assert signal is None


def test_lookahead_earnings_payload_carries_proxy_label() -> None:
    """Emitted Signal carries is_proxy==1.0 and sue_score on strategy_payload."""
    strategy = _strategy_with_surprise(_ABOVE_THRESHOLD_POS, _ANN_IN_WINDOW, is_proxy=True)
    candles = _build_candles(_N_CANDLES, _D)

    signal = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=0.0)

    assert signal is not None
    assert signal.strategy_payload["is_proxy"] == 1.0
    assert signal.strategy_payload["sue_score"] == _ABOVE_THRESHOLD_POS
    assert "is_proxy" in signal.reasoning


def test_lookahead_earnings_independent_of_sentiment() -> None:
    """The earnings path fires regardless of sentiment_score (engine passes 0.0)."""
    strategy = _strategy_with_surprise(_ABOVE_THRESHOLD_POS, _ANN_IN_WINDOW)
    candles = _build_candles(_N_CANDLES, _D)

    sig_zero = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=0.0)
    sig_neg = strategy.generate_signal(_SYMBOL, candles, _SEGMENT, sentiment_score=-0.9)

    assert sig_zero is not None
    assert sig_neg is not None
    assert sig_zero.metadata.event_type == EventType.EARNINGS
    assert sig_neg.metadata.event_type == EventType.EARNINGS
    assert sig_zero.direction == sig_neg.direction == SignalDirection.BUY


def test_lookahead_earnings_seeded_proxy_is_labelled() -> None:
    """A seeded eps_ttm series produces a labelled is_proxy SUE > 0."""
    ann = _ANN_IN_WINDOW
    eps_history = [
        (ann - timedelta(days=days), value)
        for days, value in zip(_EPS_SERIES_DAYS, _EPS_SERIES_VALUES, strict=True)
    ]
    surprise = compute_sue_proxy(_SYMBOL, ann, eps_history)

    assert surprise.is_proxy is True
    assert surprise.symbol == _SYMBOL


# ── run_iteration loader (Task 3) ────────────────────────────────────────────
_RU_ENERGY_SYMBOLS = ["LKOH", "ROSN"]


def test_lookahead_earnings_loader_registers_seeded_ru_energy() -> None:
    """_setup_event_driven_earnings seeds >=1 labelled ru_energy surprise."""
    strategy = EventDrivenStrategy()
    count = ri._setup_event_driven_earnings("ru_energy", _RU_ENERGY_SYMBOLS, None, strategy)

    assert count >= 1
    # Every seeded surprise is labelled is_proxy (Phase-59 D-01).
    for surprises in strategy._surprises.values():
        for surprise in surprises:
            assert surprise.is_proxy is True


def test_lookahead_earnings_loader_skips_non_ru_segment() -> None:
    """The loader is ru_-gated: a us_ segment registers nothing."""
    strategy = EventDrivenStrategy()
    count = ri._setup_event_driven_earnings("us_tech", ["AAPL"], None, strategy)

    assert count == 0
    assert strategy._surprises == {}
