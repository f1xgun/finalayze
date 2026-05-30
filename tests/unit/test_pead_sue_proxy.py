"""Tests for the labelled SUE proxy (EARN-02, Layer 4).

Per D-01 + RESEARCH Q1: ``get_asset_reports`` is calendar-only (no actual EPS),
so the proxy's "actual EPS" comes from an ``eps_ttm`` fundamental time-series
(FUND-01 storage). The resulting ``EarningsSurprise`` is ALWAYS labelled
``is_proxy=True`` so backtest attribution stays honest, and ``compute_sue_proxy``
is point-in-time: a SUE computed as-of D uses only history entries dated <= D
(closes RESEARCH Assumption A4 — ``report_date`` as as-of cutoff).

All inputs are constructed in-test; no live data / token needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.pead import (
    EarningsSurprise,
    PEADStrategy,
    compute_sue_proxy,
)

# ---------------------------------------------------------------------------
# Constants (ruff PLR2004)
# ---------------------------------------------------------------------------
_SYMBOL = "SBER"
_MARKET_ID_MOEX = "MOEX"
_SEGMENT_RU = "ru_blue_chips"
_TIMEFRAME = "1d"

_ANN_DATE = datetime(2026, 4, 25, tzinfo=UTC)

_ACTUAL_EPS = 130.0
_EXPECTED_EPS = 120.0
_STD_SURPRISES = 5.0
_EXPECTED_SUE = (_ACTUAL_EPS - _EXPECTED_EPS) / _STD_SURPRISES  # == 2.0

_ZERO_SUE = 0.0
_DRIFT_WINDOW_BARS = 60
_MIN_CONFIDENCE = 0.35


def _make_candle(ts: datetime) -> Candle:
    from decimal import Decimal

    return Candle(
        symbol=_SYMBOL,
        market_id=_MARKET_ID_MOEX,
        timeframe=_TIMEFRAME,
        timestamp=ts,
        open=Decimal("250.0"),
        high=Decimal("251.0"),
        low=Decimal("249.0"),
        close=Decimal("250.0"),
        volume=1000,
    )


def _eps_history_two_per_year() -> list[tuple[datetime, float]]:
    """eps_ttm time-series engineered so the SUE hand-calc resolves to 2.0.

    Construction (one point per ``year`` so prior-year same-period lookup is
    well-defined):
      - D-3y = 100.0
      - D-2y = 105.0   -> surprise vs prior-year = +5
      - D-1y = 120.0   -> surprise vs prior-year = +15 ; this is ``expected``
      - D    = 130.0   -> surprise vs prior-year = +10 ; this is ``actual``

    historical surprises = {+5, +15, +10}; sample std (ddof=1) == 5.0.
    SUE = (actual - expected) / std = (130 - 120) / 5.0 == 2.0.
    """
    d = _ANN_DATE
    year = timedelta(days=365)
    return [
        (d - 3 * year, 100.0),
        (d - 2 * year, 105.0),  # +5 vs prior
        (d - 1 * year, 120.0),  # +15 vs prior ; expected for as-of D
        (d, 130.0),  # +10 vs prior ; actual at as-of D
    ]


# ===================================================================
# Test 1: is_proxy field + default (backwards-compatible, frozen+slots)
# ===================================================================
class TestIsProxyField:
    def test_default_is_false(self) -> None:
        es = EarningsSurprise(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            sue_score=1.2,
            actual_eps=_ACTUAL_EPS,
            expected_eps=_EXPECTED_EPS,
        )
        assert es.is_proxy is False

    def test_can_set_true(self) -> None:
        es = EarningsSurprise(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            sue_score=1.2,
            actual_eps=_ACTUAL_EPS,
            expected_eps=_EXPECTED_EPS,
            is_proxy=True,
        )
        assert es.is_proxy is True

    def test_still_frozen(self) -> None:
        es = EarningsSurprise(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            sue_score=1.2,
            actual_eps=_ACTUAL_EPS,
            expected_eps=_EXPECTED_EPS,
        )
        with pytest.raises((AttributeError, Exception)):
            es.is_proxy = True  # type: ignore[misc]


# ===================================================================
# Test 2: hand-calc SUE + always labelled
# ===================================================================
class TestHandCalcSue:
    def test_sue_score_and_label(self) -> None:
        es = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=_eps_history_two_per_year(),
        )
        assert es.is_proxy is True
        assert es.actual_eps == pytest.approx(_ACTUAL_EPS)
        assert es.expected_eps == pytest.approx(_EXPECTED_EPS)
        assert es.sue_score == pytest.approx(_EXPECTED_SUE)  # (130-120)/5.0 == 2.0
        assert es.symbol == _SYMBOL
        assert es.announcement_date == _ANN_DATE


# ===================================================================
# Test 3: registers into pead + fires in drift window
# ===================================================================
class TestRegistersAndFires:
    def test_proxy_fires_signal(self) -> None:
        strategy = PEADStrategy(
            positive_threshold=1.0,
            negative_threshold=-1.0,
            drift_window_bars=_DRIFT_WINDOW_BARS,
            min_confidence=_MIN_CONFIDENCE,
        )
        es = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=_eps_history_two_per_year(),
        )
        strategy.add_earnings_surprise(es)

        # Candles spanning a few days before to a few after announcement.
        start = _ANN_DATE - timedelta(days=3)
        candles = [_make_candle(start + timedelta(days=i)) for i in range(6)]

        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id=_SEGMENT_RU,
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY  # SUE 2.0 > 1.0
        assert signal.strategy_name == "pead"


# ===================================================================
# Test 4: degenerate history -> 0.0 fallback (no NaN / raise)
# ===================================================================
class TestDegenerateFallback:
    def test_single_point_history(self) -> None:
        es = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=[(_ANN_DATE, _ACTUAL_EPS)],
        )
        assert es.is_proxy is True
        assert es.sue_score == pytest.approx(_ZERO_SUE)
        # no NaN
        assert es.sue_score == es.sue_score  # noqa: PLR0124

    def test_empty_history(self) -> None:
        es = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=[],
        )
        assert es.is_proxy is True
        assert es.sue_score == pytest.approx(_ZERO_SUE)


# ===================================================================
# Test 5: A4 look-ahead — a future report_date must NOT contribute
# ===================================================================
class TestA4LookAhead:
    def test_future_dated_entry_excluded(self) -> None:
        baseline = _eps_history_two_per_year()
        es_baseline = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=baseline,
        )

        # Append one entry dated strictly AFTER D (a future earnings point).
        future = (_ANN_DATE + timedelta(days=90), 999.0)
        es_with_future = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=[*baseline, future],
        )

        # The future-dated entry was filtered by the as-of<=D cutoff: identical.
        assert es_with_future.actual_eps == pytest.approx(es_baseline.actual_eps)
        assert es_with_future.expected_eps == pytest.approx(es_baseline.expected_eps)
        assert es_with_future.sue_score == pytest.approx(es_baseline.sue_score)

    def test_same_entry_in_window_does_change(self) -> None:
        """Moving that entry to <= D MUST change the result (proves the filter)."""
        baseline = _eps_history_two_per_year()
        es_baseline = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=baseline,
        )

        # Same value, now dated AT D (latest) -> becomes the actual, changes result.
        in_window = (_ANN_DATE, 999.0)
        es_in_window = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=[*baseline, in_window],
        )
        assert es_in_window.actual_eps != pytest.approx(es_baseline.actual_eps)
