"""Consolidated phase-59 look-ahead test suite (DATA-INTEG-01).

This is the single, discoverable (``pytest -k lookahead``) point-in-time
correctness gate across ALL three new phase-59 data feeds:

  (a) FUNDAMENTALS (Plan 04, ``compute_fundamental_features``): a
      ``FundamentalSnapshot`` dated ``as_of > D`` MUST NOT change the feature
      vector computed at bar date D; the same datum dated ``as_of <= D`` MUST
      change it (spike-injection over ``as_of`` — mirrors
      ``test_features_moex.py::TestLagEnforcement``).
  (b) CPI (Plan 02, ``refresh_cpi_data`` + ``get_cpi_yoy_fraction``): a month
      whose Rosstat publication date is AFTER ``as_of`` MUST NOT be overlaid
      into the single ``_CPI_DATA`` source (stays unreadable / ``None``); once
      ``as_of`` reaches the publication date it overlays and becomes readable.
  (c) EARNINGS / SUE proxy (Plan 05, ``compute_sue_proxy``): an ``eps_ttm``
      point dated AFTER the announcement date D MUST NOT contribute to a SUE
      computed as-of D (history truncation at D).
  (d) EARN-01 calendar window (Plan 03 ``fetch_reports`` -> Plan 05 proxy ->
      ``PEADStrategy``): with ``fetch_reports`` gRPC STUBBED to return dated
      ``ReportEvent``s (some inside, some outside a PEAD drift window), filter
      them to the drift window, build proxy SUEs, and assert that only the
      in-window event surfaces and that pead fires for an in-window event while
      an out-of-window (future) event does not — proving the
      ``fetch_reports -> drift-window -> pead`` BACKTEST path end-to-end without
      a live token. (D-04: backtest path only; live-loop wiring stays deferred.)

Every test is named ``test_lookahead_*`` so ``-k lookahead`` collects the whole
suite. Fixtures/builders are reused from the Plan 02/04/05 test modules where it
reads cleanly rather than duplicating data. No live data / token is required.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import finalayze.data.fetchers.cbr as cbr_mod
from finalayze.core.schemas import (
    FundamentalSnapshot,
    MoexMarketData,
    ReportEvent,
    SignalDirection,
)
from finalayze.data.fetchers.cbr import (
    get_cpi_yoy_fraction,
    latest_cpi_month,
    refresh_cpi_data,
)
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry
from finalayze.ml.features.fundamental import compute_fundamental_features
from finalayze.strategies.pead import (
    EarningsSurprise,
    PEADStrategy,
    compute_sue_proxy,
)

# ── Shared constants (ruff PLR2004: no magic numbers) ───────────────────────
_SYMBOL = "SBER"
_SBER_FIGI = "BBG004730N88"
_FAKE_TOKEN = "fake_token"  # noqa: S105
_MARKET_ID_MOEX = "MOEX"
_TIMEFRAME = "1d"

_D = datetime(2025, 3, 1, tzinfo=UTC)
_THIRTY_DAYS = timedelta(days=30)
_ONE_DAY = timedelta(days=1)
_FOUR_HUNDRED_DAYS = timedelta(days=400)
_YEAR = timedelta(days=365)

# (a) fundamentals spike values
_PE_BASE = 5.0
_PE_FUTURE_SPIKE = 999.0
_PE_PAST_SPIKE = 50.0
_REV_BASE = 2e12
_REV_SPIKE = 9e12

# (b) CPI: 2026-04 is unseeded; effective publication = month-end + 2mo lag
#     (~2026-06-30). An as_of of 2026-05-10 is BEFORE publication; 2026-07-01 is after.
_CPI_TEST_MONTH_KEY = "2026-04"
_CPI_TEST_YEAR = 2026
_CPI_TEST_MONTH = 4
_CPI_PCT = Decimal("5.5")
_CPI_FRACTION = 0.055
_AS_OF_BEFORE_PUB = date(2026, 5, 10)
_AS_OF_AFTER_PUB = date(2026, 7, 1)

# (c)/(d) earnings SUE proxy hand-calc series -> SUE == 2.0
_ANN_DATE = datetime(2026, 4, 25, tzinfo=UTC)
_POSITIVE_THRESHOLD = 1.0
_NEGATIVE_THRESHOLD = -1.0
_DRIFT_WINDOW_BARS = 60
_MIN_CONFIDENCE = 0.35
_FUTURE_OFFSET_DAYS = 90


def _snapshot(
    symbol: str,
    as_of: datetime,
    *,
    pe_ratio: float | None = None,
    revenue_ttm: float | None = None,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        as_of=as_of,
        pe_ratio=pe_ratio,
        revenue_ttm=revenue_ttm,
    )


def _eps_history_two_per_year() -> list[tuple[datetime, float]]:
    """eps_ttm series engineered so the SUE hand-calc resolves to 2.0.

    Reuses the Plan-05 construction: surprises {+5, +15, +10}; sample std 5.0;
    SUE = (130 - 120) / 5.0 == 2.0.
    """
    d = _ANN_DATE
    return [
        (d - 3 * _YEAR, 100.0),
        (d - 2 * _YEAR, 105.0),
        (d - 1 * _YEAR, 120.0),
        (d, 130.0),
    ]


# ===========================================================================
# (a) FUNDAMENTALS: as_of <= D look-ahead guard (spike-injection over as_of)
# ===========================================================================
class TestLookaheadFundamentals:
    def test_lookahead_fundamentals_future_snapshot_ignored(self) -> None:
        """A snapshot dated AFTER D must not change the feature vector at D."""
        base = _snapshot(_SYMBOL, _D - _THIRTY_DAYS, pe_ratio=_PE_BASE, revenue_ttm=_REV_BASE)
        spike_future = _snapshot(
            _SYMBOL,
            _D + _THIRTY_DAYS,
            pe_ratio=_PE_FUTURE_SPIKE,
            revenue_ttm=_REV_SPIKE,
        )
        clean = compute_fundamental_features(MoexMarketData(fundamentals=(base,)), as_of=_D)
        with_future = compute_fundamental_features(
            MoexMarketData(fundamentals=(base, spike_future)), as_of=_D
        )
        assert with_future == clean

    def test_lookahead_fundamentals_in_window_snapshot_applied(self) -> None:
        """The SAME spike dated <= D MUST change the output (proves a real filter)."""
        base = _snapshot(_SYMBOL, _D - _FOUR_HUNDRED_DAYS, pe_ratio=_PE_BASE, revenue_ttm=_REV_BASE)
        spike_past = _snapshot(
            _SYMBOL, _D - _ONE_DAY, pe_ratio=_PE_PAST_SPIKE, revenue_ttm=_REV_SPIKE
        )
        clean = compute_fundamental_features(MoexMarketData(fundamentals=(base,)), as_of=_D)
        with_past = compute_fundamental_features(
            MoexMarketData(fundamentals=(base, spike_past)), as_of=_D
        )
        assert with_past != clean


# ===========================================================================
# (b) CPI: publication-lag look-ahead (unpublished month not overlaid)
# ===========================================================================
@pytest.fixture
def _restore_cpi_state() -> object:
    """Snapshot/restore module-level _CPI_DATA / CPI_PUBLICATION_DATES.

    ``refresh_cpi_data`` mutates these in-memory dicts; this keeps state from
    leaking across tests (mirrors test_cpi_live_feed.py).
    """
    cpi_snapshot = dict(cbr_mod._CPI_DATA)
    pub_snapshot = dict(cbr_mod.CPI_PUBLICATION_DATES)
    yield
    cbr_mod._CPI_DATA.clear()
    cbr_mod._CPI_DATA.update(cpi_snapshot)
    cbr_mod.CPI_PUBLICATION_DATES.clear()
    cbr_mod.CPI_PUBLICATION_DATES.update(pub_snapshot)


@pytest.mark.usefixtures("_restore_cpi_state")
class TestLookaheadCpi:
    def test_lookahead_cpi_unpublished_month_not_readable(self) -> None:
        """An as_of BEFORE the month's publication date must not overlay it."""
        before_latest = latest_cpi_month()
        overlaid = refresh_cpi_data({_CPI_TEST_MONTH_KEY: _CPI_PCT}, as_of=_AS_OF_BEFORE_PUB)
        assert overlaid == 0
        assert latest_cpi_month() == before_latest
        assert get_cpi_yoy_fraction(_CPI_TEST_YEAR, _CPI_TEST_MONTH) is None

    def test_lookahead_cpi_published_month_becomes_readable(self) -> None:
        """Once as_of >= publication date, the month overlays and reads back."""
        overlaid = refresh_cpi_data({_CPI_TEST_MONTH_KEY: _CPI_PCT}, as_of=_AS_OF_AFTER_PUB)
        assert overlaid == 1
        assert latest_cpi_month() == _CPI_TEST_MONTH_KEY
        assert get_cpi_yoy_fraction(_CPI_TEST_YEAR, _CPI_TEST_MONTH) == pytest.approx(_CPI_FRACTION)


# ===========================================================================
# (c) EARNINGS / SUE: a future eps_ttm point must not feed the as-of-D SUE
# ===========================================================================
class TestLookaheadSue:
    def test_lookahead_sue_future_eps_point_excluded(self) -> None:
        """An eps_ttm entry dated AFTER D must not change the as-of-D SUE."""
        baseline = _eps_history_two_per_year()
        es_baseline = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=baseline,
        )
        future = (_ANN_DATE + timedelta(days=_FUTURE_OFFSET_DAYS), _PE_FUTURE_SPIKE)
        es_with_future = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=[*baseline, future],
        )
        assert es_with_future.actual_eps == pytest.approx(es_baseline.actual_eps)
        assert es_with_future.expected_eps == pytest.approx(es_baseline.expected_eps)
        assert es_with_future.sue_score == pytest.approx(es_baseline.sue_score)

    def test_lookahead_sue_in_window_eps_point_changes_result(self) -> None:
        """Moving that entry to <= D MUST change the SUE (proves the truncation)."""
        baseline = _eps_history_two_per_year()
        es_baseline = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=baseline,
        )
        in_window = (_ANN_DATE, _PE_FUTURE_SPIKE)
        es_in_window = compute_sue_proxy(
            symbol=_SYMBOL,
            announcement_date=_ANN_DATE,
            eps_history=[*baseline, in_window],
        )
        assert es_in_window.actual_eps != pytest.approx(es_baseline.actual_eps)


# ===========================================================================
# (d) EARN-01 calendar window: fetch_reports -> drift-window -> pead (backtest)
# ===========================================================================
def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_calendar_event(report_date: datetime) -> SimpleNamespace:
    """A GetAssetReportsEvent-like object (period_type is an SDK enum member)."""
    return SimpleNamespace(
        instrument_id=_SBER_FIGI,
        report_date=report_date,
        period_year=datetime(report_date.year, 1, 1, tzinfo=UTC),
        period_num=datetime(report_date.year, report_date.month, 1, tzinfo=UTC),
        period_type=SimpleNamespace(name="PERIOD_TYPE_QUARTER"),
        created_at=report_date,
    )


def _stub_fetch_reports(events: list[object]) -> TinkoffFetcher:
    """A TinkoffFetcher whose gRPC ``get_asset_reports`` is fully stubbed (no token)."""
    fetcher = TinkoffFetcher(token=_FAKE_TOKEN, registry=_make_registry(), sandbox=True)
    services = MagicMock()
    services.instruments.get_asset_reports = AsyncMock(return_value=SimpleNamespace(events=events))
    fetcher._get_services_async = AsyncMock(return_value=services)  # type: ignore[method-assign]
    return fetcher


def _make_candle(ts: datetime) -> object:
    from finalayze.core.schemas import Candle

    price = Decimal("250.0")
    return Candle(
        symbol=_SYMBOL,
        market_id=_MARKET_ID_MOEX,
        timeframe=_TIMEFRAME,
        timestamp=ts,
        open=price,
        high=price + _ONE_DAY.days,
        low=price - _ONE_DAY.days,
        close=price,
        volume=1000,
    )


class TestLookaheadEarn01CalendarWindow:
    """Closes the SPEC EARN-01 acceptance: calendar events resolve into the PEAD
    drift window, and only IN-WINDOW events drive a (backtest) signal — an
    out-of-window (future) report does not. Live-loop wiring stays deferred (D-04).
    """

    def test_lookahead_earn01_calendar_window(self) -> None:
        # Bar date D: the "now" of the backtest loop.
        bar_date = _ANN_DATE + timedelta(days=5)  # 5 bars into the in-window drift

        # In-window report: announced just before D (inside the drift window).
        in_window_date = _ANN_DATE
        # Out-of-window report: dated in the FUTURE relative to D (must not fire).
        out_of_window_date = bar_date + timedelta(days=_FUTURE_OFFSET_DAYS)

        fetcher = _stub_fetch_reports(
            [
                _make_calendar_event(in_window_date),
                _make_calendar_event(out_of_window_date),
            ]
        )

        # 1) fetch_reports returns BOTH dated calendar events (no token used).
        reports = fetcher.fetch_reports(_SYMBOL)
        assert len(reports) == 2  # noqa: PLR2004 — exactly the two stubbed events
        assert all(isinstance(r, ReportEvent) for r in reports)

        # 2) Filter the calendar events to the PEAD drift window as-of the bar date:
        #    a report is in-window iff announcement_date <= D AND it is within
        #    drift_window_bars of D. (out_of_window_date > D -> excluded.)
        in_window = [
            r
            for r in reports
            if r.report_date <= bar_date and (bar_date - r.report_date).days <= _DRIFT_WINDOW_BARS
        ]
        assert len(in_window) == 1
        assert in_window[0].report_date == in_window_date

        # 3) Build proxy SUEs from each report and register only via the unchanged seam.
        strategy = PEADStrategy(
            positive_threshold=_POSITIVE_THRESHOLD,
            negative_threshold=_NEGATIVE_THRESHOLD,
            drift_window_bars=_DRIFT_WINDOW_BARS,
            min_confidence=_MIN_CONFIDENCE,
        )
        for report in reports:
            surprise = compute_sue_proxy(
                symbol=_SYMBOL,
                announcement_date=report.report_date,
                eps_history=[
                    (report.report_date - 3 * _YEAR, 100.0),
                    (report.report_date - 2 * _YEAR, 105.0),
                    (report.report_date - 1 * _YEAR, 120.0),
                    (report.report_date, 130.0),
                ],
            )
            strategy.add_earnings_surprise(surprise)

        # 4) At the bar date, pead fires for the in-window event (SUE 2.0 > 1.0)…
        candles = [_make_candle(_ANN_DATE - _ONE_DAY + timedelta(days=i)) for i in range(7)]
        signal = strategy.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id="ru_blue_chips",
        )
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "pead"

        # 5) …but a strategy carrying ONLY the out-of-window (future) report does NOT.
        future_only = PEADStrategy(
            positive_threshold=_POSITIVE_THRESHOLD,
            negative_threshold=_NEGATIVE_THRESHOLD,
            drift_window_bars=_DRIFT_WINDOW_BARS,
            min_confidence=_MIN_CONFIDENCE,
        )
        future_only.add_earnings_surprise(
            EarningsSurprise(
                symbol=_SYMBOL,
                announcement_date=out_of_window_date,
                sue_score=2.0,
                actual_eps=130.0,
                expected_eps=120.0,
                is_proxy=True,
            )
        )
        future_signal = future_only.generate_signal(
            symbol=_SYMBOL,
            candles=candles,
            segment_id="ru_blue_chips",
        )
        assert future_signal is None
