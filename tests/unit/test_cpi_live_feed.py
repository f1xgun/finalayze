"""Unit tests for the LIVE CBR CPI feed (Phase 59 Plan 02, CPI-01).

This suite covers the NEW live-feed seam added on top of the single-source CPI
machinery from PR #250 (``_CPI_DATA`` / ``get_cpi_yoy_fraction`` /
``CPI_PUBLICATION_DATES`` / ``cpi_data_staleness_months`` — all reused UNCHANGED):

  * ``CBRFetcher.fetch_cpi_yoy`` + ``_parse_inflation_html`` — scrape + lxml parse
    of the CBR inflation table, mirroring ``fetch_yield_curve`` / ``_parse_zcyc_html``.
    Malformed/unreachable HTML -> ``None`` (no fabrication; ``_CPI_DATA`` stays the seed).
  * ``refresh_cpi_data`` — overlays ONLY publication-eligible fetched months into the
    existing single ``_CPI_DATA`` source (publication-lag look-ahead safety, T-59-04).
  * staleness trends to 0 after a successful live refresh (D-03 acceptance).

Tests NEVER hit the live CBR network — the HTML response is provided inline / patched.
The live HTML table structure is UNCONFIRMED (RESEARCH A2); see the SUMMARY's
manual-verify note for the one-off live ``fetch_cpi_yoy`` check.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

import finalayze.data.fetchers.cbr as cbr_mod
from finalayze.core.exceptions import DataFetchError
from finalayze.data.fetchers.cbr import (
    CBRFetcher,
    cpi_data_staleness_months,
    get_cpi_yoy_fraction,
    latest_cpi_month,
    refresh_cpi_data,
)

# ── Constants (no magic numbers, ruff PLR2004) ──────────────────────────────

CPI_MAR_2026_PCT = Decimal("5.9")
CPI_FEB_2026_PCT = Decimal("5.1")
CPI_APR_2026_PCT = Decimal("5.5")
CPI_APR_2026_FRACTION = 0.055

# A minimal CBR-inflation-style HTML table fixture: a ``data`` table whose first
# data column is the covered month (YYYY-MM) and second column is the YoY %.
SAMPLE_INFLATION_HTML = """
<html><body>
  <table class="data">
    <tr><th>Month</th><th>Inflation, YoY %</th></tr>
    <tr><td>2026-03</td><td>5,9</td></tr>
    <tr><td>2026-02</td><td>5,1</td></tr>
    <tr><td>2026-01</td><td>6,0</td></tr>
  </table>
</body></html>
"""

# HTML with no ``data`` table -> parser must return None (mirrors _parse_zcyc_html).
NO_DATA_HTML = "<html><body><p>Service temporarily unavailable</p></body></html>"


class TestParseInflationHtml:
    """_parse_inflation_html maps covered-month rows to YoY percentage points."""

    def test_parses_rows_to_pct_points(self) -> None:
        result = CBRFetcher._parse_inflation_html(SAMPLE_INFLATION_HTML)
        assert result is not None
        assert result["2026-03"] == CPI_MAR_2026_PCT
        assert result["2026-02"] == CPI_FEB_2026_PCT

    def test_no_data_table_returns_none(self) -> None:
        assert CBRFetcher._parse_inflation_html(NO_DATA_HTML) is None


class TestFetchCpiYoy:
    """fetch_cpi_yoy wraps the scrape; success -> dict, failure -> None (no fabrication)."""

    def test_fetch_success_returns_parsed_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fetcher = CBRFetcher()
        monkeypatch.setattr(
            fetcher,
            "_request",
            lambda *a, **k: SAMPLE_INFLATION_HTML.encode("utf-8"),
        )
        result = fetcher.fetch_cpi_yoy(date(2026, 5, 30))
        assert result is not None
        # Percentage points, UNCONVERTED (get_cpi_yoy_fraction does the /100 on read).
        assert result["2026-03"] == CPI_MAR_2026_PCT

    def test_fetch_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_a: object, **_k: object) -> bytes:
            raise DataFetchError("CBR unreachable")

        fetcher = CBRFetcher()
        monkeypatch.setattr(fetcher, "_request", _boom)
        assert fetcher.fetch_cpi_yoy(date(2026, 5, 30)) is None


@pytest.fixture
def _restore_cpi_state() -> pytest.FixtureRequest:  # type: ignore[name-defined]
    """Snapshot and restore module-level _CPI_DATA / CPI_PUBLICATION_DATES.

    refresh_cpi_data mutates these in-memory dicts; this fixture prevents state
    leaking across tests.
    """
    cpi_snapshot = dict(cbr_mod._CPI_DATA)
    pub_snapshot = dict(cbr_mod.CPI_PUBLICATION_DATES)
    yield  # type: ignore[misc]
    cbr_mod._CPI_DATA.clear()
    cbr_mod._CPI_DATA.update(cpi_snapshot)
    cbr_mod.CPI_PUBLICATION_DATES.clear()
    cbr_mod.CPI_PUBLICATION_DATES.update(pub_snapshot)


@pytest.mark.usefixtures("_restore_cpi_state")
class TestRefreshCpiData:
    """refresh_cpi_data overlays ONLY publication-eligible months into _CPI_DATA."""

    def test_overlay_advances_the_source(self) -> None:
        # 2026-04 is not seeded; an as_of well past its publication date overlays it.
        overlaid = refresh_cpi_data({"2026-04": CPI_APR_2026_PCT}, as_of=date(2026, 7, 1))
        assert overlaid == 1
        assert latest_cpi_month() == "2026-04"
        assert get_cpi_yoy_fraction(2026, 4) == pytest.approx(CPI_APR_2026_FRACTION)

    def test_publication_lookahead_skips_unpublished_month(self) -> None:
        # 2026-04 effective publication = month-end + 2 months ≈ 2026-06-30, so an
        # as_of of 2026-05-10 must NOT overlay it (look-ahead safety, T-59-04).
        before_latest = latest_cpi_month()
        overlaid = refresh_cpi_data({"2026-04": CPI_APR_2026_PCT}, as_of=date(2026, 5, 10))
        assert overlaid == 0
        assert latest_cpi_month() == before_latest
        assert get_cpi_yoy_fraction(2026, 4) is None

    def test_fetch_miss_leaves_seed_intact(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When fetch_cpi_yoy returns None, the convenience refresh leaves _CPI_DATA
        # untouched — every seeded month stays readable (seeded fallback, no fabrication).
        fetcher = CBRFetcher()
        monkeypatch.setattr(fetcher, "fetch_cpi_yoy", lambda _as_of: None)
        overlaid = cbr_mod.refresh_cpi_from_cbr(fetcher, as_of=date(2026, 7, 1))
        assert overlaid == 0
        assert get_cpi_yoy_fraction(2026, 3) == pytest.approx(0.059)

    def test_staleness_trends_to_zero_after_refresh(self) -> None:
        today = date(2026, 5, 30)
        # Overlay every month up to the publication-eligible latest as of today.
        refresh_cpi_data(
            {"2026-04": CPI_APR_2026_PCT, "2026-05": Decimal("5.0")},
            as_of=today,
        )
        assert cpi_data_staleness_months(today) == 0
