"""RED TDD suite for the SmartLab fundamentals parser (BACKFILL-H-01/02/04).

Wave 0: this suite is deliberately RED. It imports production symbols that do
not yet exist (``finalayze.data.fetchers.smartlab_fundamentals`` and
``finalayze.data.fundamental_publication_dates``), so collection fails on
``ModuleNotFoundError`` / ``AttributeError`` until Plan 02 lands the parser.

Every test runs against the saved fixture HTML — no live network (T-63.1-01).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import FundamentalSnapshot

# Production symbols created in Plan 02 — import is expected to FAIL at Wave 0.
from finalayze.data.fetchers.smartlab_fundamentals import (  # noqa: E402
    SmartlabFundamentalsFetcher,
)
from finalayze.data.fundamental_publication_dates import (  # noqa: E402
    _ANNUAL_DISCLOSURE_LAG_DAYS,
    _FUNDAMENTAL_DISCLOSURE_LAG_DAYS,
    get_effective_annual_disclosure_date,
    get_effective_disclosure_date,
)

_FIXTURES = Path(__file__).parent / "fixtures"

# --- Named constants (ruff PLR2004: no magic numbers in assertions) ----------
_SBER_SYMBOL = "SBER"
_LKOH_SYMBOL = "LKOH"
_EXPECTED_QUARTERS = 4  # 2025Q1..Q4 (LTM column is not a fiscal quarter)
_EXPECTED_PE_Q1 = 4.33
_EXPECTED_ROE_Q1 = 0.214  # "21,4%" -> fraction
_EXPECTED_MARKET_CAP_Q1 = 7_200 * 1_000_000_000  # "7 200" bln RUB -> raw RUB
_EXPECTED_REVENUE_Q1 = 2_100 * 1_000_000_000  # LKOH "2 100" bln RUB -> raw RUB
_EXPECTED_NET_MARGIN_Q1 = 0.08  # LKOH "8,0%" -> fraction
_FLOAT_TOL = 1e-9
_LAG_DAYS = 75
_SBER_Q1_DISCLOSURE = date(2025, 4, 28)  # "28.04.2025"
_LKOH_Q1_QUARTER_END = date(2025, 3, 31)

# --- Annual (/f/y/) named constants (ruff PLR2004) ---------------------------
_ANNUAL_LAG_DAYS = 120
_LKOH_FY2024_DISCLOSURE = date(2025, 3, 28)  # "28.03.2025" for FY2024
_EXPECTED_ANNUAL_YEARS = 4  # 2021..2024 (LTM excluded)
_ANNUAL_FY_YEARS = (2021, 2022, 2023, 2024)
_FY_ENDS = frozenset({date(y, 12, 31) for y in _ANNUAL_FY_YEARS})
_FY2021_EMPTY_YEAR = 2021  # date cell empty -> +120d fallback


@pytest.fixture
def fetcher() -> SmartlabFundamentalsFetcher:
    return SmartlabFundamentalsFetcher()


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _by_period(snapshots: list[FundamentalSnapshot]) -> dict[date, FundamentalSnapshot]:
    return {s.as_of.date(): s for s in snapshots}


class TestSmartlabParse:
    def test_parse_fields(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """Parse SBER fixture by @field -> one FundamentalSnapshot per fiscal quarter."""
        snapshots = fetcher.parse_html(_read("smartlab_sber_msfo_q.html"), _SBER_SYMBOL)

        assert len(snapshots) == _EXPECTED_QUARTERS
        assert all(isinstance(s, FundamentalSnapshot) for s in snapshots)
        assert all(s.symbol == _SBER_SYMBOL for s in snapshots)
        # Quarter columns located by quarter-regex, NOT hard-coded index 2.
        first = sorted(snapshots, key=lambda s: s.as_of)[0]
        assert first.pe_ratio is not None
        assert abs(first.pe_ratio - _EXPECTED_PE_Q1) < _FLOAT_TOL
        assert first.roe is not None
        assert abs(first.roe - _EXPECTED_ROE_Q1) < _FLOAT_TOL
        assert first.eps_ttm is not None

    def test_bank_fields_none(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """SBER is a bank: revenue_ttm and ev_ebitda are None (never fabricated).

        The net-interest row is read via SmartLab's typo'd attr ``net_intertest_margin``
        without raising KeyError (RESEARCH Pitfall 3/4).
        """
        snapshots = fetcher.parse_html(_read("smartlab_sber_msfo_q.html"), _SBER_SYMBOL)

        assert snapshots, "expected at least one snapshot"
        for snap in snapshots:
            assert snap.revenue_ttm is None
            assert snap.ev_ebitda is None

    def test_industrial_fields_populated(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """LKOH is industrial: revenue_ttm / ev_ebitda / net_margin populated."""
        snapshots = fetcher.parse_html(_read("smartlab_lkoh_msfo_q.html"), _LKOH_SYMBOL)
        by_q = _by_period(snapshots)
        # LKOH 2025Q1 has an empty date cell -> resolved via the +75d lag helper.
        q1 = by_q[get_effective_disclosure_date(_LKOH_SYMBOL, "2025Q1")]

        assert q1.revenue_ttm is not None
        assert abs(q1.revenue_ttm - _EXPECTED_REVENUE_Q1) < _FLOAT_TOL
        assert q1.ev_ebitda is not None
        assert q1.net_margin is not None
        assert abs(q1.net_margin - _EXPECTED_NET_MARGIN_Q1) < _FLOAT_TOL


class TestDisclosureDate:
    def test_disclosure_date_and_lag(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """Populated date cell -> as_of == that DD.MM.YYYY UTC.

        Empty date cell (LKOH 2025Q1/Q3) -> as_of == get_effective_disclosure_date
        (quarter-end + 75d), and as_of is NEVER equal to the fiscal-quarter-end.
        """
        sber = fetcher.parse_html(_read("smartlab_sber_msfo_q.html"), _SBER_SYMBOL)
        sber_dates = {s.as_of.date() for s in sber}
        assert _SBER_Q1_DISCLOSURE in sber_dates
        for snap in sber:
            assert snap.as_of.tzinfo is not None
            assert snap.as_of.tzinfo == UTC or snap.as_of.utcoffset() is not None

        lkoh = fetcher.parse_html(_read("smartlab_lkoh_msfo_q.html"), _LKOH_SYMBOL)
        lkoh_dates = {s.as_of.date() for s in lkoh}
        # Empty-cell quarters fall back to +75d lag, never the fiscal-quarter end.
        expected_q1_lag = get_effective_disclosure_date(_LKOH_SYMBOL, "2025Q1")
        assert expected_q1_lag in lkoh_dates
        assert _LKOH_Q1_QUARTER_END not in lkoh_dates

    def test_lag_helper_uses_75_days(self) -> None:
        """The disclosure-lag constant and helper mirror cbr.py (+75 days)."""
        assert _FUNDAMENTAL_DISCLOSURE_LAG_DAYS == _LAG_DAYS
        effective = get_effective_disclosure_date(_LKOH_SYMBOL, "2025Q1")
        assert effective == _LKOH_Q1_QUARTER_END + timedelta(days=_LAG_DAYS)


class TestNumericParsing:
    def test_percent_and_separator_parsing(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """'21,4%' -> 0.214; '7 200' (U+00A0 sep) bln -> 7_200 * 1e9."""
        snapshots = fetcher.parse_html(_read("smartlab_sber_msfo_q.html"), _SBER_SYMBOL)
        first = sorted(snapshots, key=lambda s: s.as_of)[0]

        assert first.roe is not None
        assert abs(first.roe - _EXPECTED_ROE_Q1) < _FLOAT_TOL
        assert first.market_cap is not None
        assert abs(first.market_cap - _EXPECTED_MARKET_CAP_Q1) < _FLOAT_TOL


class TestRobotsGate:
    def test_robots_gate_disallowed_raises(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """robots disallow -> DataFetchError before any pull (BACKFILL-H-04)."""
        with (
            patch("urllib.robotparser.RobotFileParser.read"),
            patch("urllib.robotparser.RobotFileParser.can_fetch", return_value=False),
            pytest.raises(DataFetchError),
        ):
            fetcher.assert_robots_allowed("/q/SBER/f/q/MSFO/")

    def test_robots_gate_allowed_passes(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """robots allow -> no exception."""
        with (
            patch("urllib.robotparser.RobotFileParser.read"),
            patch("urllib.robotparser.RobotFileParser.can_fetch", return_value=True),
        ):
            fetcher.assert_robots_allowed("/q/SBER/f/q/MSFO/")


class TestSmartlabAnnualParse:
    """Annual /f/y/ parse: one snapshot per 4-digit fiscal year, LTM excluded."""

    def test_annual_parse_one_per_year(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """LKOH annual fixture -> one snapshot per year (LTM excluded), fields populated."""
        snaps = fetcher.parse_html_annual(_read("smartlab_lkoh_msfo_y.html"), _LKOH_SYMBOL)

        assert len(snaps) == _EXPECTED_ANNUAL_YEARS
        assert all(isinstance(s, FundamentalSnapshot) for s in snaps)
        assert all(s.symbol == _LKOH_SYMBOL for s in snaps)
        # Every as_of's fiscal year is a real year (LTM column excluded). Disclosure
        # dates land in the year AFTER the FY, so the as_of years are {2022..2025}.
        as_of_years = {s.as_of.year for s in snaps}
        assert as_of_years <= {y + 1 for y in _ANNUAL_FY_YEARS}
        # The full-data FY2024 column must populate the industrial fields.
        fy2024 = next(s for s in snaps if s.as_of.date() == _LKOH_FY2024_DISCLOSURE)
        assert fy2024.pe_ratio is not None
        assert fy2024.revenue_ttm is not None
        assert fy2024.ev_ebitda is not None
        assert fy2024.net_margin is not None
        assert fy2024.roe is not None
        assert fy2024.eps_ttm is not None
        assert fy2024.market_cap is not None

    def test_annual_bank_fields_none(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """SBER bank annual fixture (no revenue/ev_ebitda rows) -> those fields None."""
        snaps = fetcher.parse_html_annual(_read("smartlab_sber_msfo_y.html"), _SBER_SYMBOL)

        assert snaps, "expected at least one annual snapshot"
        assert len(snaps) == _EXPECTED_ANNUAL_YEARS
        for snap in snaps:
            assert snap.revenue_ttm is None
            assert snap.ev_ebitda is None

    def test_annual_year_regex_not_index(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """Year columns are located by header regex, not a fixed index.

        The fixture has a leading "Показатель" label column; index-based parsing
        would pick the wrong column. Recovering exactly {2021..2024} from the
        fallback-lag periods proves the header-scan locator.
        """
        snaps = fetcher.parse_html_annual(_read("smartlab_lkoh_msfo_y.html"), _LKOH_SYMBOL)
        # Reconstruct fiscal years from the FY2021 fallback (as_of = FY-end + 120d)
        # and the real disclosure dates (which fall in FY+1). Year set must be exact.
        recovered_years: set[int] = set()
        for snap in snaps:
            d = snap.as_of.date()
            if d == date(_FY2021_EMPTY_YEAR, 12, 31) + timedelta(days=_ANNUAL_LAG_DAYS):
                recovered_years.add(_FY2021_EMPTY_YEAR)
            else:
                recovered_years.add(d.year - 1)  # real disclosure lands in FY+1
        assert recovered_years == set(_ANNUAL_FY_YEARS)


class TestAnnualDisclosureDate:
    """Annual look-ahead: as_of is real date else FY-end + 120d, never bare FY-end."""

    def test_annual_lookahead_real_date(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """A real "Дата отчёта" cell -> as_of == that date (UTC-aware)."""
        snaps = fetcher.parse_html_annual(_read("smartlab_lkoh_msfo_y.html"), _LKOH_SYMBOL)
        fy2024 = next(s for s in snaps if s.as_of.date() == _LKOH_FY2024_DISCLOSURE)
        assert fy2024.as_of == datetime(2025, 3, 28, tzinfo=UTC)

    def test_annual_lookahead_fallback_120d(self, fetcher: SmartlabFundamentalsFetcher) -> None:
        """An EMPTY date cell (FY2021) -> as_of == FY-end + 120d, never the bare FY-end."""
        snaps = fetcher.parse_html_annual(_read("smartlab_lkoh_msfo_y.html"), _LKOH_SYMBOL)
        expected = date(_FY2021_EMPTY_YEAR, 12, 31) + timedelta(days=_ANNUAL_LAG_DAYS)
        fy2021 = next(s for s in snaps if s.as_of.date() == expected)
        assert fy2021.as_of.date() == expected
        assert fy2021.as_of.date() not in _FY_ENDS
        assert fy2021.as_of.tzinfo is not None

    def test_annual_lag_helper_120_days(self) -> None:
        """The annual disclosure-lag constant and helper use +120 days."""
        assert _ANNUAL_DISCLOSURE_LAG_DAYS == _ANNUAL_LAG_DAYS
        effective = get_effective_annual_disclosure_date(_LKOH_SYMBOL, "2023")
        assert effective == date(2023, 12, 31) + timedelta(days=_ANNUAL_LAG_DAYS)
