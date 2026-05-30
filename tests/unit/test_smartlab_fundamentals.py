"""RED TDD suite for the SmartLab fundamentals parser (BACKFILL-H-01/02/04).

Wave 0: this suite is deliberately RED. It imports production symbols that do
not yet exist (``finalayze.data.fetchers.smartlab_fundamentals`` and
``finalayze.data.fundamental_publication_dates``), so collection fails on
``ModuleNotFoundError`` / ``AttributeError`` until Plan 02 lands the parser.

Every test runs against the saved fixture HTML — no live network (T-63.1-01).
"""

from __future__ import annotations

from datetime import UTC, date, timedelta
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
    _FUNDAMENTAL_DISCLOSURE_LAG_DAYS,
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
