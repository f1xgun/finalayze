"""Tests for MacroContextProvider and MacroSnapshot.

Verifies that macro data is provided without look-ahead bias,
RUONIA proxy is computed correctly, and CPI respects publication lag.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.data.fetchers.cbr import MacroContextProvider, MacroSnapshot

# ── Named constants ──────────────────────────────────────────────────────────

# Known CBR meeting: 2024-10-25 -> hike to 21.00%
AFTER_OCT_2024_HIKE = date(2024, 10, 26)
OCT_2024_RATE_AFTER = Decimal("21.00")
RUONIA_PROXY_OFFSET = Decimal("0.50")

# Date before any meeting in our calendar (2022-01-01)
EARLY_DATE = date(2021, 12, 1)

# Known CBR meeting: 2023-07-21 -> hike to 8.50%
AFTER_JUL_2023_HIKE = date(2023, 7, 22)
JUL_2023_RATE_AFTER = Decimal("8.50")

# CPI publication: "2024-09" published on 2024-10-11
CPI_PUB_DATE_SEP2024 = date(2024, 10, 11)
CPI_SEP_2024_VALUE = Decimal("8.6")

# Date before any CPI publication in our calendar
DATE_BEFORE_CPI_COVERAGE = date(2024, 1, 1)

# CPI for 2024-01 published on 2024-02-09
DATE_AFTER_JAN2024_CPI_PUB = date(2024, 2, 10)
CPI_JAN_2024_VALUE = Decimal("7.4")

# Date between meetings: 2024-12-20 hold at 21.00, next is 2025-02-14
MID_MEETING_DATE = date(2025, 1, 15)


# ── Test: MacroSnapshot is a frozen dataclass ────────────────────────────────


class TestMacroSnapshotDataclass:
    """MacroSnapshot fields and defaults."""

    def test_default_all_none(self) -> None:
        snap = MacroSnapshot()
        assert snap.key_rate is None
        assert snap.ruonia_7d_avg is None
        assert snap.cpi_yoy is None
        assert snap.last_cbr_decision is None

    def test_fields_set(self) -> None:
        snap = MacroSnapshot(
            key_rate=Decimal("21.00"),
            ruonia_7d_avg=Decimal("20.50"),
            cpi_yoy=Decimal("8.6"),
            last_cbr_decision="hike",
        )
        assert snap.key_rate == Decimal("21.00")
        assert snap.ruonia_7d_avg == Decimal("20.50")
        assert snap.cpi_yoy == Decimal("8.6")
        assert snap.last_cbr_decision == "hike"


# ── Test: key_rate from most recent CBR decision ────────────────────────────


class TestKeyRate:
    """Key rate reflects most recent CBR decision (no look-ahead)."""

    def test_key_rate_after_oct_2024_hike(self) -> None:
        provider = MacroContextProvider()
        snap = provider.get_snapshot(AFTER_OCT_2024_HIKE)
        assert snap.key_rate == OCT_2024_RATE_AFTER

    def test_key_rate_after_jul_2023_hike(self) -> None:
        provider = MacroContextProvider()
        snap = provider.get_snapshot(AFTER_JUL_2023_HIKE)
        assert snap.key_rate == JUL_2023_RATE_AFTER

    def test_key_rate_none_before_calendar(self) -> None:
        """Before any CBR meeting in calendar, key_rate is None."""
        provider = MacroContextProvider()
        snap = provider.get_snapshot(EARLY_DATE)
        assert snap.key_rate is None

    def test_no_look_ahead(self) -> None:
        """Day before Oct 2024 hike should return previous rate (19.00)."""
        provider = MacroContextProvider()
        snap = provider.get_snapshot(date(2024, 10, 24))
        # Previous meeting was 2024-09-13 -> hike to 19.00%
        assert snap.key_rate == Decimal("19.00")

    def test_key_rate_between_meetings(self) -> None:
        """Between meetings, key_rate uses the last decided meeting."""
        provider = MacroContextProvider()
        snap = provider.get_snapshot(MID_MEETING_DATE)
        # Last meeting: 2024-12-20, hold at 21.00%
        assert snap.key_rate == Decimal("21.00")


# ── Test: RUONIA proxy ───────────────────────────────────────────────────────


class TestRuoniaProxy:
    """RUONIA approximated as key_rate - 50bps."""

    def test_ruonia_proxy_calculation(self) -> None:
        provider = MacroContextProvider()
        snap = provider.get_snapshot(AFTER_OCT_2024_HIKE)
        expected_ruonia = OCT_2024_RATE_AFTER - RUONIA_PROXY_OFFSET
        assert snap.ruonia_7d_avg == expected_ruonia

    def test_ruonia_none_when_key_rate_none(self) -> None:
        provider = MacroContextProvider()
        snap = provider.get_snapshot(EARLY_DATE)
        assert snap.ruonia_7d_avg is None


# ── Test: CPI respects publication lag ───────────────────────────────────────


class TestCPIPublicationLag:
    """CPI uses only data published on or before as_of."""

    def test_cpi_after_publication(self) -> None:
        """After Sep 2024 CPI is published (2024-10-11), it is available."""
        provider = MacroContextProvider()
        snap = provider.get_snapshot(CPI_PUB_DATE_SEP2024)
        assert snap.cpi_yoy == CPI_SEP_2024_VALUE

    def test_cpi_before_publication(self) -> None:
        """Day before Sep 2024 CPI publication should get Aug 2024 or earlier."""
        provider = MacroContextProvider()
        snap = provider.get_snapshot(date(2024, 10, 10))
        # Aug 2024 CPI published on 2024-09-13
        assert snap.cpi_yoy == Decimal("9.1")  # Aug 2024 CPI

    def test_cpi_none_before_coverage(self) -> None:
        """Before CPI publication dates start (2024-02-09), cpi_yoy is None."""
        provider = MacroContextProvider()
        snap = provider.get_snapshot(date(2024, 2, 8))
        assert snap.cpi_yoy is None

    def test_cpi_after_jan2024_pub(self) -> None:
        provider = MacroContextProvider()
        snap = provider.get_snapshot(DATE_AFTER_JAN2024_CPI_PUB)
        assert snap.cpi_yoy == CPI_JAN_2024_VALUE


# ── Test: last_cbr_decision ──────────────────────────────────────────────────


class TestLastCBRDecision:
    """Last CBR decision field mirrors the meeting calendar."""

    def test_decision_after_hike(self) -> None:
        provider = MacroContextProvider()
        snap = provider.get_snapshot(AFTER_OCT_2024_HIKE)
        assert snap.last_cbr_decision == "hike"

    def test_decision_after_hold(self) -> None:
        provider = MacroContextProvider()
        # 2024-12-20 was a "hold" meeting
        snap = provider.get_snapshot(date(2024, 12, 21))
        assert snap.last_cbr_decision == "hold"

    def test_decision_after_cut(self) -> None:
        provider = MacroContextProvider()
        # 2022-06-10 was a "cut" to 9.50%
        snap = provider.get_snapshot(date(2022, 6, 11))
        assert snap.last_cbr_decision == "cut"

    def test_decision_none_early(self) -> None:
        provider = MacroContextProvider()
        snap = provider.get_snapshot(EARLY_DATE)
        assert snap.last_cbr_decision is None


# ── Test: Complete snapshot consistency ───────────────────────────────────────


class TestSnapshotConsistency:
    """Full snapshot has all fields populated for dates with data."""

    def test_mid_2024_full_snapshot(self) -> None:
        """Mid-2024 should have all four fields."""
        provider = MacroContextProvider()
        snap = provider.get_snapshot(date(2024, 8, 15))
        assert snap.key_rate is not None
        assert snap.ruonia_7d_avg is not None
        assert snap.cpi_yoy is not None
        assert snap.last_cbr_decision is not None

    def test_snapshot_is_frozen(self) -> None:
        """MacroSnapshot is frozen (immutable)."""
        snap = MacroSnapshot(key_rate=Decimal("21.00"))
        try:
            snap.key_rate = Decimal("22.00")  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")  # noqa: TRY301
        except AttributeError:
            pass  # Expected: frozen dataclass
