"""Unit tests for CPI data freshness, the single-source CPI accessor, and the
staleness guard.

Regression coverage for the May-2026 finding that two divergent hardcoded CPI
tables (``cbr._CPI_DATA`` and the now-removed ``macro.TRAILING_CPI``) had both
gone stale, silently degrading the ``real_rate_zscore`` MOEX ML feature to 0.0.

After the fix:
  * ``cbr._CPI_DATA`` is the single source of truth and covers through 2026-Q1.
  * ``macro.compute_macro_features`` reads CPI via ``cbr.get_cpi_yoy_fraction``.
  * ``cbr.cpi_data_staleness_months`` lets callers detect/log silent rot.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from finalayze.core.schemas import KeyRateRecord, MoexMarketData
from finalayze.data.fetchers.cbr import (
    _CPI_DATA,
    cpi_data_staleness_months,
    get_cpi_yoy_fraction,
    latest_cpi_month,
)
from finalayze.ml.features.macro import compute_macro_features

# ── Constants (no magic numbers, ruff PLR2004) ──────────────────────────────

REQUIRED_LATEST_CPI_MONTH = "2026-03"  # data must cover at least through 2026-Q1
CPI_MAR_2026_PCT = Decimal("5.9")
CPI_MAR_2026_FRACTION = 0.059
CPI_DEC_2025_FRACTION = 0.056
FRACTION_TOL = 1e-9

# Staleness: publication lag is 2 months, so data ending 2026-03 is "fresh"
# when queried in May 2026 (diff 2 - lag 2 = 0).
FRESH_AS_OF = date(2026, 5, 15)
STALE_AS_OF = date(2027, 6, 1)
MIN_STALE_MONTHS = 12  # 2027-06 vs 2026-03 minus lag → at least a year stale


class TestCpiDataCoverage:
    """The single CPI table must stay current enough to feed the ML feature."""

    def test_covers_through_2026_q1(self) -> None:
        assert latest_cpi_month() >= REQUIRED_LATEST_CPI_MONTH

    def test_march_2026_value(self) -> None:
        assert _CPI_DATA["2026-03"] == CPI_MAR_2026_PCT

    def test_series_is_monotonic_keys(self) -> None:
        keys = list(_CPI_DATA.keys())
        assert keys == sorted(keys), "CPI months must be in chronological order"


class TestGetCpiYoyFraction:
    """Single-source accessor returns CPI as a decimal fraction (10% -> 0.10)."""

    def test_known_month_returns_fraction(self) -> None:
        assert get_cpi_yoy_fraction(2026, 3) == pytest.approx(
            CPI_MAR_2026_FRACTION, abs=FRACTION_TOL
        )

    def test_december_2025(self) -> None:
        assert get_cpi_yoy_fraction(2025, 12) == pytest.approx(
            CPI_DEC_2025_FRACTION, abs=FRACTION_TOL
        )

    def test_missing_month_returns_none(self) -> None:
        assert get_cpi_yoy_fraction(2019, 1) is None


class TestCpiStaleness:
    """Staleness guard makes silent CPI rot observable."""

    def test_fresh_when_recent(self) -> None:
        assert cpi_data_staleness_months(FRESH_AS_OF) == 0

    def test_stale_when_old(self) -> None:
        assert cpi_data_staleness_months(STALE_AS_OF) >= MIN_STALE_MONTHS

    def test_future_data_never_negative(self) -> None:
        # Querying a date before coverage starts must not report negative staleness.
        assert cpi_data_staleness_months(date(2020, 1, 1)) == 0


class TestMacroFeatureUsesRecentCpi:
    """The MOEX real-rate feature must compute against 2026 CPI, not collapse to 0.

    The z-score needs >=20 daily observations, so we span a ~40-day window in
    early 2026 with a mid-window rate change. Pre-fix, CPI lookups for late-2025
    / 2026 returned ``None`` (even via the 6-month fallback), the sparse series
    was empty, and the feature returned exactly 0.0. After extending the table
    the feature produces a genuine, non-zero z-score.
    """

    _WINDOW_DAYS = 40
    _WINDOW_START = datetime(2026, 1, 5, tzinfo=UTC)

    def _market_data(self) -> MoexMarketData:
        # Rate steps up mid-window so the forward-filled real-rate series varies
        # (constant series → std 0 → z 0, which would mask the regression).
        records = (
            KeyRateRecord(timestamp=datetime(2025, 12, 19, tzinfo=UTC), rate=Decimal("0.17")),
            KeyRateRecord(timestamp=datetime(2026, 2, 1, tzinfo=UTC), rate=Decimal("0.21")),
        )
        return MoexMarketData(key_rates=records)

    def test_real_rate_zscore_is_computed(self) -> None:
        moex = self._market_data()
        timestamps = [self._WINDOW_START + timedelta(days=i) for i in range(self._WINDOW_DAYS)]
        result = compute_macro_features(moex, candle_timestamps=timestamps)
        assert result["real_rate_zscore"] != 0.0
