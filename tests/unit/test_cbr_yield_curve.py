"""Unit tests for CBR yield curve fetching and MacroSnapshot extensions."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.data.fetchers.cbr import (
    CBRFetcher,
    MacroContextProvider,
    MacroSnapshot,
)

# --- MacroSnapshot field tests ---


def test_macro_snapshot_has_yield_curve_field() -> None:
    snap = MacroSnapshot(yield_curve={"0.25": Decimal("12.85"), "30.0": Decimal("13.60")})
    assert snap.yield_curve is not None
    assert snap.yield_curve["0.25"] == Decimal("12.85")
    assert snap.yield_curve["30.0"] == Decimal("13.60")


def test_macro_snapshot_has_breakeven_inflation_field() -> None:
    snap = MacroSnapshot(breakeven_inflation=Decimal("5.50"))
    assert snap.breakeven_inflation == Decimal("5.50")


def test_macro_snapshot_has_usdrub_field() -> None:
    snap = MacroSnapshot(usdrub=Decimal("92.4500"))
    assert snap.usdrub == Decimal("92.4500")


def test_macro_snapshot_new_fields_default_none() -> None:
    snap = MacroSnapshot()
    assert snap.yield_curve is None
    assert snap.breakeven_inflation is None
    assert snap.usdrub is None


# --- CBR yield curve HTML parsing tests ---

_SAMPLE_ZCYC_HTML = """
<html>
<body>
<table class="data">
<tr>
<th>Date</th><th>0.25</th><th>0.50</th><th>0.75</th><th>1</th>
<th>2</th><th>3</th><th>5</th><th>7</th><th>10</th><th>15</th><th>20</th><th>30</th>
</tr>
<tr>
<td>10.03.2026</td>
<td>16.85</td><td>16.70</td><td>16.55</td><td>16.40</td>
<td>15.80</td><td>15.30</td><td>14.50</td><td>14.00</td>
<td>13.70</td><td>13.50</td><td>13.40</td><td>13.30</td>
</tr>
</table>
</body>
</html>
"""

_EMPTY_ZCYC_HTML = """
<html>
<body>
<table class="data">
<tr>
<th>Date</th><th>0.25</th><th>0.50</th><th>0.75</th><th>1</th>
<th>2</th><th>3</th><th>5</th><th>7</th><th>10</th><th>15</th><th>20</th><th>30</th>
</tr>
</table>
</body>
</html>
"""

EXPECTED_MATURITIES = 12


def test_parse_zcyc_html_produces_correct_dict() -> None:
    fetcher = CBRFetcher()
    result = fetcher._parse_zcyc_html(_SAMPLE_ZCYC_HTML)
    assert result is not None
    assert len(result) == EXPECTED_MATURITIES
    assert result["0.25"] == Decimal("16.85")
    assert result["30"] == Decimal("13.30")
    assert result["5"] == Decimal("14.50")
    fetcher.close()


def test_parse_zcyc_html_empty_table_returns_none() -> None:
    fetcher = CBRFetcher()
    result = fetcher._parse_zcyc_html(_EMPTY_ZCYC_HTML)
    assert result is None
    fetcher.close()


def test_fetch_yield_curve_returns_dict_on_success() -> None:
    fetcher = CBRFetcher()
    with patch.object(fetcher, "_request", return_value=_SAMPLE_ZCYC_HTML.encode("utf-8")):
        result = fetcher.fetch_yield_curve(date(2026, 3, 10))
    assert result is not None
    assert len(result) == EXPECTED_MATURITIES
    fetcher.close()


def test_fetch_yield_curve_returns_none_on_empty() -> None:
    fetcher = CBRFetcher()
    with patch.object(fetcher, "_request", return_value=_EMPTY_ZCYC_HTML.encode("utf-8")):
        result = fetcher.fetch_yield_curve(date(2026, 3, 14))
    assert result is None
    fetcher.close()


def test_fetch_yield_curve_returns_none_on_http_error() -> None:
    from finalayze.core.exceptions import DataFetchError

    fetcher = CBRFetcher()
    with patch.object(fetcher, "_request", side_effect=DataFetchError("timeout")):
        result = fetcher.fetch_yield_curve(date(2026, 3, 10))
    assert result is None
    fetcher.close()


# --- MacroContextProvider integration ---


def test_macro_context_provider_yield_curve_none_in_backtest() -> None:
    """MacroContextProvider (backtest mode) does NOT fetch yield curve."""
    provider = MacroContextProvider()
    snap = provider.get_snapshot(date(2025, 6, 1))
    # Backtest provider should not have yield curve (no HTTP calls)
    assert snap.yield_curve is None


# --- MacroSnapshotModel ORM tests ---


def test_macro_snapshot_model_instantiation() -> None:
    from datetime import UTC, datetime

    from finalayze.core.models import MacroSnapshotModel

    model = MacroSnapshotModel(
        timestamp=datetime(2026, 3, 10, tzinfo=UTC),
        key_rate=Decimal("16.00"),
        ruonia_7d_avg=Decimal("15.50"),
        cpi_yoy=Decimal("10.00"),
        last_cbr_decision="hold",
        breakeven_inflation=Decimal("5.50"),
        yield_curve={"0.25": 16.85, "30": 13.30},
        usdrub=Decimal("92.4500"),
    )
    assert model.timestamp.year == 2026
    assert model.key_rate == Decimal("16.00")
    assert model.yield_curve is not None
    assert model.usdrub == Decimal("92.4500")
    assert model.breakeven_inflation == Decimal("5.50")
