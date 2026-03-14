"""Unit tests for OFZ-IN indexation coefficient fetching."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pytest

from finalayze.core.exceptions import DataFetchError
from finalayze.data.fetchers.cbr import CBRFetcher, MacroContextProvider, MacroSnapshot

# Sample HTML response from CBR indexation page
_SAMPLE_INDEXATION_HTML = """
<html>
<body>
<table class="data">
<tr>
<th>Date</th><th>Indexation coefficient</th>
</tr>
<tr>
<td>10.03.2026</td><td>1.052300</td>
</tr>
</table>
</body>
</html>
"""

_EMPTY_INDEXATION_HTML = """
<html>
<body>
<table class="data">
<tr>
<th>Date</th><th>Indexation coefficient</th>
</tr>
</table>
</body>
</html>
"""

REASONABLE_COEFF_LOW = Decimal("0.5")
REASONABLE_COEFF_HIGH = Decimal("3.0")


def test_macro_snapshot_has_indexation_field() -> None:
    snap = MacroSnapshot(ofzin_indexation_coefficient=Decimal("1.0523"))
    assert snap.ofzin_indexation_coefficient == Decimal("1.0523")


def test_macro_snapshot_indexation_defaults_none() -> None:
    snap = MacroSnapshot()
    assert snap.ofzin_indexation_coefficient is None


def test_parse_indexation_response_correct_decimal() -> None:
    fetcher = CBRFetcher()
    result = fetcher._parse_indexation_response(_SAMPLE_INDEXATION_HTML)
    assert result is not None
    assert result == Decimal("1.052300")
    fetcher.close()


def test_parse_indexation_response_empty_returns_none() -> None:
    fetcher = CBRFetcher()
    result = fetcher._parse_indexation_response(_EMPTY_INDEXATION_HTML)
    assert result is None
    fetcher.close()


def test_fetch_ofzin_indexation_coefficient_success() -> None:
    fetcher = CBRFetcher()
    with patch.object(
        fetcher, "_request", return_value=_SAMPLE_INDEXATION_HTML.encode("utf-8")
    ):
        result = fetcher.fetch_ofzin_indexation_coefficient(date(2026, 3, 10))
    assert result is not None
    assert result == Decimal("1.052300")
    assert REASONABLE_COEFF_LOW < result < REASONABLE_COEFF_HIGH
    fetcher.close()


def test_fetch_ofzin_indexation_coefficient_missing_data() -> None:
    fetcher = CBRFetcher()
    with patch.object(
        fetcher, "_request", return_value=_EMPTY_INDEXATION_HTML.encode("utf-8")
    ):
        result = fetcher.fetch_ofzin_indexation_coefficient(date(2030, 1, 1))
    assert result is None
    fetcher.close()


def test_fetch_ofzin_indexation_coefficient_http_error() -> None:
    fetcher = CBRFetcher()
    with patch.object(fetcher, "_request", side_effect=DataFetchError("timeout")):
        result = fetcher.fetch_ofzin_indexation_coefficient(date(2026, 3, 10))
    assert result is None
    fetcher.close()


def test_backtest_provider_indexation_none() -> None:
    """MacroContextProvider (backtest mode) does NOT fetch indexation."""
    provider = MacroContextProvider()
    snap = provider.get_snapshot(date(2025, 6, 1))
    assert snap.ofzin_indexation_coefficient is None


def test_macro_snapshot_model_has_indexation_field() -> None:
    from datetime import UTC, datetime

    from finalayze.core.models import MacroSnapshotModel

    model = MacroSnapshotModel(
        timestamp=datetime(2026, 3, 10, tzinfo=UTC),
        key_rate=Decimal("16.00"),
        ofzin_indexation_coefficient=Decimal("1.052300"),
    )
    assert model.ofzin_indexation_coefficient == Decimal("1.052300")
