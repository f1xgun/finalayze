"""Unit tests for MoexISSFetcher.fetch_currency_close_history (gold/FX currency engine).

Mirrors ``test_moex_iss_fetcher.py``: patch ``_get_json`` with fixture rows. Confirms
the currency/selt CETS endpoint parsing, the holiday CLOSE<=0 skip, and pagination.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from finalayze.data.fetchers.moex_iss import MoexISSFetcher

_COLUMNS = [
    "BOARDID",
    "TRADEDATE",
    "SHORTNAME",
    "SECID",
    "OPEN",
    "LOW",
    "HIGH",
    "CLOSE",
    "NUMTRADES",
]
_START = datetime(2022, 2, 21, tzinfo=UTC)
_END = datetime(2022, 2, 26, tzinfo=UTC)
_EXPECTED_CLOSE_0 = Decimal("4876.5")
_EXPECTED_CLOSE_1 = Decimal("5330.0")
_EXPECTED_N = 2


def _row(d: str, close: float) -> list[object]:
    return ["CETS", d, "GLDRUB_TOM", "GLDRUB_TOM", close, close, close, close, 100]


def _block(rows: list[list[object]]) -> dict[str, object]:
    return {"history": {"columns": _COLUMNS, "data": rows}}


def test_fetch_currency_close_history_parses_cets_and_skips_holidays() -> None:
    fetcher = MoexISSFetcher()
    # One page: a real bar, a CLOSE=0 holiday row (skipped), and another real bar.
    page = _block(
        [
            _row("2022-02-21", 4876.5),
            _row("2022-02-23", 0),  # holiday / no-trade — must be skipped
            _row("2022-02-24", 5330.0),
        ]
    )
    with patch.object(fetcher, "_get_json", return_value=page):
        out = fetcher.fetch_currency_close_history("GLDRUB_TOM", _START, _END)
    fetcher.close()
    assert len(out) == _EXPECTED_N
    assert out[0] == (date(2022, 2, 21), _EXPECTED_CLOSE_0)
    assert out[1] == (date(2022, 2, 24), _EXPECTED_CLOSE_1)


def test_fetch_currency_close_history_paginates() -> None:
    fetcher = MoexISSFetcher()
    # First page full (_PAGE_SIZE rows) triggers a second fetch; empty page stops.
    base = date(2022, 3, 1)
    full_page = _block([_row((base + timedelta(days=i)).isoformat(), 6000 + i) for i in range(100)])
    empty_page = _block([])
    with patch.object(fetcher, "_get_json", side_effect=[full_page, empty_page]):
        out = fetcher.fetch_currency_close_history("GLDRUB_TOM", _START, _END)
    fetcher.close()
    assert len(out) == 100  # noqa: PLR2004 — one full ISS page
