"""Unit tests for TinkoffFetcher.fetch_reports (EARN-01, calendar-only).

gRPC is fully stubbed — no live token. ``_get_services_async`` returns a mock
that yields canned ``get_asset_reports`` events. Reports resolve their
``instrument_id`` via the registry FIGI (``_symbol_to_figi``), NOT the
fundamentals-specific asset_uid.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.core.schemas import ReportEvent
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry

FAKE_TOKEN = "fake_token"  # noqa: S105
SBER_SYMBOL = "SBER"
SBER_FIGI = "BBG004730N88"

REPORT_DATE = datetime(2026, 5, 15, tzinfo=UTC)
PERIOD_YEAR_DT = datetime(2026, 1, 1, tzinfo=UTC)
# Q1 -> a datetime in the first quarter (month 1..3); mapper derives quarter from month.
PERIOD_NUM_DT = datetime(2026, 2, 1, tzinfo=UTC)
EXPECTED_YEAR = 2026
EXPECTED_NUM = 1


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_fetcher() -> TinkoffFetcher:
    return TinkoffFetcher(token=FAKE_TOKEN, registry=_make_registry(), sandbox=True)


def _make_event() -> SimpleNamespace:
    """Build a GetAssetReportsEvent-like object (period_type is an enum member)."""
    period_type = SimpleNamespace(name="PERIOD_TYPE_QUARTER")
    return SimpleNamespace(
        instrument_id=SBER_FIGI,
        report_date=REPORT_DATE,
        period_year=PERIOD_YEAR_DT,
        period_num=PERIOD_NUM_DT,
        period_type=period_type,
        created_at=REPORT_DATE,
    )


def _stub_services(events: list[object]) -> MagicMock:
    services = MagicMock()
    resp = SimpleNamespace(events=events)
    services.instruments.get_asset_reports = AsyncMock(return_value=resp)
    return services


def _wire(fetcher: TinkoffFetcher, services: MagicMock) -> None:
    fetcher._get_services_async = AsyncMock(return_value=services)  # type: ignore[method-assign]


class TestFetchReports:
    def test_maps_calendar_event(self) -> None:
        """A stubbed event maps to a ReportEvent with the SDK enum name stripped."""
        fetcher = _make_fetcher()
        services = _stub_services([_make_event()])
        _wire(fetcher, services)

        reports = fetcher.fetch_reports(SBER_SYMBOL)

        assert len(reports) == 1
        ev = reports[0]
        assert isinstance(ev, ReportEvent)
        assert ev.symbol == SBER_SYMBOL
        assert ev.period_year == EXPECTED_YEAR
        assert ev.period_num == EXPECTED_NUM
        assert ev.period_type == "QUARTER"  # PERIOD_TYPE_ prefix stripped

    def test_report_date_preserved(self) -> None:
        """report_date (publication date, usable as_of) survives the mapping."""
        fetcher = _make_fetcher()
        services = _stub_services([_make_event()])
        _wire(fetcher, services)

        reports = fetcher.fetch_reports(SBER_SYMBOL)

        assert reports[0].report_date == REPORT_DATE

    def test_instrument_id_is_registry_figi(self) -> None:
        """The get_asset_reports request uses the registry FIGI, NOT asset_uid."""
        fetcher = _make_fetcher()
        services = _stub_services([_make_event()])
        _wire(fetcher, services)

        fetcher.fetch_reports(SBER_SYMBOL)

        services.instruments.get_asset_reports.assert_awaited_once()
        req = services.instruments.get_asset_reports.await_args.args[0]
        assert req.instrument_id == SBER_FIGI

    def test_unknown_symbol_returns_empty(self) -> None:
        """_symbol_to_figi raising for an unknown symbol -> [] (no raise)."""
        fetcher = _make_fetcher()
        services = _stub_services([_make_event()])
        _wire(fetcher, services)

        # Force the registry resolver to behave like an unknown symbol.
        def _raise(_symbol: str) -> str:
            raise InstrumentNotFoundError("unknown")

        fetcher._symbol_to_figi = _raise  # type: ignore[method-assign]

        assert fetcher.fetch_reports("NOPE") == []
