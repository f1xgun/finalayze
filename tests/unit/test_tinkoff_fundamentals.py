"""Unit tests for TinkoffFetcher.fetch_fundamentals (FUND-01).

gRPC is fully stubbed — no live token. ``_get_services_async`` is replaced with a
mock that returns canned ``share_by`` / ``get_asset_fundamentals`` responses so the
real async worker (resolver + mapper + 0.0->None rule) is exercised end-to-end.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from finalayze.core.schemas import FundamentalSnapshot
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS, InstrumentRegistry

FAKE_TOKEN = "fake_token"  # noqa: S105
SBER_SYMBOL = "SBER"
SBER_ASSET_UID = "uid-sber"

PE = 5.2
EPS = 120.0
REVENUE = 3.1e12
NET_MARGIN = 0.31
ROE = 0.24
EV_EBITDA = 4.1
DIV_YIELD = 0.11
MARKET_CAP = 6.2e12


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    return registry


def _make_fetcher() -> TinkoffFetcher:
    return TinkoffFetcher(token=FAKE_TOKEN, registry=_make_registry(), sandbox=True)


def _make_statistic(**overrides: float | str) -> SimpleNamespace:
    """Build a StatisticResponse-like object with all relevant float fields."""
    base: dict[str, float | str] = {
        "pe_ratio_ttm": PE,
        "eps_ttm": EPS,
        "revenue_ttm": REVENUE,
        "net_margin_mrq": NET_MARGIN,
        "roe": ROE,
        "ev_to_ebitda_mrq": EV_EBITDA,
        "dividend_yield_daily_ttm": DIV_YIELD,
        "market_capitalization": MARKET_CAP,
        "currency": "RUB",
        "fiscal_period_end_date": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _stub_services(
    *,
    asset_uid: str | None,
    fundamentals: list[object],
    share_by_raises: bool = False,
) -> MagicMock:
    """Build a mock AsyncServices with stubbed share_by + get_asset_fundamentals."""
    services = MagicMock()
    if share_by_raises:
        services.instruments.share_by = AsyncMock(side_effect=RuntimeError("not found"))
    else:
        share_resp = SimpleNamespace(instrument=SimpleNamespace(asset_uid=asset_uid))
        services.instruments.share_by = AsyncMock(return_value=share_resp)
    fund_resp = SimpleNamespace(fundamentals=fundamentals)
    services.instruments.get_asset_fundamentals = AsyncMock(return_value=fund_resp)
    return services


def _wire(fetcher: TinkoffFetcher, services: MagicMock) -> None:
    fetcher._get_services_async = AsyncMock(return_value=services)  # type: ignore[method-assign]


class TestFetchFundamentals:
    def test_populated_snapshot(self) -> None:
        """A fully-populated StatisticResponse maps to a FundamentalSnapshot."""
        fetcher = _make_fetcher()
        services = _stub_services(asset_uid=SBER_ASSET_UID, fundamentals=[_make_statistic()])
        _wire(fetcher, services)

        snap = fetcher.fetch_fundamentals(SBER_SYMBOL)

        assert isinstance(snap, FundamentalSnapshot)
        assert snap.symbol == SBER_SYMBOL
        assert snap.pe_ratio == PE
        assert snap.eps_ttm == EPS
        assert snap.currency == "RUB"
        assert snap.as_of.tzinfo is not None  # UTC-aware (schema validator)
        # resolution used asset_uid, not FIGI
        services.instruments.get_asset_fundamentals.assert_awaited_once()
        req = services.instruments.get_asset_fundamentals.await_args.args[0]
        assert req.assets == [SBER_ASSET_UID]

    def test_missing_field_maps_to_none(self) -> None:
        """A 0.0 ratio is treated as unavailable (Pitfall 1), other fields stay set."""
        fetcher = _make_fetcher()
        stat = _make_statistic(pe_ratio_ttm=0.0)
        services = _stub_services(asset_uid=SBER_ASSET_UID, fundamentals=[stat])
        _wire(fetcher, services)

        snap = fetcher.fetch_fundamentals(SBER_SYMBOL)

        assert snap is not None
        assert snap.pe_ratio is None  # 0.0 -> None, no fabrication
        assert snap.eps_ttm == EPS  # other populated fields remain set

    def test_unknown_symbol_returns_none(self) -> None:
        """share_by raising for every class_code -> None (no raise)."""
        fetcher = _make_fetcher()
        services = _stub_services(asset_uid=None, fundamentals=[], share_by_raises=True)
        _wire(fetcher, services)

        assert fetcher.fetch_fundamentals("NOPE") is None

    def test_empty_response_returns_none(self) -> None:
        """resp.fundamentals == [] -> None."""
        fetcher = _make_fetcher()
        services = _stub_services(asset_uid=SBER_ASSET_UID, fundamentals=[])
        _wire(fetcher, services)

        assert fetcher.fetch_fundamentals(SBER_SYMBOL) is None
