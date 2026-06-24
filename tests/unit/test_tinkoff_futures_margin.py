"""Phase 86: TinkoffFetcher.fetch_futures_margin -- fail-LOUD futures initial margin.

The fully-funded equity reserve is sized off the future's initial margin, so the fetch must fail
LOUD (raise ``DataFetchError`` on a gRPC error, reject a zero / non-finite margin) -- never swallow
an outage into ``[]``/0 the way ``fetch_all_futures`` does, which would silently under-reserve.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from finalayze.core.exceptions import DataFetchError
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import Instrument, InstrumentRegistry

_FAKE_TOKEN = "t.fake"  # noqa: S105 -- test stub, not a real secret
_FUTURE_SYMBOL = "IMOEXF"
_FUTURE_FIGI = "FUTIMOEXF000"


def _make_registry() -> InstrumentRegistry:
    registry = InstrumentRegistry()
    registry.register(
        Instrument(
            symbol=_FUTURE_SYMBOL,
            market_id="moex",
            name="MOEX Index future",
            instrument_type="future",
            figi=_FUTURE_FIGI,
            lot_size=1,
            currency="RUB",
        )
    )
    return registry


def _make_fetcher() -> TinkoffFetcher:
    return TinkoffFetcher(token=_FAKE_TOKEN, registry=_make_registry(), sandbox=True)


def test_returns_initial_margin_as_decimal() -> None:
    """A successful fetch returns initial_margin_on_buy as a Decimal."""
    fetcher = _make_fetcher()
    with patch.object(fetcher, "_run_async", return_value=Decimal(2342)):
        assert fetcher.fetch_futures_margin(_FUTURE_SYMBOL) == Decimal(2342)


def test_raises_on_grpc_error_does_not_return_empty() -> None:
    """A gRPC error RAISES DataFetchError (never swallows it into an empty/zero margin)."""
    fetcher = _make_fetcher()
    with (
        patch.object(fetcher, "_run_async", side_effect=RuntimeError("gRPC UNIMPLEMENTED")),
        pytest.raises(DataFetchError, match="futures margin"),
    ):
        fetcher.fetch_futures_margin(_FUTURE_SYMBOL)


@pytest.mark.parametrize("bad", [Decimal(0), Decimal(-1), Decimal("inf"), Decimal("nan")])
def test_rejects_nonpositive_or_nonfinite_margin(bad: Decimal) -> None:
    """A zero / negative / non-finite margin is rejected at the boundary (a real IM is never 0)."""
    fetcher = _make_fetcher()
    with (
        patch.object(fetcher, "_run_async", return_value=bad),
        pytest.raises(DataFetchError, match="non-positive/non-finite"),
    ):
        fetcher.fetch_futures_margin(_FUTURE_SYMBOL)


def test_async_parses_money_value() -> None:
    """_fetch_futures_margin_async maps initial_margin_on_buy MoneyValue -> Decimal."""
    fetcher = _make_fetcher()
    resp = SimpleNamespace(initial_margin_on_buy=SimpleNamespace(units=2342, nano=500_000_000))
    mock_services = SimpleNamespace(
        instruments=SimpleNamespace(get_futures_margin=AsyncMock(return_value=resp))
    )
    with patch.object(fetcher, "_get_services_async", AsyncMock(return_value=mock_services)):
        result = asyncio.run(fetcher._fetch_futures_margin_async(_FUTURE_FIGI))
    assert result == Decimal("2342.5")  # 2342 + 0.5 (nano)
