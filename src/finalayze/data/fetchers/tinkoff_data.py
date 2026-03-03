"""Tinkoff Invest MOEX data fetcher (Layer 2).

Fetches OHLCV candles from MOEX via the t-tech-investments gRPC SDK.
Wraps async SDK calls in asyncio.run() to provide a sync interface
consistent with BaseFetcher.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

# gRPC env vars MUST be set before importing grpc (via t_tech.invest).
# C-ares DNS resolver may fail; force native (system) resolver.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
# T-Bank uses Russian Trusted Root CA not in standard CA bundles.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_GRPC_ROOTS = _PROJECT_ROOT / "certs" / "grpc_roots.pem"
if _GRPC_ROOTS.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_GRPC_ROOTS))

from datetime import UTC, datetime  # noqa: E402
from decimal import Decimal  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

from t_tech.invest import AsyncClient, CandleInterval  # noqa: E402
from t_tech.invest.sandbox.async_client import AsyncSandboxClient  # noqa: E402

from finalayze.core.exceptions import DataFetchError, InstrumentNotFoundError  # noqa: E402
from finalayze.core.schemas import Candle  # noqa: E402
from finalayze.data.fetchers.base import BaseFetcher  # noqa: E402

if TYPE_CHECKING:
    from finalayze.data.rate_limiter import RateLimiter
    from finalayze.markets.instruments import InstrumentRegistry

_TIMEFRAME_MAP: dict[str, CandleInterval] = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
    "1d": CandleInterval.CANDLE_INTERVAL_DAY,
}

_MOEX_MARKET_ID = "moex"
_TINKOFF_SOURCE = "tinkoff"
_NANO_DIVISOR = Decimal(1_000_000_000)

# T-Bank (formerly Tinkoff) gRPC endpoints — the SDK defaults use the old
# tinkoff.ru domain which no longer resolves; override with tbank.ru.
_TBANK_GRPC_TARGET = "invest-public-api.tbank.ru:443"
_TBANK_GRPC_SANDBOX_TARGET = "sandbox-invest-public-api.tbank.ru:443"


class TinkoffFetcher(BaseFetcher):
    """Fetch MOEX candles from Tinkoff Invest gRPC API.

    Uses sandbox endpoint when sandbox=True (default for development).
    FIGI lookup is handled via InstrumentRegistry -- raises InstrumentNotFoundError
    if the symbol is not registered.
    """

    def __init__(
        self,
        token: str,
        registry: InstrumentRegistry,
        *,
        sandbox: bool = True,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._token = token
        self._registry = registry
        self._sandbox = sandbox
        self._rate_limiter = rate_limiter
    def _make_client(self) -> AsyncClient | AsyncSandboxClient:
        """Create a new async client instance."""
        if self._sandbox:
            return AsyncSandboxClient(
                self._token, target=_TBANK_GRPC_SANDBOX_TARGET
            )
        return AsyncClient(self._token, target=_TBANK_GRPC_TARGET)

    def close(self) -> None:
        """No-op — each fetch creates and closes its own channel."""

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch OHLCV candles for a MOEX symbol."""
        if timeframe not in _TIMEFRAME_MAP:
            supported = ", ".join(sorted(_TIMEFRAME_MAP))
            msg = f"Unsupported timeframe '{timeframe}'. Supported: {supported}"
            raise DataFetchError(msg)

        figi = self._symbol_to_figi(symbol)
        interval = _TIMEFRAME_MAP[timeframe]

        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        try:
            raw_candles = asyncio.run(self._fetch_async(figi, start, end, interval))
        except InstrumentNotFoundError:
            raise
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching {symbol}: {exc}"
            raise DataFetchError(msg) from exc

        return [self._map_candle(c, symbol, timeframe) for c in raw_candles]

    async def _fetch_async(
        self,
        figi: str,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> list[Any]:
        """Async call to Tinkoff SDK get_candles.

        Creates a fresh client per call — the SDK closes the gRPC channel
        on context exit, so we cannot reuse across ``asyncio.run()`` calls.
        """
        client = self._make_client()
        async with client as services:
            response = await services.market_data.get_candles(
                figi=figi,
                from_=start,
                to=end,
                interval=interval,
            )
            return list(response.candles)

    def _symbol_to_figi(self, symbol: str) -> str:
        """Look up FIGI for a MOEX symbol via the instrument registry."""
        instrument = self._registry.get(symbol, _MOEX_MARKET_ID)
        if instrument.figi is None:
            msg = f"Instrument '{symbol}' has no FIGI assigned"
            raise InstrumentNotFoundError(msg)
        return instrument.figi

    def _quotation_to_decimal(self, q: Any) -> Decimal:
        """Convert Tinkoff Quotation(units, nano) to Decimal.

        Quotation.units: integer part
        Quotation.nano: fractional part in billionths (1/1_000_000_000)
        """
        return Decimal(q.units) + Decimal(q.nano) / _NANO_DIVISOR

    def _map_candle(self, raw: Any, symbol: str, timeframe: str = "1d") -> Candle:
        """Map a Tinkoff HistoricCandle to our Candle schema."""
        ts = raw.time
        # SDK returns datetime directly (not protobuf Timestamp)
        timestamp = ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

        return Candle(
            symbol=symbol,
            market_id=_MOEX_MARKET_ID,
            timeframe=timeframe,
            timestamp=timestamp,
            open=self._quotation_to_decimal(raw.open),
            high=self._quotation_to_decimal(raw.high),
            low=self._quotation_to_decimal(raw.low),
            close=self._quotation_to_decimal(raw.close),
            volume=int(raw.volume),
            source=_TINKOFF_SOURCE,
        )
