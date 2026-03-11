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

from datetime import UTC, date, datetime  # noqa: E402
from decimal import Decimal  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

from t_tech.invest import AsyncClient, CandleInterval  # noqa: E402
from t_tech.invest.sandbox.async_client import AsyncSandboxClient  # noqa: E402

from finalayze.core.exceptions import DataFetchError, InstrumentNotFoundError  # noqa: E402
from finalayze.core.schemas import AccruedInterest, BondInfo, Candle, CouponPayment  # noqa: E402
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
            return AsyncSandboxClient(self._token, target=_TBANK_GRPC_SANDBOX_TARGET)
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

    def _money_to_decimal(self, m: Any) -> Decimal:
        """Convert Tinkoff MoneyValue(units, nano, currency) to Decimal."""
        return Decimal(m.units) + Decimal(m.nano) / _NANO_DIVISOR

    @staticmethod
    def _business_days_before(d: date, n: int) -> date:
        """Go back *n* business days from date *d* (skip weekends)."""
        from datetime import timedelta  # noqa: PLC0415

        current = d
        count = 0
        while count < n:
            current -= timedelta(days=1)
            if current.weekday() < 5:  # noqa: PLR2004  # Monday-Friday
                count += 1
        return current

    def fetch_dividends(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch dividend events for a MOEX symbol.

        Returns a list of dicts with keys: ex_date (datetime), amount (float).
        ``ex_date`` is ``last_buy_date`` from Tinkoff (last day to buy for dividend).
        """
        figi = self._symbol_to_figi(symbol)

        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        try:
            raw_dividends = asyncio.run(self._fetch_dividends_async(figi, start, end))
        except InstrumentNotFoundError:
            raise
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching dividends for {symbol}: {exc}"
            raise DataFetchError(msg) from exc

        return [self._map_dividend(d) for d in raw_dividends]

    async def _fetch_dividends_async(
        self,
        figi: str,
        start: datetime,
        end: datetime,
    ) -> list[Any]:
        """Async call to Tinkoff SDK get_dividends."""
        client = self._make_client()
        async with client as services:
            response = await services.instruments.get_dividends(
                figi=figi,
                from_=start,
                to=end,
            )
            return list(response.dividends)

    # ── Bond data methods ──────────────────────────────────────────────────

    def fetch_bond_info(self, symbol: str) -> BondInfo:
        """Fetch bond metadata from T-Bank API.

        Args:
            symbol: Bond symbol (e.g. SU26244RMFS2). Resolved to FIGI via registry.

        Returns:
            BondInfo with static bond metadata.
        """
        figi = self._symbol_to_figi(symbol)
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            result = asyncio.run(self._fetch_bond_info_async(figi))
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching bond info for {symbol} (FIGI={figi}): {exc}"
            raise DataFetchError(msg) from exc
        return result

    async def _fetch_bond_info_async(self, figi: str) -> BondInfo:
        """Async call to T-Bank SDK bond_by."""
        from t_tech.invest.schemas import InstrumentIdType  # noqa: PLC0415

        client = self._make_client()
        async with client as services:
            resp = await services.instruments.bond_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
                id=figi,
            )
            bond = resp.instrument
            nominal = self._money_to_decimal(bond.nominal)

            # For floating-coupon bonds, coupon_rate stores the spread over RUONIA.
            # Will be populated from coupon data when available.
            coupon_rate_val = Decimal(0)

            maturity = (
                bond.maturity_date.date()
                if hasattr(bond.maturity_date, "date")
                else bond.maturity_date
            )

            return BondInfo(
                figi=bond.figi,
                ticker=bond.ticker,
                isin=bond.isin,
                name=bond.name,
                face_value=nominal,
                coupon_rate=coupon_rate_val,
                coupon_frequency=bond.coupon_quantity_per_year,
                maturity_date=maturity,
                floating_coupon=bond.floating_coupon_flag,
                class_code=bond.class_code,
                currency=bond.currency,
            )

    def fetch_bond_coupons(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[CouponPayment]:
        """Fetch coupon payment schedule for a bond.

        Args:
            symbol: Bond symbol (e.g. SU26244RMFS2). Resolved to FIGI via registry.
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of CouponPayment events within the date range.
        """
        figi = self._symbol_to_figi(symbol)
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            result = asyncio.run(self._fetch_bond_coupons_async(figi, start, end))
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching coupons for {symbol} (FIGI={figi}): {exc}"
            raise DataFetchError(msg) from exc
        return result

    async def _fetch_bond_coupons_async(
        self, figi: str, start: datetime, end: datetime
    ) -> list[CouponPayment]:
        """Async call to T-Bank SDK get_bond_coupons."""
        client = self._make_client()
        async with client as services:
            resp = await services.instruments.get_bond_coupons(
                figi=figi,
                from_=start,
                to=end,
            )
            coupons: list[CouponPayment] = []
            for c in resp.events:
                amount = self._money_to_decimal(c.pay_one_bond)
                coupon_date = (
                    c.coupon_date.date() if hasattr(c.coupon_date, "date") else c.coupon_date
                )
                # Record date is typically T-2 business days before payment.
                # T-Bank doesn't provide it directly; estimate from coupon_date.
                record_date = self._business_days_before(coupon_date, 2)
                coupons.append(
                    CouponPayment(
                        bond_figi=figi,
                        coupon_date=coupon_date,
                        record_date=record_date,
                        amount_per_bond=amount,
                        coupon_number=c.coupon_number,
                    )
                )
            return coupons

    def fetch_accrued_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[AccruedInterest]:
        """Fetch daily accrued interest (NKD) for a bond.

        Args:
            symbol: Bond symbol (e.g. SU26244RMFS2). Resolved to FIGI via registry.
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of daily AccruedInterest records.
        """
        figi = self._symbol_to_figi(symbol)
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            result = asyncio.run(self._fetch_accrued_interest_async(figi, start, end))
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching NKD for {symbol} (FIGI={figi}): {exc}"
            raise DataFetchError(msg) from exc
        return result

    async def _fetch_accrued_interest_async(
        self, figi: str, start: datetime, end: datetime
    ) -> list[AccruedInterest]:
        """Async call to T-Bank SDK get_accrued_interests."""
        client = self._make_client()
        async with client as services:
            resp = await services.instruments.get_accrued_interests(
                figi=figi,
                from_=start,
                to=end,
            )
            results: list[AccruedInterest] = []
            for ai in resp.accrued_interests:
                value = self._money_to_decimal(ai.value)
                value_pct = self._quotation_to_decimal(ai.value_percent)
                ai_date: date = ai.date.date() if hasattr(ai.date, "date") else ai.date
                results.append(
                    AccruedInterest(
                        bond_figi=figi,
                        date=ai_date,
                        value=value,
                        value_percent=value_pct,
                    )
                )
            return results

    @staticmethod
    def _next_business_day(dt: datetime) -> datetime:
        """Advance a datetime to the next business day (skip weekends)."""
        from datetime import timedelta  # noqa: PLC0415

        one_day = timedelta(days=1)
        nxt = dt + one_day
        # Saturday=5, Sunday=6
        while nxt.weekday() >= 5:  # noqa: PLR2004
            nxt += one_day
        return nxt

    def _map_dividend(self, d: Any) -> dict[str, Any]:
        """Map a Tinkoff Dividend to a plain dict.

        Tinkoff returns ``last_buy_date`` — the last day to buy for dividend.
        The actual ex-dividend gap occurs on the **next trading day**, so we
        shift forward by one business day to align with DividendGapStrategy.
        """
        amount = self._quotation_to_decimal(d.dividend_net)
        ex_date = self._next_business_day(d.last_buy_date)
        return {"ex_date": ex_date, "amount": float(amount)}

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
