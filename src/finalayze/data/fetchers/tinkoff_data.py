"""Tinkoff Invest MOEX data fetcher (Layer 2).

Fetches OHLCV candles from MOEX via the t-tech-investments gRPC SDK.
Uses a persistent background event loop with run_coroutine_threadsafe()
to reuse a single gRPC channel across all fetch calls.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import concurrent.futures  # noqa: TC003 - used at runtime for Future type
import os
import threading
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

import structlog  # noqa: E402
from t_tech.invest import AsyncClient, CandleInterval  # noqa: E402

from finalayze.core.exceptions import DataFetchError, InstrumentNotFoundError  # noqa: E402
from finalayze.core.schemas import (  # noqa: E402
    AccruedInterest,
    BondInfo,
    Candle,
    CouponPayment,
    FundamentalSnapshot,
    ReportEvent,
)
from finalayze.data.fetchers.base import BaseFetcher  # noqa: E402

_log = structlog.get_logger()

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
        grpc_timeout: float = 60.0,
        grpc_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._token = token
        self._registry = registry
        self._sandbox = sandbox
        self._rate_limiter = rate_limiter
        self._grpc_timeout = grpc_timeout
        # Externally injected gRPC event loop (from TradingLoop / run_sandbox.py).
        # When set, _run_async uses this loop instead of creating a self-managed one.
        self._grpc_loop = grpc_loop
        # Persistent gRPC channel state (mirrors TinkoffBroker pattern)
        self._client: AsyncClient | None = None
        self._services: object | None = None
        # Self-managed loop fallback (tests, standalone scripts)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_init_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def _make_client(self) -> AsyncClient:
        """Create a new async client instance.

        Uses AsyncClient for both sandbox and production — AsyncSandboxClient
        forcibly overrides target with the old tinkoff.ru domain.
        """
        target = _TBANK_GRPC_SANDBOX_TARGET if self._sandbox else _TBANK_GRPC_TARGET
        return AsyncClient(self._token, target=target)

    def _run_async(self, coro: object) -> object:
        """Run an async coroutine on a persistent background event loop.

        Uses run_coroutine_threadsafe() so gRPC channels survive across
        multiple fetch calls. Mirrors TinkoffBroker._run_async pattern.

        When an external grpc_loop is injected, uses that directly (no
        self-managed loop). Falls back to creating a self-managed loop for
        backward compatibility (tests, standalone scripts).
        """
        loop = self._grpc_loop
        if loop is None:
            # Fallback: self-managed loop for backward compat (tests, standalone use)
            if self._loop is None or self._loop.is_closed():
                with self._loop_init_lock:
                    if self._loop is None or self._loop.is_closed():
                        new_loop = asyncio.new_event_loop()
                        self._loop = new_loop
                        thread = threading.Thread(target=new_loop.run_forever, daemon=True)
                        thread.start()
                        self._loop_thread = thread
            loop = self._loop
        future: concurrent.futures.Future[object] = asyncio.run_coroutine_threadsafe(
            coro,  # type: ignore[arg-type]
            loop,
        )
        return future.result(timeout=self._grpc_timeout)

    async def _get_services_async(self) -> object:
        """Return persistent AsyncServices, creating client lazily.

        Double-checked locking ensures a single gRPC channel is reused
        across all fetch_candles / fetch_dividends / etc. calls.
        """
        if self._services is None:
            async with self._async_lock:
                if self._services is None:
                    self._client = self._make_client()
                    self._services = await self._client.__aenter__()
        return self._services

    def close(self) -> None:
        """Close the persistent gRPC channel and stop the background event loop.

        When grpc_loop is externally injected, only closes the gRPC client/services.
        The loop owner (TradingLoop / run_sandbox.py) manages the loop lifecycle.
        """
        if self._client is not None:
            loop = self._grpc_loop or self._loop
            if loop and not loop.is_closed():
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._client.__aexit__(None, None, None),  # type: ignore[no-untyped-call]
                        loop,
                    )
                    future.result(timeout=5)
                except Exception as exc:
                    _log.warning(
                        "grpc_channel_close_failed",
                        resource="grpc_client",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
        self._client = None
        self._services = None
        # Only stop self-managed loop (not injected grpc_loop)
        if self._grpc_loop is None and self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                _log.debug("event_loop_stop_failed_on_close")
            self._loop = None
            self._loop_thread = None

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
            raw_candles = self._run_async(self._fetch_async(figi, start, end, interval))
        except InstrumentNotFoundError:
            raise
        except TimeoutError as exc:
            _log.error(
                "candle_fetch_timeout", symbol=symbol, figi=figi, timeout_s=self._grpc_timeout
            )
            msg = f"Tinkoff gRPC timeout ({self._grpc_timeout}s) fetching {symbol}"
            raise DataFetchError(msg) from exc
        except Exception as exc:
            _log.exception(
                "candle_fetch_failed",
                symbol=symbol,
                figi=figi,
                timeframe=timeframe,
                error_type=type(exc).__name__,
            )
            msg = f"Tinkoff gRPC error fetching {symbol}: {exc}"
            raise DataFetchError(msg) from exc

        candles = [self._map_candle(c, symbol, timeframe) for c in raw_candles]  # type: ignore[attr-defined]
        _log.debug("candles_fetched", symbol=symbol, count=len(candles), timeframe=timeframe)
        return candles

    async def _fetch_async(
        self,
        figi: str,
        start: datetime,
        end: datetime,
        interval: CandleInterval,
    ) -> list[Any]:
        """Async call to Tinkoff SDK get_candles.

        Uses persistent gRPC channel via _get_services_async.
        """
        services = await self._get_services_async()
        response = await asyncio.wait_for(
            services.market_data.get_candles(  # type: ignore[attr-defined]
                figi=figi,
                from_=start,
                to=end,
                interval=interval,
            ),
            timeout=self._grpc_timeout,
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

    # ── Fundamentals (FUND-01) ─────────────────────────────────────────────

    async def _symbol_to_asset_uid(self, services: Any, ticker: str) -> str | None:
        """Resolve a MOEX ticker to its T-Bank asset_uid via share_by.

        Fundamentals are keyed on asset_uid, NOT FIGI (unlike every other call
        in this fetcher). Mirrors scripts/fetch_moex_dividends._resolve_figi but
        returns ``instrument.asset_uid``. Returns None on an unknown symbol
        (share_by raises for every class_code) — no raise.
        """
        from t_tech.invest.exceptions import AioRequestError, RequestError  # noqa: PLC0415
        from t_tech.invest.schemas import InstrumentIdType  # noqa: PLC0415

        for class_code in ("TQBR", "TQTF", "TQPI"):
            try:
                resp = await services.instruments.share_by(
                    id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                    class_code=class_code,
                    id=ticker,
                )
                return str(resp.instrument.asset_uid)
            except Exception as exc:  # try next class_code; None if all fail
                # IN-06: a "ticker not on this board" gRPC error is the expected
                # not-found signal (SDK RequestError/AioRequestError) -> debug + continue.
                # Anything else (e.g. an AttributeError from an SDK API change) is a
                # genuine bug masquerading as "symbol not found" and would silently
                # drop fundamentals -- log it at warning so real breakage is visible.
                # The "returns None when not found on any board" contract is unchanged.
                if isinstance(exc, (AioRequestError, RequestError)):
                    _log.debug(
                        "asset_uid_lookup_miss",
                        symbol=ticker,
                        class_code=class_code,
                        error_type=type(exc).__name__,
                    )
                else:
                    _log.warning(
                        "asset_uid_lookup_unexpected_error",
                        symbol=ticker,
                        class_code=class_code,
                        error_type=type(exc).__name__,
                    )
                continue
        return None

    def _map_fundamentals(self, stat: Any, symbol: str, as_of: datetime) -> FundamentalSnapshot:
        """Map a StatisticResponse to a FundamentalSnapshot.

        StatisticResponse fields are plain ``float``/``str``/``datetime`` — no
        Quotation/MoneyValue conversion. A protobuf scalar ``0.0`` is the unset
        default for a ratio/level, so it maps to ``None`` (Pitfall 1: never
        fabricate a P/E of zero).
        """

        def opt(v: Any) -> float | None:
            return None if v == 0.0 else float(v)

        return FundamentalSnapshot(
            symbol=symbol,
            as_of=as_of,
            pe_ratio=opt(stat.pe_ratio_ttm),
            ev_ebitda=opt(stat.ev_to_ebitda_mrq),
            revenue_ttm=opt(stat.revenue_ttm),
            net_margin=opt(stat.net_margin_mrq),
            roe=opt(stat.roe),
            eps_ttm=opt(stat.eps_ttm),
            dividend_yield=opt(stat.dividend_yield_daily_ttm),
            market_cap=opt(stat.market_capitalization),
            currency=stat.currency or None,
        )

    def fetch_fundamentals(self, symbol: str) -> FundamentalSnapshot | None:
        """Fetch a point-in-time FundamentalSnapshot for a MOEX symbol.

        Resolves the symbol to an ``asset_uid`` (NOT FIGI) via share_by, then
        calls ``get_asset_fundamentals``. Returns None on an unknown symbol or
        an empty response; missing scalar fields map to None (no fabrication).
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            result = self._run_async(self._fetch_fundamentals_async(symbol))
        except Exception as exc:
            _log.exception(
                "fundamentals_fetch_failed", symbol=symbol, error_type=type(exc).__name__
            )
            msg = f"Tinkoff gRPC error fetching fundamentals for {symbol}: {exc}"
            raise DataFetchError(msg) from exc
        _log.debug("fundamentals_fetched", symbol=symbol, found=result is not None)
        return result  # type: ignore[return-value]

    async def _fetch_fundamentals_async(self, symbol: str) -> FundamentalSnapshot | None:
        """Async worker: resolve asset_uid then get_asset_fundamentals."""
        from t_tech.invest.schemas import GetAssetFundamentalsRequest  # noqa: PLC0415

        services = await self._get_services_async()
        asset_uid = await self._symbol_to_asset_uid(services, symbol)
        if asset_uid is None:
            return None
        resp = await services.instruments.get_asset_fundamentals(  # type: ignore[attr-defined]
            GetAssetFundamentalsRequest(assets=[asset_uid]),
        )
        fundamentals = list(resp.fundamentals)
        if not fundamentals:
            return None
        stat = fundamentals[0]
        as_of = getattr(stat, "fiscal_period_end_date", None)
        if as_of is None:
            as_of = datetime.now(tz=UTC)
        elif as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        return self._map_fundamentals(stat, symbol, as_of)

    # ── Report calendar (EARN-01, calendar-only) ───────────────────────────

    def fetch_reports(self, symbol: str) -> list[ReportEvent]:
        """Fetch the earnings/report *calendar* for a MOEX symbol.

        ``get_asset_reports`` returns calendar events only (report_date +
        period metadata) — NO actual EPS/financials. The ``instrument_id`` is
        the registry FIGI (``_symbol_to_figi``), the same identifier every other
        Tinkoff instrument call in this fetcher uses; whether reports instead
        require the asset_uid is UNCONFIRMED and is the live-token MANUAL check
        (see 59-VALIDATION.md). Returns [] on an unknown symbol (no raise).
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            figi = self._symbol_to_figi(symbol)
        except InstrumentNotFoundError:
            _log.debug("reports_unknown_symbol", symbol=symbol)
            return []
        try:
            result = self._run_async(self._fetch_reports_async(figi, symbol))
        except Exception as exc:
            _log.exception("reports_fetch_failed", symbol=symbol, error_type=type(exc).__name__)
            msg = f"Tinkoff gRPC error fetching reports for {symbol}: {exc}"
            raise DataFetchError(msg) from exc
        _log.debug("reports_fetched", symbol=symbol, count=len(result))  # type: ignore[arg-type]
        return result  # type: ignore[return-value]

    async def _fetch_reports_async(self, figi: str, symbol: str) -> list[ReportEvent]:
        """Async worker: get_asset_reports keyed on the registry FIGI."""
        from datetime import timedelta  # noqa: PLC0415

        from t_tech.invest.schemas import GetAssetReportsRequest  # noqa: PLC0415

        now = datetime.now(tz=UTC)
        services = await self._get_services_async()
        # Registry FIGI is the DEFAULT instrument_id (live MANUAL check confirms).
        resp = await services.instruments.get_asset_reports(  # type: ignore[attr-defined]
            GetAssetReportsRequest(
                instrument_id=figi,
                from_=now - timedelta(days=730),
                to=now + timedelta(days=90),
            ),
        )
        return [self._map_report(ev, symbol) for ev in resp.events]

    def _map_report(self, ev: Any, symbol: str) -> ReportEvent:
        """Map a calendar GetAssetReportsEvent to a ReportEvent.

        period_year / period_num arrive as datetimes; period_num's quarter is
        derived from its month. period_type is the SDK enum name with the
        ``PERIOD_TYPE_`` prefix stripped. NO actuals are read (none exist).
        """
        report_date = ev.report_date
        if report_date.tzinfo is None:
            report_date = report_date.replace(tzinfo=UTC)
        period_year = (
            ev.period_year.year if hasattr(ev.period_year, "year") else int(ev.period_year)
        )
        if hasattr(ev.period_num, "month"):
            period_num = (ev.period_num.month - 1) // 3 + 1
        else:
            period_num = int(ev.period_num)
        period_type = ev.period_type.name.removeprefix("PERIOD_TYPE_")
        return ReportEvent(
            symbol=symbol,
            report_date=report_date,
            period_year=period_year,
            period_num=period_num,
            period_type=period_type,
        )

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
            raw_dividends = self._run_async(self._fetch_dividends_async(figi, start, end))
        except InstrumentNotFoundError:
            raise
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching dividends for {symbol}: {exc}"
            raise DataFetchError(msg) from exc

        return [self._map_dividend(d) for d in raw_dividends]  # type: ignore[attr-defined]

    async def _fetch_dividends_async(
        self,
        figi: str,
        start: datetime,
        end: datetime,
    ) -> list[Any]:
        """Async call to Tinkoff SDK get_dividends."""
        services = await self._get_services_async()
        response = await services.instruments.get_dividends(  # type: ignore[attr-defined]
            figi=figi,
            from_=start,
            to=end,
        )
        return list(response.dividends)

    # ── Bond discovery methods ─────────────────────────────────────────────

    def fetch_all_bonds(self) -> list[dict[str, Any]]:
        """Fetch all MOEX bonds from T-Invest API.

        Returns a list of dicts with bond metadata fields:
        figi, ticker, isin, name, lot, currency, nominal, initial_nominal,
        coupon_quantity_per_year, maturity_date, floating_coupon_flag,
        amortization_flag, risk_level, aci_value, class_code,
        subordinated_flag, liquidity_flag, sector, bond_type,
        call_date, perpetual_flag, api_trade_available_flag.

        Handles gRPC errors gracefully (logs and returns empty list).
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            return self._run_async(self._fetch_all_bonds_async())  # type: ignore[return-value]
        except Exception as exc:
            _log.exception("fetch_all_bonds_failed", error_type=type(exc).__name__)
            return []

    async def _fetch_all_bonds_async(self) -> list[dict[str, Any]]:
        """Async call to T-Bank SDK bonds()."""
        from t_tech.invest.schemas import InstrumentStatus  # noqa: PLC0415

        services = await self._get_services_async()
        resp = await services.instruments.bonds(  # type: ignore[attr-defined]
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
        )
        result: list[dict[str, Any]] = []
        for bond in resp.instruments:
            # Guard each money field (IN-01): the SDK can return None for these on
            # exotic instruments, and _money_to_decimal would AttributeError on None.
            # Consistent with the currency path (_fetch_all_currencies_async). Without
            # the guard, one malformed bond would propagate out and the broad
            # `except Exception: return []` in fetch_all_bonds would zero the whole list.
            nominal_raw = getattr(bond, "nominal", None)
            nominal = self._money_to_decimal(nominal_raw) if nominal_raw is not None else None
            initial_nominal_raw = getattr(bond, "initial_nominal", None)
            initial_nom = (
                self._money_to_decimal(initial_nominal_raw)
                if initial_nominal_raw is not None
                else None
            )
            aci_raw = getattr(bond, "aci_value", None)
            aci = self._money_to_decimal(aci_raw) if aci_raw is not None else None

            maturity = None
            if hasattr(bond, "maturity_date") and bond.maturity_date:
                maturity = (
                    bond.maturity_date.date()
                    if hasattr(bond.maturity_date, "date")
                    else bond.maturity_date
                )

            call_d = None
            if hasattr(bond, "call_date") and bond.call_date:
                call_d = (
                    bond.call_date.date() if hasattr(bond.call_date, "date") else bond.call_date
                )

            result.append(
                {
                    "figi": bond.figi,
                    "ticker": bond.ticker,
                    "isin": bond.isin,
                    "name": bond.name,
                    "lot": bond.lot,
                    "currency": bond.currency,
                    "nominal": nominal,
                    "initial_nominal": initial_nom,
                    "coupon_quantity_per_year": bond.coupon_quantity_per_year,
                    "maturity_date": maturity,
                    "floating_coupon_flag": bond.floating_coupon_flag,
                    "amortization_flag": bond.amortization_flag,
                    "risk_level": getattr(bond, "risk_level", 0),
                    "aci_value": aci,
                    "class_code": bond.class_code,
                    "subordinated_flag": getattr(bond, "subordinated_flag", False),
                    "liquidity_flag": getattr(bond, "liquidity_flag", False),
                    "sector": getattr(bond, "sector", ""),
                    "bond_type": getattr(bond, "bond_type", ""),
                    "call_date": call_d,
                    "perpetual_flag": getattr(bond, "perpetual_flag", False),
                    "api_trade_available_flag": getattr(bond, "api_trade_available_flag", False),
                }
            )
        return result

    # ── All-asset-class discovery (UNIV-03) ────────────────────────────────

    def fetch_all_shares(self) -> list[dict[str, Any]]:
        """Fetch all MOEX shares from T-Invest API (sibling of fetch_all_bonds).

        Each dict carries figi, ticker, isin, class_code, name, lot, currency,
        asset_uid, first_1day_candle_date. Filtered to real_exchange == MOEX.
        Handles gRPC errors gracefully (logs and returns empty list).
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            return self._run_async(self._fetch_all_shares_async())  # type: ignore[return-value]
        except Exception as exc:
            _log.exception("fetch_all_shares_failed", error_type=type(exc).__name__)
            return []

    def fetch_all_etfs(self) -> list[dict[str, Any]]:
        """Fetch all MOEX ETFs from T-Invest API (sibling of fetch_all_bonds).

        Same field shape as fetch_all_shares. Filtered to real_exchange == MOEX.
        Handles gRPC errors gracefully (logs and returns empty list).
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            return self._run_async(self._fetch_all_etfs_async())  # type: ignore[return-value]
        except Exception as exc:
            _log.exception("fetch_all_etfs_failed", error_type=type(exc).__name__)
            return []

    def fetch_all_futures(self) -> list[dict[str, Any]]:
        """Fetch all MOEX futures from T-Invest API (sibling of fetch_all_bonds).

        Each dict carries figi, ticker, class_code, name, lot, currency,
        basic_asset, expiration_date. Futures have no isin. Filtered to
        real_exchange == MOEX. Handles gRPC errors gracefully (returns []).
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            return self._run_async(self._fetch_all_futures_async())  # type: ignore[return-value]
        except Exception as exc:
            _log.exception("fetch_all_futures_failed", error_type=type(exc).__name__)
            return []

    def fetch_all_currencies(self) -> list[dict[str, Any]]:
        """Fetch all MOEX currencies from T-Invest API (sibling of fetch_all_bonds).

        Each dict carries figi, ticker, class_code, name, lot, currency, isin,
        nominal. Filtered to real_exchange == MOEX. Handles gRPC errors
        gracefully (logs and returns empty list).
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            return self._run_async(self._fetch_all_currencies_async())  # type: ignore[return-value]
        except Exception as exc:
            _log.exception("fetch_all_currencies_failed", error_type=type(exc).__name__)
            return []

    @staticmethod
    def _to_date(value: Any) -> date | None:
        """Normalize an SDK datetime/date to date (mirrors fetch_all_bonds)."""
        if not value:
            return None
        result: date = value.date() if hasattr(value, "date") else value
        return result

    async def _fetch_all_shares_async(self) -> list[dict[str, Any]]:
        """Async call to T-Bank SDK shares()."""
        from t_tech.invest.schemas import InstrumentStatus, RealExchange  # noqa: PLC0415

        services = await self._get_services_async()
        resp = await services.instruments.shares(  # type: ignore[attr-defined]
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
        )
        result: list[dict[str, Any]] = []
        for inst in resp.instruments:
            if inst.real_exchange != RealExchange.REAL_EXCHANGE_MOEX:
                continue  # MOEX filter -- one stable enum across all 5 classes
            result.append(
                {
                    "figi": getattr(inst, "figi", None),
                    "ticker": getattr(inst, "ticker", None),
                    "isin": getattr(inst, "isin", None),
                    "class_code": getattr(inst, "class_code", None),
                    "name": getattr(inst, "name", None),
                    "lot": getattr(inst, "lot", None),
                    "currency": getattr(inst, "currency", None),
                    "asset_uid": getattr(inst, "asset_uid", None),
                    "first_1day_candle_date": self._to_date(
                        getattr(inst, "first_1day_candle_date", None)
                    ),
                }
            )
        return result

    async def _fetch_all_etfs_async(self) -> list[dict[str, Any]]:
        """Async call to T-Bank SDK etfs()."""
        from t_tech.invest.schemas import InstrumentStatus, RealExchange  # noqa: PLC0415

        services = await self._get_services_async()
        resp = await services.instruments.etfs(  # type: ignore[attr-defined]
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
        )
        result: list[dict[str, Any]] = []
        for inst in resp.instruments:
            if inst.real_exchange != RealExchange.REAL_EXCHANGE_MOEX:
                continue  # MOEX filter
            result.append(
                {
                    "figi": getattr(inst, "figi", None),
                    "ticker": getattr(inst, "ticker", None),
                    "isin": getattr(inst, "isin", None),
                    "class_code": getattr(inst, "class_code", None),
                    "name": getattr(inst, "name", None),
                    "lot": getattr(inst, "lot", None),
                    "currency": getattr(inst, "currency", None),
                    "asset_uid": getattr(inst, "asset_uid", None),
                    "first_1day_candle_date": self._to_date(
                        getattr(inst, "first_1day_candle_date", None)
                    ),
                }
            )
        return result

    async def _fetch_all_futures_async(self) -> list[dict[str, Any]]:
        """Async call to T-Bank SDK futures()."""
        from t_tech.invest.schemas import InstrumentStatus, RealExchange  # noqa: PLC0415

        services = await self._get_services_async()
        resp = await services.instruments.futures(  # type: ignore[attr-defined]
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
        )
        result: list[dict[str, Any]] = []
        for inst in resp.instruments:
            if inst.real_exchange != RealExchange.REAL_EXCHANGE_MOEX:
                continue  # MOEX filter
            result.append(
                {
                    "figi": getattr(inst, "figi", None),
                    "ticker": getattr(inst, "ticker", None),
                    "class_code": getattr(inst, "class_code", None),
                    "name": getattr(inst, "name", None),
                    "lot": getattr(inst, "lot", None),
                    "currency": getattr(inst, "currency", None),
                    "basic_asset": getattr(inst, "basic_asset", None),
                    "expiration_date": self._to_date(getattr(inst, "expiration_date", None)),
                }
            )
        return result

    async def _fetch_all_currencies_async(self) -> list[dict[str, Any]]:
        """Async call to T-Bank SDK currencies()."""
        from t_tech.invest.schemas import InstrumentStatus, RealExchange  # noqa: PLC0415

        services = await self._get_services_async()
        resp = await services.instruments.currencies(  # type: ignore[attr-defined]
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE,
        )
        result: list[dict[str, Any]] = []
        for inst in resp.instruments:
            if inst.real_exchange != RealExchange.REAL_EXCHANGE_MOEX:
                continue  # MOEX filter
            nominal_raw = getattr(inst, "nominal", None)
            nominal = self._money_to_decimal(nominal_raw) if nominal_raw is not None else None
            result.append(
                {
                    "figi": getattr(inst, "figi", None),
                    "ticker": getattr(inst, "ticker", None),
                    "class_code": getattr(inst, "class_code", None),
                    "name": getattr(inst, "name", None),
                    "lot": getattr(inst, "lot", None),
                    "currency": getattr(inst, "currency", None),
                    "isin": getattr(inst, "isin", None),
                    "nominal": nominal,
                }
            )
        return result

    def fetch_amortization_schedule(self, instrument_id: str) -> list[dict[str, Any]]:
        """Fetch amortization schedule for a bond.

        Args:
            instrument_id: Bond FIGI or instrument UID.

        Returns:
            List of dicts with: event_date, pay_one_bond (Decimal), event_number.
        """
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        try:
            return self._run_async(self._fetch_amortization_async(instrument_id))  # type: ignore[return-value]
        except Exception as exc:
            _log.exception(
                "fetch_amortization_failed",
                instrument_id=instrument_id,
                error_type=type(exc).__name__,
            )
            return []

    async def _fetch_amortization_async(self, instrument_id: str) -> list[dict[str, Any]]:
        """Async call to T-Bank SDK get_bond_events for amortization."""
        from t_tech.invest.schemas import EventType, GetBondEventsRequest  # noqa: PLC0415

        services = await self._get_services_async()
        resp = await services.instruments.get_bond_events(  # type: ignore[attr-defined]
            request=GetBondEventsRequest(
                instrument_id=instrument_id,
                type=EventType.EVENT_TYPE_MTY,
            ),
        )
        events: list[dict[str, Any]] = []
        for ev in resp.events:
            ev_date = ev.event_date.date() if hasattr(ev.event_date, "date") else ev.event_date
            pay = self._money_to_decimal(ev.pay_one_bond)
            events.append(
                {
                    "event_date": ev_date,
                    "pay_one_bond": pay,
                    "event_number": ev.event_number,
                }
            )
        return events

    def fetch_bond_candles(
        self,
        figi: str,
        from_date: date,
        to_date: date,
        interval: CandleInterval | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch daily OHLCV candle data for a bond FIGI.

        Bond candle prices on MOEX are quoted as percentage of face value
        (e.g. 95.50 means 95.50% of face). Same convention as bond_math.py clean_price_pct.

        Args:
            figi: Bond FIGI identifier.
            from_date: Start date (inclusive).
            to_date: End date (inclusive).
            interval: Candle interval (defaults to daily).

        Returns:
            List of dicts with: date, open, high, low, close (Decimal), volume (int).
            Returns empty list on gRPC error.
        """
        if interval is None:
            interval = CandleInterval.CANDLE_INTERVAL_DAY

        if self._rate_limiter is not None:
            self._rate_limiter.acquire()

        try:
            coro = self._fetch_bond_candles_async(figi, from_date, to_date, interval)
            return self._run_async(coro)  # type: ignore[return-value]
        except Exception as exc:
            _log.exception("bond_candle_fetch_failed", figi=figi, error_type=type(exc).__name__)
            return []

    async def _fetch_bond_candles_async(
        self,
        figi: str,
        from_date: date,
        to_date: date,
        interval: CandleInterval,
    ) -> list[dict[str, Any]]:
        """Async call to T-Bank SDK get_candles for bonds."""
        start_dt = datetime.combine(from_date, datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(to_date, datetime.max.time(), tzinfo=UTC)

        services = await self._get_services_async()
        response = await services.market_data.get_candles(  # type: ignore[attr-defined]
            figi=figi,
            from_=start_dt,
            to=end_dt,
            interval=interval,
        )
        result: list[dict[str, Any]] = []
        for c in response.candles:
            ts = c.time
            candle_date = ts.date() if hasattr(ts, "date") else ts
            result.append(
                {
                    "date": candle_date,
                    "open": self._quotation_to_decimal(c.open),
                    "high": self._quotation_to_decimal(c.high),
                    "low": self._quotation_to_decimal(c.low),
                    "close": self._quotation_to_decimal(c.close),
                    "volume": int(c.volume),
                }
            )
        return result

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
            result = self._run_async(self._fetch_bond_info_async(figi))
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching bond info for {symbol} (FIGI={figi}): {exc}"
            raise DataFetchError(msg) from exc
        return result  # type: ignore[return-value]

    async def _fetch_bond_info_async(self, figi: str) -> BondInfo:
        """Async call to T-Bank SDK bond_by."""
        from t_tech.invest.schemas import InstrumentIdType  # noqa: PLC0415

        services = await self._get_services_async()
        resp = await services.instruments.bond_by(  # type: ignore[attr-defined]
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
            id=figi,
        )
        bond = resp.instrument
        nominal = self._money_to_decimal(bond.nominal)

        # For floating-coupon bonds, coupon_rate stores the spread over RUONIA.
        # Will be populated from coupon data when available.
        coupon_rate_val = Decimal(0)

        maturity = (
            bond.maturity_date.date() if hasattr(bond.maturity_date, "date") else bond.maturity_date
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
            result = self._run_async(self._fetch_bond_coupons_async(figi, start, end))
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching coupons for {symbol} (FIGI={figi}): {exc}"
            raise DataFetchError(msg) from exc
        return result  # type: ignore[return-value]

    async def _fetch_bond_coupons_async(
        self, figi: str, start: datetime, end: datetime
    ) -> list[CouponPayment]:
        """Async call to T-Bank SDK get_bond_coupons."""
        services = await self._get_services_async()
        resp = await services.instruments.get_bond_coupons(  # type: ignore[attr-defined]
            figi=figi,
            from_=start,
            to=end,
        )
        coupons: list[CouponPayment] = []
        for c in resp.events:
            amount = self._money_to_decimal(c.pay_one_bond)
            coupon_date = c.coupon_date.date() if hasattr(c.coupon_date, "date") else c.coupon_date
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
            result = self._run_async(self._fetch_accrued_interest_async(figi, start, end))
        except Exception as exc:
            msg = f"Tinkoff gRPC error fetching NKD for {symbol} (FIGI={figi}): {exc}"
            raise DataFetchError(msg) from exc
        return result  # type: ignore[return-value]

    async def _fetch_accrued_interest_async(
        self, figi: str, start: datetime, end: datetime
    ) -> list[AccruedInterest]:
        """Async call to T-Bank SDK get_accrued_interests."""
        services = await self._get_services_async()
        resp = await services.instruments.get_accrued_interests(  # type: ignore[attr-defined]
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
