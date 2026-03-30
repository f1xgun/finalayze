"""Tinkoff Invest broker for MOEX sandbox/live trading (Layer 5).

Uses t-tech-investments gRPC SDK wrapped in asyncio.run() to provide
a sync interface consistent with BrokerBase.

Lot-size aware: MOEX shares trade in lots. Quantity is always rounded
down to the nearest multiple of the instrument's lot_size.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog
from t_tech.invest import AsyncClient, OrderDirection, OrderType

from finalayze.core.exceptions import BrokerError, InstrumentNotFoundError
from finalayze.core.schemas import PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult

_log = structlog.get_logger()

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.execution.retry import RetryPolicy
    from finalayze.markets.instruments import InstrumentRegistry

# gRPC C-ares DNS resolver may fail on some systems; use native resolver.
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

_MOEX_MARKET_ID = "moex"
_NANO_DIVISOR = Decimal(1_000_000_000)
_TBANK_GRPC_TARGET = "invest-public-api.tbank.ru:443"
_TBANK_GRPC_SANDBOX_TARGET = "sandbox-invest-public-api.tbank.ru:443"

# T-Invest execution report status codes -> human-readable names
_EXECUTION_STATUS_MAP: dict[int, str] = {
    0: "unspecified",
    1: "fill",
    2: "partially_fill",
    3: "cancelled",
    4: "new",
    5: "rejected",
}
_TERMINAL_STATUSES = frozenset({"fill", "cancelled", "rejected"})


@dataclass(frozen=True)
class OrderStateResult:
    """Result of querying order state from T-Invest API."""

    order_id: str
    execution_status: str  # "fill", "partially_fill", "new", "cancelled", "rejected"
    filled_quantity: Decimal
    filled_price: Decimal
    is_terminal: bool  # True if fill, cancelled, rejected


class TinkoffBroker(BrokerBase):
    """Tinkoff Invest broker for MOEX market.

    Uses sandbox endpoint when sandbox=True (for development/testing).
    Lot-size aware: quantities are rounded down to the nearest lot multiple.
    Raises InstrumentNotFoundError for unknown symbols, BrokerError for API failures.
    """

    def __init__(
        self,
        token: str,
        registry: InstrumentRegistry,
        *,
        sandbox: bool = True,
        retry_policy: RetryPolicy | None = None,
        grpc_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._token = token
        self._registry = registry
        self._sandbox = sandbox
        self._retry = retry_policy
        self._account_id: str = ""  # populated lazily on first API call
        self._client: AsyncClient | None = None
        self._services: object | None = None  # AsyncServices from __aenter__
        self._client_lock = threading.Lock()  # for sync _get_client only
        self._async_lock = asyncio.Lock()  # for async _get_services_async
        # Externally injected gRPC event loop (from TradingLoop / run_sandbox.py).
        # When set, _run_async uses this loop instead of creating a self-managed one.
        self._grpc_loop = grpc_loop
        # Persistent background event loop for gRPC — asyncio.run() closes the
        # loop after each call, killing the gRPC channel. We keep a daemon thread
        # with its own loop alive. Uses run_coroutine_threadsafe() so it's safe
        # to call from any thread (APScheduler, uvicorn, etc.).
        # Only used as fallback when grpc_loop is not injected (tests, standalone).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_init_lock = threading.Lock()  # guards event loop creation

    async def _get_services_async(self) -> object:
        """Return the persistent AsyncServices, creating client lazily.

        AsyncClient must be entered via __aenter__ to get AsyncServices
        which provides .users, .orders, .operations, .market_data etc.
        """
        if self._services is None:
            async with self._async_lock:
                if self._services is None:  # double-check
                    target = _TBANK_GRPC_SANDBOX_TARGET if self._sandbox else _TBANK_GRPC_TARGET
                    self._client = AsyncClient(self._token, target=target)
                    self._services = await self._client.__aenter__()
        return self._services

    # Keep backward compat alias
    def _get_client(self) -> AsyncClient:
        """Return raw AsyncClient (deprecated — use _get_services_async)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    target = _TBANK_GRPC_SANDBOX_TARGET if self._sandbox else _TBANK_GRPC_TARGET
                    self._client = AsyncClient(self._token, target=target)
        return self._client

    def close(self) -> None:
        """Close the persistent gRPC channel and event loop.

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
                pass
            self._loop = None
            self._loop_thread = None

    def _run_async(self, coro: object) -> object:
        """Run an async coroutine on a persistent background event loop.

        Uses run_coroutine_threadsafe() so it's safe to call from any thread
        (APScheduler workers, uvicorn, init code). The background loop stays
        alive so gRPC channels survive across multiple calls.

        When an external grpc_loop is injected, uses that directly (no
        self-managed loop). Falls back to creating a self-managed loop for
        backward compatibility (tests, standalone scripts).
        """
        _timeout = 30
        loop = self._grpc_loop
        if loop is None:
            # Fallback: self-managed loop for backward compat (tests, standalone use)
            if self._loop is None or self._loop.is_closed():
                with self._loop_init_lock:
                    if self._loop is None or self._loop.is_closed():  # double-check
                        new_loop = asyncio.new_event_loop()
                        self._loop = new_loop
                        thread = threading.Thread(target=new_loop.run_forever, daemon=True)
                        thread.start()
                        self._loop_thread = thread
            loop = self._loop
        future = asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
        return future.result(timeout=_timeout)

    def _call(self, fn: object) -> object:
        """Execute fn with retry if a RetryPolicy is configured."""
        if self._retry is not None:
            return self._retry.execute(fn)  # type: ignore[arg-type]
        return fn()  # type: ignore[operator]

    def _ensure_account_id(self) -> None:
        """Fetch and cache the account ID from the API if not already set."""
        if self._account_id:
            return
        response = self._call(lambda: self._run_async(self._get_accounts_async()))
        accounts = getattr(response, "accounts", [])
        if not accounts:
            msg = "Tinkoff: no accounts found for the provided token"
            _log.error("tinkoff_no_accounts")
            raise BrokerError(msg)
        self._account_id = accounts[0].id
        _log.info("tinkoff_account_resolved", account_id=self._account_id, sandbox=self._sandbox)

    async def _get_accounts_async(self) -> object:
        """Async call to fetch accounts list."""
        services = await self._get_services_async()
        return await services.users.get_accounts()  # type: ignore[attr-defined]

    def submit_order(
        self,
        order: OrderRequest,
        fill_candle: Candle | None = None,  # noqa: ARG002 -- ignored for live broker
    ) -> OrderResult:
        """Submit a market order to Tinkoff Invest. fill_candle is not used."""
        instrument = self._registry.get(order.symbol, _MOEX_MARKET_ID)
        if instrument.figi is None:
            msg = f"Instrument '{order.symbol}' has no FIGI assigned"
            raise InstrumentNotFoundError(msg)
        figi: str = instrument.figi

        # Round quantity down to nearest lot multiple
        lot_size = instrument.lot_size
        actual_qty = math.floor(float(order.quantity) / lot_size) * lot_size

        if actual_qty <= 0:
            _log.warning(
                "order_rejected_lot_size",
                symbol=order.symbol,
                side=order.side,
                requested_qty=str(order.quantity),
                lot_size=lot_size,
            )
            return OrderResult(
                filled=False,
                symbol=order.symbol,
                side=order.side,
                quantity=Decimal(0),
                reason=f"Quantity {order.quantity} is less than lot size {lot_size}",
            )

        direction = (
            OrderDirection.ORDER_DIRECTION_BUY
            if order.side == "BUY"
            else OrderDirection.ORDER_DIRECTION_SELL
        )

        try:
            self._ensure_account_id()
            result = self._call(
                lambda: self._run_async(self._post_order_async(figi, actual_qty, direction))
            )
        except InstrumentNotFoundError:
            raise
        except Exception as exc:
            _log.exception(
                "order_submission_failed",
                symbol=order.symbol,
                side=order.side,
                qty=actual_qty,
                figi=figi,
            )
            msg = f"Tinkoff order failed for {order.symbol}: {exc}"
            raise BrokerError(msg) from exc

        fill_price = self._quotation_to_decimal(result.executed_order_price)  # type: ignore[attr-defined]
        result_order_id: str = getattr(result, "order_id", "")
        _log.info(
            "order_filled",
            symbol=order.symbol,
            side=order.side,
            qty=actual_qty,
            fill_price=float(fill_price),
            figi=figi,
            order_id=result_order_id,
        )
        return OrderResult(
            filled=True,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=Decimal(actual_qty),
            order_id=result_order_id,
        )

    async def _post_order_async(
        self,
        figi: str,
        quantity: int,
        direction: OrderDirection,
    ) -> object:
        """Async call to Tinkoff SDK post_order."""
        client = await self._get_services_async()
        return await client.orders.post_order(  # type: ignore[attr-defined]
            figi=figi,
            quantity=quantity,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
            account_id=self._account_id,
        )

    def get_last_prices(self, symbols: list[str]) -> dict[str, Decimal]:
        """Fetch last prices for given symbols via T-Invest GetLastPrices.

        Maps symbols to FIGIs via the instrument registry, calls the API,
        and maps results back to symbol keys.

        Args:
            symbols: List of ticker symbols (e.g. ["SU26238RMFS4"]).

        Returns:
            Dict of symbol -> price as Decimal (% of face for bonds).
        """
        # Build symbol <-> FIGI mappings
        symbol_to_figi: dict[str, str] = {}
        figi_to_symbol: dict[str, str] = {}
        for sym in symbols:
            try:
                instrument = self._registry.get(sym, _MOEX_MARKET_ID)
                if instrument.figi is not None:
                    symbol_to_figi[sym] = instrument.figi
                    figi_to_symbol[instrument.figi] = sym
            except Exception:
                _log.warning("get_last_prices_symbol_not_found", symbol=sym)

        if not symbol_to_figi:
            return {}

        figis = list(symbol_to_figi.values())
        try:
            self._ensure_account_id()
            response = self._call(lambda: self._run_async(self._get_last_prices_async(figis)))
        except Exception as exc:
            msg = f"Tinkoff get_last_prices failed: {exc}"
            raise BrokerError(msg) from exc

        result: dict[str, Decimal] = {}
        for item in response.last_prices:  # type: ignore[attr-defined]
            figi = item.figi
            sym = figi_to_symbol.get(figi)
            if sym is not None:
                price = self._quotation_to_decimal(item.price)
                result[sym] = price

        return result

    async def _get_last_prices_async(self, figis: list[str]) -> object:
        """Async call to T-Invest GetLastPrices."""
        client = await self._get_services_async()
        return await client.market_data.get_last_prices(figi=figis)  # type: ignore[attr-defined]

    def get_order_state(self, order_id: str) -> OrderStateResult:
        """Query order state from T-Invest API.

        Args:
            order_id: The broker-assigned order identifier.

        Returns:
            OrderStateResult with execution status, filled quantity/price, terminal flag.
        """
        try:
            self._ensure_account_id()
            state = self._call(lambda: self._run_async(self._get_order_state_async(order_id)))
        except Exception as exc:
            msg = f"Tinkoff get_order_state failed for {order_id}: {exc}"
            raise BrokerError(msg) from exc

        raw_status = getattr(state, "execution_report_status", 0)
        status_str = _EXECUTION_STATUS_MAP.get(raw_status, "unspecified")
        filled_qty = Decimal(getattr(state, "lots_executed", 0))
        filled_price = self._quotation_to_decimal(state.executed_order_price)  # type: ignore[attr-defined]

        return OrderStateResult(
            order_id=order_id,
            execution_status=status_str,
            filled_quantity=filled_qty,
            filled_price=filled_price,
            is_terminal=status_str in _TERMINAL_STATUSES,
        )

    async def _get_order_state_async(self, order_id: str) -> object:
        """Async call to T-Invest get_order_state."""
        client = await self._get_services_async()
        return await client.orders.get_order_state(  # type: ignore[attr-defined]
            account_id=self._account_id,
            order_id=order_id,
        )

    def get_portfolio(self) -> PortfolioState:
        """Return current MOEX portfolio state from Tinkoff."""
        try:
            self._ensure_account_id()
            portfolio = self._call(lambda: self._run_async(self._get_portfolio_async()))
        except Exception as exc:
            _log.exception("portfolio_fetch_failed")
            msg = f"Tinkoff portfolio fetch failed: {exc}"
            raise BrokerError(msg) from exc

        total = self._quotation_to_decimal(portfolio.total_amount_portfolio)  # type: ignore[attr-defined]
        pos_map: dict[str, Decimal] = {}
        cash_sum = Decimal(0)
        for pos in portfolio.positions:  # type: ignore[attr-defined]
            qty = self._quotation_to_decimal(pos.quantity)
            if getattr(pos, "instrument_type", "") == "currency":
                cash_sum += qty
            else:
                pos_map[pos.figi] = qty  # Tinkoff positions are FIGI-keyed

        _log.debug(
            "portfolio_fetched",
            equity=float(total),
            cash=float(cash_sum),
            positions=len(pos_map),
        )
        return PortfolioState(
            cash=cash_sum,
            positions=pos_map,
            equity=total,
            timestamp=datetime.now(tz=UTC),
        )

    async def _get_portfolio_async(self) -> object:
        """Async call to Tinkoff SDK get_portfolio."""
        services = await self._get_services_async()
        return await services.operations.get_portfolio(account_id=self._account_id)  # type: ignore[attr-defined]

    def has_position(self, symbol: str) -> bool:
        """Return True if Tinkoff account holds a non-zero position in symbol."""
        instrument = self._registry.get(symbol, _MOEX_MARKET_ID)
        if instrument.figi is None:
            msg = f"Instrument '{symbol}' has no FIGI assigned"
            raise InstrumentNotFoundError(msg)
        figi = instrument.figi
        portfolio = self.get_portfolio()
        held = portfolio.positions.get(figi, Decimal(0))
        return held > 0

    def get_positions(self) -> dict[str, Decimal]:
        """Return current Tinkoff positions (FIGI-keyed) as Decimal quantities."""
        return dict(self.get_portfolio().positions)

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending Tinkoff order by ID."""
        try:
            self._ensure_account_id()
            self._call(lambda: self._run_async(self._cancel_order_async(order_id)))
        except Exception as exc:
            msg = f"Tinkoff cancel_order failed for {order_id}: {exc}"
            raise BrokerError(msg) from exc

    async def _cancel_order_async(self, order_id: str) -> None:
        """Async call to Tinkoff SDK cancel_order."""
        client = await self._get_services_async()
        await client.orders.cancel_order(account_id=self._account_id, order_id=order_id)  # type: ignore[attr-defined]

    def reconnect_client(self) -> bool:
        """Destroy existing gRPC client and create a new one.

        Thread-safe. Resets _account_id and re-fetches it from the API.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        with self._client_lock:
            # Close old client (suppress exceptions)
            self.close()
            self._account_id = ""
            try:
                target = _TBANK_GRPC_SANDBOX_TARGET if self._sandbox else _TBANK_GRPC_TARGET
                self._client = AsyncClient(self._token, target=target)
                self._ensure_account_id()
                _log.info(
                    "tinkoff_reconnected",
                    account_id=self._account_id,
                    sandbox=self._sandbox,
                )
                return True
            except Exception:
                _log.exception("tinkoff_reconnect_failed")
                self._client = None
                return False

    def get_open_orders(self) -> list[OrderStateResult]:
        """Return all non-terminal orders in the account.

        Returns empty list on API failure (logs warning, does not raise).
        """
        try:
            self._ensure_account_id()
            response = self._call(lambda: self._run_async(self._get_orders_async()))
            orders: list[OrderStateResult] = []
            for order in getattr(response, "orders", []):
                raw_status = getattr(order, "execution_report_status", 0)
                status_str = _EXECUTION_STATUS_MAP.get(raw_status, "unspecified")
                if status_str in _TERMINAL_STATUSES:
                    continue
                filled_qty = Decimal(getattr(order, "lots_executed", 0))
                filled_price = self._quotation_to_decimal(order.executed_order_price)
                orders.append(
                    OrderStateResult(
                        order_id=order.order_id,
                        execution_status=status_str,
                        filled_quantity=filled_qty,
                        filled_price=filled_price,
                        is_terminal=False,
                    )
                )
            return orders
        except Exception:
            _log.warning("get_open_orders_failed", exc_info=True)
            return []

    async def _get_orders_async(self) -> object:
        """Async call to T-Invest get_orders."""
        client = await self._get_services_async()
        return await client.orders.get_orders(account_id=self._account_id)  # type: ignore[attr-defined]

    def cancel_order_safe(self, order_id: str) -> bool:
        """Cancel a specific order by ID. Returns True on success, False on error.

        Unlike cancel_order(), this method does not raise on failure.
        """
        try:
            self._ensure_account_id()
            self._call(lambda: self._run_async(self._cancel_order_async(order_id)))
            _log.info("order_cancelled", order_id=order_id)
            return True
        except Exception:
            _log.warning("cancel_order_safe_failed", order_id=order_id, exc_info=True)
            return False

    @staticmethod
    def _quotation_to_decimal(q: object) -> Decimal:
        """Convert Tinkoff Quotation(units, nano) to Decimal."""
        units = getattr(q, "units", 0)
        nano = getattr(q, "nano", 0)
        return Decimal(units) + Decimal(nano) / _NANO_DIVISOR


def make_bond_broker(equity_broker: TinkoffBroker) -> TinkoffBroker:
    """Create a TinkoffBroker for bonds sharing the equity broker's gRPC client.

    Reuses the same AsyncClient (and therefore the same gRPC channel) to avoid
    opening a second connection.  The returned instance is a separate object
    so it can carry bond-specific state if needed.

    Args:
        equity_broker: The existing equity TinkoffBroker instance.

    Returns:
        A new TinkoffBroker sharing the same AsyncClient.
    """
    bond_broker = TinkoffBroker(
        token=equity_broker._token,
        registry=equity_broker._registry,
        sandbox=equity_broker._sandbox,
        retry_policy=equity_broker._retry,
        grpc_loop=equity_broker._grpc_loop,
    )
    # Share the same gRPC client to avoid a second connection
    bond_broker._client = equity_broker._client
    bond_broker._account_id = equity_broker._account_id
    return bond_broker
