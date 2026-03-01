"""Tinkoff Invest broker for MOEX sandbox/live trading (Layer 5).

Uses t-tech-investments gRPC SDK wrapped in asyncio.run() to provide
a sync interface consistent with BrokerBase.

Lot-size aware: MOEX shares trade in lots. Quantity is always rounded
down to the nearest multiple of the instrument's lot_size.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import threading
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from t_tech.invest import AsyncClient, OrderDirection, OrderType
from t_tech.invest.sandbox.async_client import AsyncSandboxClient

from finalayze.core.exceptions import BrokerError, InstrumentNotFoundError
from finalayze.core.schemas import PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.execution.retry import RetryPolicy
    from finalayze.markets.instruments import InstrumentRegistry

_MOEX_MARKET_ID = "moex"
_NANO_DIVISOR = Decimal(1_000_000_000)


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
    ) -> None:
        self._token = token
        self._registry = registry
        self._sandbox = sandbox
        self._retry = retry_policy
        self._account_id: str = ""  # populated lazily on first API call
        self._client: AsyncClient | AsyncSandboxClient | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> AsyncClient | AsyncSandboxClient:
        """Return the persistent async client, creating it lazily."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:  # double-check
                    cls = AsyncSandboxClient if self._sandbox else AsyncClient
                    self._client = cls(self._token)
        return self._client

    def close(self) -> None:
        """Close the persistent gRPC channel."""
        if self._client is not None:
            with contextlib.suppress(Exception):
                asyncio.run(self._client.__aexit__(None, None, None))  # type: ignore[no-untyped-call]
            self._client = None

    def _call(self, fn: object) -> object:
        """Execute fn with retry if a RetryPolicy is configured."""
        if self._retry is not None:
            return self._retry.execute(fn)  # type: ignore[arg-type]
        return fn()  # type: ignore[operator]

    def _ensure_account_id(self) -> None:
        """Fetch and cache the account ID from the API if not already set."""
        if self._account_id:
            return
        response = self._call(lambda: asyncio.run(self._get_accounts_async()))
        accounts = getattr(response, "accounts", [])
        if not accounts:
            msg = "Tinkoff: no accounts found for the provided token"
            raise BrokerError(msg)
        self._account_id = accounts[0].id

    async def _get_accounts_async(self) -> object:
        """Async call to fetch accounts list."""
        client = self._get_client()
        return await client.users.get_accounts()  # type: ignore[attr-defined]

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
                lambda: asyncio.run(self._post_order_async(figi, actual_qty, direction))
            )
        except InstrumentNotFoundError:
            raise
        except Exception as exc:
            msg = f"Tinkoff order failed for {order.symbol}: {exc}"
            raise BrokerError(msg) from exc

        fill_price = self._quotation_to_decimal(result.executed_order_price)  # type: ignore[attr-defined]
        return OrderResult(
            filled=True,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=Decimal(actual_qty),
        )

    async def _post_order_async(
        self,
        figi: str,
        quantity: int,
        direction: OrderDirection,
    ) -> object:
        """Async call to Tinkoff SDK post_order."""
        client = self._get_client()
        return await client.orders.post_order(  # type: ignore[attr-defined]
            figi=figi,
            quantity=quantity,
            direction=direction,
            order_type=OrderType.ORDER_TYPE_MARKET,
            account_id=self._account_id,
        )

    def get_portfolio(self) -> PortfolioState:
        """Return current MOEX portfolio state from Tinkoff."""
        try:
            self._ensure_account_id()
            portfolio = self._call(lambda: asyncio.run(self._get_portfolio_async()))
        except Exception as exc:
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

        return PortfolioState(
            cash=cash_sum,
            positions=pos_map,
            equity=total,
            timestamp=datetime.now(tz=UTC),
        )

    async def _get_portfolio_async(self) -> object:
        """Async call to Tinkoff SDK get_portfolio."""
        client = self._get_client()
        return await client.operations.get_portfolio(account_id=self._account_id)  # type: ignore[attr-defined]

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
            self._call(lambda: asyncio.run(self._cancel_order_async(order_id)))
        except Exception as exc:
            msg = f"Tinkoff cancel_order failed for {order_id}: {exc}"
            raise BrokerError(msg) from exc

    async def _cancel_order_async(self, order_id: str) -> None:
        """Async call to Tinkoff SDK cancel_order."""
        client = self._get_client()
        await client.orders.cancel_order(account_id=self._account_id, order_id=order_id)  # type: ignore[attr-defined]

    @staticmethod
    def _quotation_to_decimal(q: object) -> Decimal:
        """Convert Tinkoff Quotation(units, nano) to Decimal."""
        units = getattr(q, "units", 0)
        nano = getattr(q, "nano", 0)
        return Decimal(units) + Decimal(nano) / _NANO_DIVISOR
