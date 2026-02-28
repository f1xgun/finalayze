"""Telegram alerting for trade events and system notifications (Layer 0/6 boundary).

TelegramAlerter is stateless and fire-and-forget:
  - If bot_token is empty, all methods are no-ops (safe default for dev/test).
  - HTTP errors are caught and logged -- they never propagate to the trading loop.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from finalayze.execution.broker_base import OrderRequest, OrderResult
    from finalayze.risk.circuit_breaker import CircuitLevel

_TELEGRAM_API_BASE = "https://api.telegram.org/bot"
_SEND_MESSAGE_PATH = "/sendMessage"

_log = logging.getLogger(__name__)


class TelegramAlerter:
    """Sends Telegram messages for trade fills, rejections, circuit breaker events,
    daily summaries, and errors.

    When ``bot_token`` is an empty string, all methods return immediately
    without any network call (safe default for debug and test modes).
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id

    # ── Public API ───────────────────────────────────────────────────────────

    def on_trade_filled(self, result: OrderResult, market_id: str, broker: str) -> None:
        """Alert on a successful order fill.

        Example: ``BUY AAPL x10 @ $150.00 (Alpaca paper)``
        """
        price = result.fill_price if result.fill_price is not None else Decimal(0)
        text = (
            f"\U0001f7e2 {result.side} {result.symbol} \xd7{result.quantity} "
            f"@ ${price:.2f} ({broker} {market_id})"
        )
        self.send_alert(text)

    def on_trade_rejected(self, order: OrderRequest, reason: str) -> None:
        """Alert on an order rejection.

        Example: ``AAPL BUY rejected: insufficient funds``
        """
        text = f"\u26a0\ufe0f {order.symbol} {order.side} rejected: {reason}"
        self.send_alert(text)

    def on_circuit_breaker_trip(
        self, market_id: str, level: CircuitLevel, drawdown_pct: float
    ) -> None:
        """Alert on a circuit breaker level change.

        Example: ``[US] Circuit breaker HALTED -- trading halted (-10.3% daily)``
        """
        text = (
            f"\U0001f534 [{market_id.upper()}] Circuit breaker {level.upper()} "
            f"-- trading {level} ({drawdown_pct * 100:.1f}% daily drawdown)"
        )
        self.send_alert(text)

    def on_circuit_breaker_reset(self, market_id: str) -> None:
        """Alert on circuit breaker reset.

        Example: ``[US] Circuit breaker reset -- trading resumed``
        """
        text = f"\u2705 [{market_id.upper()}] Circuit breaker reset \u2014 trading resumed"
        self.send_alert(text)

    def on_daily_summary(
        self,
        market_pnl: dict[str, Decimal],
        total_equity_usd: Decimal,
    ) -> None:
        """Alert with daily P&L summary.

        Example: ``Daily: US +$342 | MOEX +1,200 | Equity $51,200``
        """
        parts = []
        for market_id, pnl in sorted(market_pnl.items()):
            sign = "+" if pnl >= Decimal(0) else ""
            parts.append(f"{market_id.upper()} {sign}{pnl}")
        summary = " | ".join(parts)
        text = f"\U0001f4ca Daily: {summary} | Equity ${total_equity_usd:,.0f}"
        self.send_alert(text)

    def on_error(self, component: str, message: str) -> None:
        """Alert on system errors.

        Example: ``TinkoffFetcher error: gRPC timeout``
        """
        text = f"\U0001f6a8 {component} error: {message}"
        self.send_alert(text)

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _send(self, text: str) -> None:
        """Async POST a message to the Telegram Bot API.

        Silently returns if token is empty or if any error occurs.
        """
        if not self._token:
            return

        url = f"{_TELEGRAM_API_BASE}{self._token}{_SEND_MESSAGE_PATH}"
        payload = {"chat_id": self._chat_id, "text": text}
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json=payload, timeout=10)
        except Exception:
            _log.exception("TelegramAlerter failed to send message")

    def send_alert(self, message: str) -> None:
        """Schedule or run ``_send`` safely from any thread context.

        If an asyncio event loop is running (e.g. inside the trading loop coroutine),
        a fire-and-forget task is created.  Otherwise ``asyncio.run()`` is used to
        send the message synchronously from the calling thread.

        Exceptions are always suppressed -- alerts must never crash the caller.
        """
        if not self._token:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _task = loop.create_task(self._send(message))  # noqa: RUF006 -- fire-and-forget
            else:
                asyncio.run(self._send(message))
        except Exception:
            _log.exception("TelegramAlerter send_alert failed")
