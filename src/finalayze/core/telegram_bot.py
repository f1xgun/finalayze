"""Telegram bot handler for webhook-based commands (Layer 6 boundary).

Provides read-only commands for querying system state via Telegram:
  - /status: portfolio positions, equity, P&L per market
  - /breakers: circuit breaker states for all layers

No trading commands are exposed -- this is read-only per design decision.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.core.alerts import TelegramAlerter
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.risk.circuit_breaker import CircuitBreaker

_log = structlog.get_logger()
_ZERO = Decimal(0)


class TelegramBotHandler:
    """Handles incoming Telegram webhook updates with read-only commands.

    Validates chat_id against settings whitelist. Dispatches to command
    handlers. Uses TelegramAlerter._send directly for interactive responses
    (bypasses queue for immediate feedback).
    """

    def __init__(
        self,
        alerter: TelegramAlerter,
        broker_router: BrokerRouter,
        circuit_breakers: dict[str, CircuitBreaker],
        settings: Settings,
        bond_processor: object | None = None,
        trading_loop: object | None = None,
    ) -> None:
        self._alerter = alerter
        self._broker_router = broker_router
        self._circuit_breakers = circuit_breakers
        self._settings = settings
        self._bond_processor = bond_processor
        self._trading_loop = trading_loop

        self._commands: dict[str, Any] = {
            "/status": self.handle_status,
            "/breakers": self.handle_breakers,
            "/stop": self.handle_stop,
        }

    async def handle_update(self, update: dict[str, Any]) -> dict[str, str]:
        """Process a raw Telegram update JSON.

        Extracts message.chat.id and message.text. Validates chat_id
        against whitelist. Dispatches to registered command handler.

        Returns:
            {"ok": "processed"} on success, {"ok": "ignored"} if not whitelisted
            or no recognized command.
        """
        message = update.get("message")
        if message is None:
            return {"ok": "no_message"}

        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))
        text = (message.get("text") or "").strip()

        # Validate chat_id against whitelist
        allowed = self._settings.telegram_allowed_chat_ids
        if chat_id not in allowed:
            _log.debug("telegram_chat_id_rejected", chat_id=chat_id)
            return {"ok": "ignored"}

        # Extract command (first word)
        command = text.split()[0] if text else ""
        handler = self._commands.get(command)

        if handler is None:
            return {"ok": "no_command"}

        try:
            await handler(chat_id)
            return {"ok": "processed"}
        except Exception:
            _log.exception("telegram_command_failed", command=command, chat_id=chat_id)
            return {"ok": "error"}

    async def handle_status(self, chat_id: str) -> None:  # noqa: ARG002
        """Query broker_router for portfolio state and send formatted response.

        Shows equity, positions, and cash per market. If bond_processor is
        available, includes bond layer status.

        Args:
            chat_id: Telegram chat ID (reserved for per-chat responses).
        """
        lines: list[str] = ["<b>Portfolio Status</b>\n"]

        if self._broker_router is None:
            lines.append("Broker not connected (API-only mode)")
        else:
            markets = self._broker_router.registered_markets
            for market_id in markets:
                try:
                    broker = self._broker_router.route(market_id)
                    portfolio = broker.get_portfolio()
                    equity = portfolio.equity
                    cash = portfolio.cash
                    positions = portfolio.positions

                    lines.append(f"<b>{market_id.upper()}</b>")
                    lines.append(f"  Equity: <code>{equity:,.2f}</code>")
                    lines.append(f"  Cash: <code>{cash:,.2f}</code>")

                    if positions:
                        for sym, qty in sorted(positions.items()):
                            if qty > _ZERO:
                                lines.append(f"  {sym}: <code>{qty}</code>")
                    else:
                        lines.append("  No positions")
                    lines.append("")
                except Exception:
                    lines.append(f"<b>{market_id.upper()}</b>: unavailable\n")
                    _log.debug("telegram_status_market_failed", market_id=market_id)

        # Bond layer status
        if self._bond_processor is not None:
            try:
                ledgers = getattr(self._bond_processor, "_layer_ledgers", {})
                if ledgers:
                    lines.append("<b>Bond Layers</b>")
                    for layer, ledger in ledgers.items():
                        layer_name = layer.value if hasattr(layer, "value") else str(layer)
                        lines.append(
                            f"  {layer_name}: equity=<code>{ledger.current_equity:,.2f}</code>"
                        )
                    lines.append("")
            except Exception:
                _log.debug("telegram_status_bonds_failed")

        await self._alerter._send("\n".join(lines))

    async def handle_stop(self, chat_id: str) -> None:  # noqa: ARG002
        """Emergency stop: halt all trading cycles.

        Args:
            chat_id: Telegram chat ID (reserved for per-chat responses).
        """
        _log.critical("telegram_stop_command", chat_id=chat_id)
        if self._trading_loop is not None:
            try:
                self._trading_loop.stop()  # type: ignore[union-attr]
                await self._alerter._send(
                    "<b>TRADING HALTED</b>\n\n"
                    "All cycles stopped. Manual restart required to resume."
                )
            except Exception:
                _log.exception("telegram_stop_failed")
                await self._alerter._send(
                    "<b>STOP FAILED</b>\n\nCheck logs. Scheduler may still be running."
                )
        else:
            await self._alerter._send(
                "<b>STOP: No trading loop</b>\n\nRunning in API-only mode."
            )

    async def handle_breakers(self, chat_id: str) -> None:  # noqa: ARG002
        """Show circuit breaker states for all layers.

        Args:
            chat_id: Telegram chat ID (reserved for per-chat responses).
        """
        lines: list[str] = ["<b>Circuit Breakers</b>\n"]

        if not self._circuit_breakers:
            lines.append("No circuit breakers configured (API-only mode)")

        for market_id, cb in sorted(self._circuit_breakers.items()):
            level = cb.level
            baseline = cb.baseline
            lines.append(
                f"<b>{market_id.upper()}</b>: {level.upper()} "
                f"(baseline: <code>{baseline:,.0f}</code>)"
            )

        # Bond layer breakers
        if self._bond_processor is not None:
            try:
                layer_breakers = getattr(self._bond_processor, "_layer_breakers", {})
                agg_breaker = getattr(self._bond_processor, "_aggregate_breaker", None)

                if layer_breakers:
                    lines.append("\n<b>Bond Layer Breakers</b>")
                    for layer, breaker in layer_breakers.items():
                        layer_name = layer.value if hasattr(layer, "value") else str(layer)
                        halted = getattr(breaker, "is_halted", False)
                        status = "HALTED" if halted else "OK"
                        lines.append(f"  {layer_name}: {status}")

                if agg_breaker is not None:
                    agg_halted = getattr(agg_breaker, "is_halted", False)
                    agg_status = "HALTED" if agg_halted else "OK"
                    lines.append(f"  Aggregate: {agg_status}")
            except Exception:
                _log.debug("telegram_breakers_bonds_failed")

        await self._alerter._send("\n".join(lines))
