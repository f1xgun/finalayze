"""Telegram bot handler for webhook-based commands (Layer 6).

Provides commands for querying and controlling system state via Telegram:
  - /status: portfolio positions, equity, P&L per market
  - /breakers: circuit breaker states for all layers
  - /stop: halt all trading cycles
  - /kill: emergency shutdown with 30s confirmation (admin-only)
  - /gonogo: run go/no-go gate evaluation
  - /approve <id8>: (Phase 58-04) approve a meta-agent FIX-severity decision
    within 30 min of its Telegram alert; dispatches the FIX spawn pipeline.

Moved from core/ to api/ in Phase 22 (dependency layer cleanup).
See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import re
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from config.settings import Settings

    from finalayze.api.alerts import TelegramAlerter
    from finalayze.core.kill_switch import KillSwitch
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.meta_agent.approver import MetaAgentApprover
    from finalayze.monitoring.go_no_go import GoNoGoReporter
    from finalayze.risk.circuit_breaker import CircuitBreaker

_log = structlog.get_logger()
_ZERO = Decimal(0)

# 58-04 D-12: locked /approve syntax — anchored regex, case-insensitive on the
# command itself BUT case-sensitive on the hex (UUID short8 is always lowercase).
# No extra arguments are accepted — anchor to end of string.
_APPROVE_PATTERN = re.compile(r"^/approve\s+([0-9a-f]{8})\s*$", re.IGNORECASE)


class TelegramBotHandler:
    """Handles incoming Telegram webhook updates with read-only commands.

    Validates chat_id against settings whitelist. Dispatches to command
    handlers. Uses TelegramAlerter._send directly for interactive responses
    (bypasses queue for immediate feedback).
    """

    _KILL_CONFIRM_TIMEOUT_S = 30
    _KILL_CLEANUP_THRESHOLD_S = 60

    def __init__(
        self,
        alerter: TelegramAlerter,
        broker_router: BrokerRouter,
        circuit_breakers: dict[str, CircuitBreaker],
        settings: Settings,
        bond_processor: object | None = None,
        trading_loop: Any | None = None,
        kill_switch: KillSwitch | None = None,
        go_no_go_reporter: GoNoGoReporter | None = None,
        meta_agent_approver: MetaAgentApprover | None = None,
    ) -> None:
        self._alerter = alerter
        self._broker_router = broker_router
        self._circuit_breakers = circuit_breakers
        self._settings = settings
        self._bond_processor = bond_processor
        self._trading_loop = trading_loop
        self._kill_switch = kill_switch
        self._go_no_go_reporter = go_no_go_reporter
        # 58-04: meta-agent /approve dispatcher (None when meta-agent not deployed).
        self._meta_agent_approver = meta_agent_approver
        self._pending_kill: dict[str, float] = {}
        # 58-04 RUF006: store fire-and-forget /approve task handles on the
        # owning instance so the event-loop tracks their lifetime.
        self._pending_approve_tasks: set[asyncio.Task[Any]] = set()

        self._commands: dict[str, Any] = {
            "/status": self.handle_status,
            "/breakers": self.handle_breakers,
            "/stop": self.handle_stop,
            "/kill": self.handle_kill,
            "/gonogo": self.handle_gonogo,
            "/approve": self.handle_approve,
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

        # Clean up expired kill confirmations (>60s old)
        self._cleanup_expired_kills()

        # Check for CONFIRM text (non-command kill confirmation)
        if text.upper() == "CONFIRM" and chat_id in self._pending_kill:
            return await self._handle_kill_confirm(chat_id)

        # Extract command (first word)
        command = text.split()[0] if text else ""
        handler = self._commands.get(command)

        if handler is None:
            return {"ok": "no_command"}

        try:
            # 58-04 D-12 + RESEARCH §5.2 Open Q #4: /approve is the first
            # command that requires the raw text (to parse the short8 arg).
            # Use an explicit branch so existing handlers keep their
            # `(chat_id)`-only signature.
            if command == "/approve":
                await handler(chat_id, raw_text=text)
            else:
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

    async def handle_stop(self, chat_id: str) -> None:
        """Emergency stop: halt all trading cycles.

        Args:
            chat_id: Telegram chat ID (reserved for per-chat responses).
        """
        _log.critical("telegram_stop_command", chat_id=chat_id)
        if self._trading_loop is not None:
            try:
                self._trading_loop.stop()
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
            await self._alerter._send("<b>STOP: No trading loop</b>\n\nRunning in API-only mode.")

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

    async def handle_kill(self, chat_id: str) -> None:
        """Start emergency kill confirmation flow (admin-only).

        Requires admin chat_id. Stores pending confirmation state with
        monotonic timestamp. User must send CONFIRM within 30s.
        """
        admin_id = getattr(self._settings, "telegram_admin_chat_id", "")
        if chat_id != admin_id:
            await self._alerter._send("Unauthorized: /kill requires admin access")
            return

        if self._kill_switch is None:
            await self._alerter._send("Kill switch not configured")
            return

        self._pending_kill[chat_id] = time.monotonic()
        await self._alerter._send(
            "<b>KILL SWITCH</b>\n\nType CONFIRM to kill all trading within 30s"
        )

    async def handle_gonogo(self, chat_id: str) -> None:  # noqa: ARG002
        """Run go/no-go gate evaluation and send formatted report.

        Calls GoNoGoReporter.evaluate() and formats the GateReport with
        verdict emoji and per-criterion pass/fail indicators.
        """
        if self._go_no_go_reporter is None:
            await self._alerter._send("Go/no-go reporter not configured")
            return

        try:
            from finalayze.core.db import get_async_session_factory  # noqa: PLC0415

            async with get_async_session_factory()() as session:
                report = await self._go_no_go_reporter.evaluate(session)
        except Exception:
            _log.exception("telegram_gonogo_db_failed")
            try:
                report = await self._go_no_go_reporter.evaluate(None)  # type: ignore[arg-type]
            except Exception:
                await self._alerter._send("Go/no-go evaluation failed (no DB connection)")
                return

        verdict_emoji = {
            "PROCEED": "\u2705",
            "DEFER": "\u26a0\ufe0f",
            "ABORT": "\u274c",
        }
        emoji = verdict_emoji.get(str(report.verdict), "\u2753")

        lines = [f"{emoji} <b>Go/No-Go: {report.verdict}</b>"]
        lines.append(f"Sandbox days: {report.sandbox_days}")
        lines.append("")

        for c in report.criteria:
            c_emoji = "\u2705" if c.passed else "\u274c"
            lines.append(f"{c_emoji} {c.name}: {c.actual:.1f} / {c.threshold:.1f} {c.unit}")

        lines.append(f"\n{report.reason}")

        await self._alerter._send("\n".join(lines))

    async def handle_approve(self, chat_id: str, *, raw_text: str) -> None:
        """/approve <id8> — dispatch a meta-agent FIX-severity approval (58-04).

        Phase 58 D-12 + AC #12 + AP-15:
          - Parses the short8 (8 hex chars) via _APPROVE_PATTERN.
            Anchored regex; rejects extra args.
          - Invalid syntax → log meta_agent_approve_invalid_syntax, return.
          - Approver not configured → log meta_agent_approve_not_configured,
            return (graceful degradation when meta-agent isn't deployed).
          - Else dispatch as asyncio.create_task to keep webhook fast
            (D-15 fire-and-forget envelope). Task handle stored on
            self._pending_approve_tasks (RUF006) with done-callback
            cleanup.
        """
        match = _APPROVE_PATTERN.match(raw_text.strip())
        if match is None:
            _log.info(
                "meta_agent_approve_invalid_syntax",
                chat_id=chat_id,
                raw=raw_text,
            )
            return

        short8 = match.group(1).lower()

        if self._meta_agent_approver is None:
            _log.warning(
                "meta_agent_approve_not_configured",
                chat_id=chat_id,
                short8=short8,
            )
            return

        # Fire-and-forget: keep the webhook 200 OK fast (AP-15) by
        # dispatching the approver as a task. Track the handle on the
        # instance so RUF006 + lifetime are correct.
        task = asyncio.create_task(
            self._meta_agent_approver.handle_approve(short8, chat_id=chat_id),
        )
        self._pending_approve_tasks.add(task)
        task.add_done_callback(self._pending_approve_tasks.discard)

    async def _handle_kill_confirm(self, chat_id: str) -> dict[str, str]:
        """Process CONFIRM text for pending kill switch activation."""
        started_at = self._pending_kill.pop(chat_id, None)
        if started_at is None:
            return {"ok": "no_command"}

        elapsed = time.monotonic() - started_at
        if elapsed > self._KILL_CONFIRM_TIMEOUT_S:
            await self._alerter._send("Confirmation expired. Send /kill again.")
            return {"ok": "processed"}

        if self._kill_switch is None:
            await self._alerter._send("Kill switch not configured")
            return {"ok": "error"}

        try:
            result = self._kill_switch.activate(reason=f"telegram:{chat_id}")
            await self._alerter._send(
                f"<b>KILL SWITCH ACTIVATED</b>\n\n"
                f"Orders cancelled: {result.orders_cancelled}\n"
                f"Scheduler stopped: {result.scheduler_stopped}\n"
                f"Breakers escalated: {result.breakers_escalated}\n"
                f"Elapsed: {result.elapsed_seconds:.2f}s"
            )
            return {"ok": "processed"}
        except Exception:
            _log.exception("telegram_kill_confirm_failed")
            await self._alerter._send("Kill switch activation failed. Check logs.")
            return {"ok": "error"}

    def _cleanup_expired_kills(self) -> None:
        """Remove pending kill confirmations older than 60s."""
        now = time.monotonic()
        expired = [
            cid
            for cid, ts in self._pending_kill.items()
            if now - ts > self._KILL_CLEANUP_THRESHOLD_S
        ]
        for cid in expired:
            del self._pending_kill[cid]
