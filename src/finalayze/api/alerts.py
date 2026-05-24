"""Telegram alerting for trade events and system notifications (Layer 6).

Architecture after refactor:
  TelegramAlerter  — pure message formatter; calls AlertQueue.post()
  AlertQueue       — thread-safe ingest, async drain, rate-limit, batch, retry
  TelegramTransport — HTTP transport + DB persistence (in telegram_transport.py)

AlertQueue.post() is the single seam. Callers from any context (async or
APScheduler threads) call post() — the loop bridge is hidden inside AlertQueue
via loop.call_soon_threadsafe().

When bot_token is empty, TelegramAlerter is a no-op (safe default for dev/test).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from finalayze.api.telegram_transport import TelegramTransport
    from finalayze.execution.broker_base import OrderRequest, OrderResult
    from finalayze.risk.circuit_breaker import CircuitLevel

_log = structlog.get_logger()


# ── Priority & Queue Types ───────────────────────────────────────────────────


class AlertPriority(IntEnum):
    """Three-tier alert priority. Lower value = higher priority."""

    CRITICAL = 0
    IMPORTANT = 1
    INFO = 2


@dataclass(order=True)
class QueuedMessage:
    """A message waiting to be sent, ordered by (priority, timestamp)."""

    priority: AlertPriority
    timestamp: float = field(compare=True)
    text: str = field(compare=False)
    parse_mode: str = field(default="HTML", compare=False)


# ── AlertQueue ───────────────────────────────────────────────────────────────


class AlertQueue:
    """Thread-safe message ingest with async drain, rate-limiting, batching, retry.

    The single seam between callers (any context) and TelegramTransport (async).
    post() is safe to call from APScheduler threads, async handlers, or tests.

    Rate limit: 20 messages per 60-second sliding window.
    Retry: one retry after 5s on failure, then drop.
    Batching: 5+ IMPORTANT messages in queue → digest.
    """

    _RATE_LIMIT_PER_MINUTE = 20
    _RATE_WINDOW_SECONDS = 60
    _BATCH_THRESHOLD = 5
    _BATCH_MAX = 10
    _RETRY_DELAY = 5

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        transport: TelegramTransport,
    ) -> None:
        self._loop = loop
        self._transport = transport
        self._queue: asyncio.PriorityQueue[QueuedMessage] = asyncio.PriorityQueue()
        self._sent_timestamps: deque[float] = deque(
            maxlen=self._RATE_LIMIT_PER_MINUTE * 2,
        )
        self._drain_task: asyncio.Task[None] | None = None

    def post(self, text: str, priority: AlertPriority) -> None:
        """Thread-safe. Call from any context — sync or async.

        Uses loop.call_soon_threadsafe so the asyncio.PriorityQueue is only
        mutated from the event loop thread, never from an APScheduler thread.
        """
        msg = QueuedMessage(
            priority=priority,
            timestamp=time.monotonic(),
            text=text,
        )
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    async def start(self) -> None:
        """Start the background drain loop."""
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def stop(self) -> None:
        """Cancel the drain loop gracefully."""
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None

    def _is_rate_limited(self) -> bool:
        now = time.monotonic()
        while self._sent_timestamps and (
            now - self._sent_timestamps[0] > self._RATE_WINDOW_SECONDS
        ):
            self._sent_timestamps.popleft()
        return len(self._sent_timestamps) >= self._RATE_LIMIT_PER_MINUTE

    def _collect_batch(self, priority: AlertPriority) -> list[str]:
        collected: list[str] = []
        temp: list[QueuedMessage] = []
        while not self._queue.empty() and len(collected) < self._BATCH_MAX:
            try:
                msg = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if msg.priority == priority:
                collected.append(msg.text)
            else:
                temp.append(msg)
                break
        for msg in temp:
            self._queue.put_nowait(msg)
        return collected

    async def _drain_loop(self) -> None:
        while True:
            msg = await self._queue.get()
            while self._is_rate_limited():
                await asyncio.sleep(1)
            if (
                msg.priority == AlertPriority.IMPORTANT
                and self._queue.qsize() >= self._BATCH_THRESHOLD - 1
            ):
                batch = [msg.text, *self._collect_batch(AlertPriority.IMPORTANT)]
                if len(batch) >= self._BATCH_THRESHOLD:
                    digest = f"{len(batch)} fills executed:\n"
                    digest += "\n".join(f"- {line}" for line in batch)
                    await self._send_with_retry(digest)
                    self._sent_timestamps.append(time.monotonic())
                    continue
                for text in batch[1:]:
                    self._queue.put_nowait(
                        QueuedMessage(
                            priority=AlertPriority.IMPORTANT,
                            timestamp=time.monotonic(),
                            text=text,
                        )
                    )
            await self._send_with_retry(msg.text, msg.parse_mode, msg.priority)
            self._sent_timestamps.append(time.monotonic())

    async def _send_with_retry(
        self,
        text: str,
        parse_mode: str = "HTML",
        priority: AlertPriority = AlertPriority.INFO,
    ) -> bool:
        ok, _ = await self._transport.send(text, parse_mode=parse_mode, priority_name=priority.name)
        if not ok:
            await asyncio.sleep(self._RETRY_DELAY)
            ok, _ = await self._transport.send(
                text, parse_mode=parse_mode, priority_name=priority.name
            )
        return ok


# Backward-compat alias — existing code using TelegramMessageQueue still works.
TelegramMessageQueue = AlertQueue


# ── TelegramAlerter ──────────────────────────────────────────────────────────


class TelegramAlerter:
    """Pure message formatter for trade events and system notifications.

    Formats domain events (fills, circuit breakers, daily summaries, …) into
    Telegram HTML strings and hands them to AlertQueue.post(). Has no knowledge
    of HTTP, event loops, or thread contexts — all of that lives in AlertQueue
    and TelegramTransport.

    When bot_token is empty, all methods are silent no-ops.
    When no queue is attached, all methods are silent no-ops.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:  # noqa: ARG002
        self._enabled = bool(bot_token)
        self._queue: AlertQueue | None = None
        self._closed: bool = False

    def set_queue(self, queue: AlertQueue) -> None:
        """Attach the delivery queue. Called during bootstrap after loop starts."""
        self._queue = queue

    async def close(self) -> None:
        """Stop the queue drain loop. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._queue is not None:
            await self._queue.stop()

    # ── Public API ───────────────────────────────────────────────────────────

    def on_trade_filled(self, result: OrderResult, market_id: str, broker: str) -> None:
        price = result.fill_price if result.fill_price is not None else Decimal(0)
        currency_symbol = "₽" if "moex" in market_id else "$"
        text = (
            f"\U0001f7e2 {result.side} <b>{result.symbol}</b> "
            f"\xd7{result.quantity} @ <code>{currency_symbol}{price:.2f}</code>"
            f" ({broker} {market_id})"
        )
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_trade_rejected(self, order: OrderRequest, reason: str) -> None:
        text = f"\u26a0\ufe0f <b>{order.symbol}</b> {order.side} rejected: {reason}"
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_circuit_breaker_trip(
        self, market_id: str, level: CircuitLevel, drawdown_pct: float
    ) -> None:
        text = (
            f"\U0001f534 [{market_id.upper()}] Circuit breaker {level.upper()} "
            f"-- trading {level} (<code>{drawdown_pct * 100:.1f}%</code> daily drawdown)"
        )
        self.send_alert(text, priority=AlertPriority.CRITICAL)

    def on_circuit_breaker_reset(self, market_id: str) -> None:
        text = f"\u2705 [{market_id.upper()}] Circuit breaker reset \u2014 trading resumed"
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_daily_summary(
        self,
        market_pnl: dict[str, Decimal],
        total_equity_usd: Decimal,
        top_movers: list[tuple[str, float]] | None = None,
        total_equity_rub: Decimal | None = None,
    ) -> None:
        parts = []
        for market_id, pnl in sorted(market_pnl.items()):
            sign = "+" if pnl >= Decimal(0) else ""
            label = market_id.upper().replace("MOEX_BONDS", "BONDS")
            parts.append(f"{label} {sign}{pnl}")
        summary = " | ".join(parts)
        text = f"\U0001f4ca Daily: {summary}"
        if top_movers:
            movers_str = ", ".join(f"<b>{sym}</b> {pct:+.1f}%" for sym, pct in top_movers[:3])
            text += f"\nTop: {movers_str}"
        if total_equity_rub is not None:
            text += (
                f"\nTotal: <code>{total_equity_rub:,.0f}</code> RUB "
                f"(<code>${total_equity_usd:,.0f}</code>)"
            )
        else:
            text += f" | Equity <code>${total_equity_usd:,.0f}</code>"
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_coupon_received(self, symbol: str, amount: Decimal, currency: str = "RUB") -> None:
        text = f"\U0001f4b0 Coupon: <b>{symbol}</b> +<code>{amount:,.2f}</code> {currency}"
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_cbr_meeting(self, meeting_date: str, decision: str, key_rate: str) -> None:
        text = (
            f"\U0001f3e6 CBR meeting {meeting_date}: {decision}, key rate <code>{key_rate}</code>"
        )
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_bond_event_trade(self, symbol: str, side: str, reason: str) -> None:
        text = f"\U0001f4c5 CBR Event {side} <b>{symbol}</b>: {reason}"
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_stop_loss_triggered(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_price: Decimal,
        current_price: Decimal,
        *,
        pnl_amount: Decimal | None = None,
        pnl_pct: float | None = None,
        hold_bars: int | None = None,
        currency: str | None = None,
    ) -> None:
        cur_sym = {"RUB": "₽", "USD": "$"}.get(currency or "", "")
        pnl_amt_str = f"{cur_sym}{pnl_amount:+.2f}" if pnl_amount is not None else "—"
        pnl_pct_str = f"{pnl_pct * 100:+.2f}%" if pnl_pct is not None else "—"
        hold_str = f"{hold_bars} bars" if hold_bars is not None else "—"
        text = (
            f"\U0001f6d1 Stop-loss: <b>{symbol}</b> "
            f"entry=<code>{entry_price:.2f}</code>, "
            f"stop=<code>{stop_price:.2f}</code>, "
            f"price=<code>{current_price:.2f}</code>\n"
            f"P&amp;L: <code>{pnl_amt_str}</code> ({pnl_pct_str}) | "
            f"Hold: {hold_str}"
        )
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_signal_generated(
        self,
        symbol: str,
        market_id: str,
        side: str,
        confidence: float,
        strategy_breakdown: list[tuple[str, float]],
        position_context: str,
    ) -> None:
        emoji = "\U0001f7e2" if side == "BUY" else "\U0001f534"
        top_strats = strategy_breakdown[:3]
        remainder = len(strategy_breakdown) - 3
        strat_str = (
            " + ".join(f"{name} {conf:.2f}" for name, conf in top_strats) if top_strats else ""
        )
        if remainder > 0:
            strat_str += f" (+{remainder} more)"
        text = (
            f"{emoji} {side} <b>{symbol}</b> [{market_id}] | "
            f"{strat_str} \u2192 conf <code>{confidence:.2f}</code> "
            f"({position_context})"
        )
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_startup(self, mode: str, markets: list[str], instruments: int) -> None:
        text = f"\U0001f680 Finalayze started: {mode}, markets={markets}, {instruments} instruments"
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_shutdown(self) -> None:
        self.send_alert("\u23f9\ufe0f Finalayze stopped", priority=AlertPriority.INFO)

    def on_anomaly_detected(self, metric: str, value: float, threshold: float) -> None:
        text = (
            f"\U0001f6a8 Sandbox anomaly: <b>{metric}</b> "
            f"= <code>{value:.2f}</code> (threshold: {threshold:.2f})"
        )
        self.send_alert(text, priority=AlertPriority.CRITICAL)

    def on_go_nogo_decision(self, verdict: str, reason: str) -> None:
        emoji_map = {"PROCEED": "\u2705", "DEFER": "\u23f3", "ABORT": "\u274c"}
        emoji = emoji_map.get(verdict, "\u2753")
        text = f"{emoji} Go/No-Go: <b>{verdict}</b>\n{reason}"
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_error(self, component: str, message: str) -> None:
        text = f"\U0001f6a8 {component} error: {message}"
        self.send_alert(text, priority=AlertPriority.CRITICAL)

    def send_alert(self, message: str, *, priority: AlertPriority | None = None) -> None:
        """Enqueue a message. Thread-safe. Never raises.

        No-op when disabled (empty token) or queue not yet attached.
        """
        if not self._enabled or self._queue is None:
            return
        try:
            self._queue.post(message, priority if priority is not None else AlertPriority.INFO)
        except Exception:
            _log.exception("send_alert_failed")
