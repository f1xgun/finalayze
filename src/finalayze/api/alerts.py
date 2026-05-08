"""Telegram alerting for trade events and system notifications (Layer 6).

TelegramAlerter sends messages via Telegram Bot API with:
  - 3-tier priority queue (CRITICAL bypass, IMPORTANT batching, INFO background)
  - Sliding-window rate limiting (20 msg/min)
  - Persistent httpx.AsyncClient (no per-message creation)
  - HTML parse_mode on all messages
  - One retry on failure after 5s delay

When ``bot_token`` is empty, all methods are no-ops (safe default for dev/test).
HTTP errors are caught and logged -- they never propagate to the trading loop.

Moved from core/ to api/ in Phase 22 (dependency layer cleanup).
See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import IntEnum
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from finalayze.execution.broker_base import OrderRequest, OrderResult
    from finalayze.orchestration.db_persistence import TradingPersistence
    from finalayze.risk.circuit_breaker import CircuitLevel

_TELEGRAM_API_BASE = "https://api.telegram.org/bot"
_SEND_MESSAGE_PATH = "/sendMessage"

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


class TelegramMessageQueue:
    """Priority message queue with rate limiting, batching, and retry.

    - CRITICAL: bypass queue, send immediately with retry
    - IMPORTANT: queued, batched if 5+ pending
    - INFO: queued, drained in order

    Rate limit: 20 messages per 60-second sliding window.
    Retry: one retry after 5s on failure, then drop.
    """

    _RATE_LIMIT_PER_MINUTE = 20
    _RATE_WINDOW_SECONDS = 60
    _BATCH_THRESHOLD = 5
    _BATCH_MAX = 10
    _RETRY_DELAY = 5

    def __init__(self, alerter: TelegramAlerter) -> None:
        self._alerter = alerter
        self._queue: asyncio.PriorityQueue[QueuedMessage] = asyncio.PriorityQueue()
        self._sent_timestamps: deque[float] = deque(
            maxlen=self._RATE_LIMIT_PER_MINUTE * 2,
        )
        self._drain_task: asyncio.Task[None] | None = None

    async def enqueue(self, text: str, priority: AlertPriority) -> None:
        """Add a message to the queue, or send immediately if CRITICAL."""
        if priority == AlertPriority.CRITICAL:
            await self._send_with_retry(text)
            return
        msg = QueuedMessage(
            priority=priority,
            timestamp=time.monotonic(),
            text=text,
        )
        await self._queue.put(msg)

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
        """Check if we've hit 20 messages in the last 60s."""
        now = time.monotonic()
        # Purge timestamps older than the window
        while self._sent_timestamps and (
            now - self._sent_timestamps[0] > self._RATE_WINDOW_SECONDS
        ):
            self._sent_timestamps.popleft()
        return len(self._sent_timestamps) >= self._RATE_LIMIT_PER_MINUTE

    def _collect_batch(self, priority: AlertPriority) -> list[str]:
        """Collect consecutive messages of the given priority for batching."""
        collected: list[str] = []
        # Peek at queue items without blocking
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
        # Put back non-matching messages
        for msg in temp:
            self._queue.put_nowait(msg)
        return collected

    async def _drain_loop(self) -> None:
        """Infinite loop: dequeue, rate-limit, batch, send."""
        while True:
            msg = await self._queue.get()
            # Wait for rate limit to clear
            while self._is_rate_limited():
                await asyncio.sleep(1)
            # Check for batching on IMPORTANT
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
                # Not enough for batch -- send individually, put rest back
                for text in batch[1:]:
                    self._queue.put_nowait(
                        QueuedMessage(
                            priority=AlertPriority.IMPORTANT,
                            timestamp=time.monotonic(),
                            text=text,
                        )
                    )
            await self._send_with_retry(msg.text, msg.parse_mode)
            self._sent_timestamps.append(time.monotonic())

    async def _send_with_retry(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send via alerter._send, retry once on failure after 5s."""
        ok, _ = await self._alerter._send(text, parse_mode=parse_mode)
        if not ok:
            await asyncio.sleep(self._RETRY_DELAY)
            ok, _ = await self._alerter._send(text, parse_mode=parse_mode)
        return ok


class TelegramAlerter:
    """Sends Telegram messages for trade fills, rejections, circuit breaker events,
    daily summaries, and errors.

    Features:
      - Persistent ``httpx.AsyncClient`` (reused across messages)
      - HTML parse_mode on all messages
      - Optional ``TelegramMessageQueue`` integration for rate limiting / batching
      - Backward compatible: works without queue (fire-and-forget via create_task)

    When ``bot_token`` is an empty string, all methods return immediately
    without any network call (safe default for debug and test modes).
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        *,
        persistence: TradingPersistence | None = None,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._client: httpx.AsyncClient = httpx.AsyncClient(timeout=10)
        self._queue: TelegramMessageQueue | None = None
        self._closed: bool = False
        # Phase 57-02 ALRT-03: optional fire-and-forget DB persistence wrapping
        # every outbound message. Backwards-compatible: existing callers that
        # construct ``TelegramAlerter(bot_token, chat_id)`` continue to work.
        self._persistence: TradingPersistence | None = persistence
        # Cache (alert_id -> timestamp) so _update_alert_status_async hits the
        # composite (timestamp, id) PK exactly. Pop on update.
        self._last_alert_ts: dict[uuid.UUID, datetime] = {}
        # Main uvicorn event loop — set via set_event_loop() in lifespan so that
        # sync callers (APScheduler threads) can bridge via run_coroutine_threadsafe
        # instead of falling back to the less reliable _send_sync path.
        self._main_loop: asyncio.AbstractEventLoop | None = None

    def set_queue(self, queue: TelegramMessageQueue) -> None:
        """Attach a message queue for rate limiting and batching."""
        self._queue = queue

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the main event loop so sync callers can use run_coroutine_threadsafe."""
        self._main_loop = loop

    async def close(self) -> None:
        """Shut down persistent httpx client and queue.

        Idempotent: safe to call multiple times. The second call is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        if self._queue is not None:
            await self._queue.stop()
        await self._client.aclose()

    # ── Public API ───────────────────────────────────────────────────────────

    def on_trade_filled(self, result: OrderResult, market_id: str, broker: str) -> None:
        """Alert on a successful order fill.

        Example: ``BUY SBER x10 @ ₽280.50 (moex sandbox)``
        """
        price = result.fill_price if result.fill_price is not None else Decimal(0)
        currency_symbol = "₽" if "moex" in market_id else "$"
        text = (
            f"\U0001f7e2 {result.side} <b>{result.symbol}</b> "
            f"\xd7{result.quantity} @ <code>{currency_symbol}{price:.2f}</code>"
            f" ({broker} {market_id})"
        )
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_trade_rejected(self, order: OrderRequest, reason: str) -> None:
        """Alert on an order rejection.

        Example: ``AAPL BUY rejected: insufficient funds``
        """
        text = f"\u26a0\ufe0f <b>{order.symbol}</b> {order.side} rejected: {reason}"
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_circuit_breaker_trip(
        self, market_id: str, level: CircuitLevel, drawdown_pct: float
    ) -> None:
        """Alert on a circuit breaker level change.

        Example: ``[US] Circuit breaker HALTED -- trading halted (-10.3% daily)``
        """
        text = (
            f"\U0001f534 [{market_id.upper()}] Circuit breaker {level.upper()} "
            f"-- trading {level} (<code>{drawdown_pct * 100:.1f}%</code> daily drawdown)"
        )
        self.send_alert(text, priority=AlertPriority.CRITICAL)

    def on_circuit_breaker_reset(self, market_id: str) -> None:
        """Alert on circuit breaker reset.

        Example: ``[US] Circuit breaker reset -- trading resumed``
        """
        text = f"\u2705 [{market_id.upper()}] Circuit breaker reset \u2014 trading resumed"
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_daily_summary(
        self,
        market_pnl: dict[str, Decimal],
        total_equity_usd: Decimal,
        top_movers: list[tuple[str, float]] | None = None,
        total_equity_rub: Decimal | None = None,
    ) -> None:
        """Alert with daily P&L summary.

        Example::

            Daily: US +$342 | MOEX +1,200 | BONDS +500
            Top: SBER +2.1%, GAZP -0.8%, SU26244 +0.3%
            Total: 2.5M RUB ($28,400)
        """
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

    def on_coupon_received(
        self,
        symbol: str,
        amount: Decimal,
        currency: str = "RUB",
    ) -> None:
        """Alert on a coupon payment received.

        Example: ``Coupon: SU26244RMFS2 +3,250.00 RUB``
        """
        text = f"\U0001f4b0 Coupon: <b>{symbol}</b> +<code>{amount:,.2f}</code> {currency}"
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_cbr_meeting(
        self,
        meeting_date: str,
        decision: str,
        key_rate: str,
    ) -> None:
        """Alert on a CBR rate decision.

        Example: ``CBR meeting 2026-03-20: HOLD, key rate 21.00%``
        """
        text = (
            f"\U0001f3e6 CBR meeting {meeting_date}: {decision}, key rate <code>{key_rate}</code>"
        )
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_bond_event_trade(
        self,
        symbol: str,
        side: str,
        reason: str,
    ) -> None:
        """Alert on a CBR event-driven bond trade.

        Example: ``CBR Event BUY SU26244RMFS2: 5d before meeting, gap=-0.25``
        """
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
        """Alert on a stop-loss trigger with enriched context (ALRT-01, D-09).

        None fields render as '—' (Phase 54 D-03 'null is the signal').

        Example::

            🛑 Stop-loss: SBER entry=280.50, stop=266.48, price=265.00
            P&L: ₽-80.50 (-8.05%) | Hold: 12 bars
        """
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
        """Alert on a new signal with strategy attribution (ALRT-02, D-14).

        Args:
            symbol: Instrument ticker (e.g., 'SBER').
            market_id: Market identifier (e.g., 'moex').
            side: 'BUY' or 'SELL'.
            confidence: Combined signal confidence in [0, 1].
            strategy_breakdown: ``[(name, confidence)]`` sorted desc by
                contribution. Truncated to top-3 + ``(+N more)`` per D-14.
            position_context: One of 'NEW', 'ADD', or 'FLIP' per D-11.

        Example::

            🟢 BUY SBER [moex] | momentum 0.72 + macd 0.64 + rsi 0.51 → conf 0.58 (NEW)
        """
        emoji = "\U0001f7e2" if side == "BUY" else "\U0001f534"  # green / red
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
        """Alert on system startup.

        Example: ``Finalayze started: sandbox, markets=[moex], 23 instruments``
        """
        text = f"\U0001f680 Finalayze started: {mode}, markets={markets}, {instruments} instruments"
        self.send_alert(text, priority=AlertPriority.INFO)

    def on_shutdown(self) -> None:
        """Alert on system shutdown."""
        self.send_alert("\u23f9\ufe0f Finalayze stopped", priority=AlertPriority.INFO)

    def on_anomaly_detected(self, metric: str, value: float, threshold: float) -> None:
        """Alert on sandbox anomaly detection."""
        text = (
            f"\U0001f6a8 Sandbox anomaly: <b>{metric}</b> "
            f"= <code>{value:.2f}</code> (threshold: {threshold:.2f})"
        )
        self.send_alert(text, priority=AlertPriority.CRITICAL)

    def on_go_nogo_decision(self, verdict: str, reason: str) -> None:
        """Alert on go/no-go gate evaluation result."""
        emoji_map = {"PROCEED": "\u2705", "DEFER": "\u23f3", "ABORT": "\u274c"}
        emoji = emoji_map.get(verdict, "\u2753")
        text = f"{emoji} Go/No-Go: <b>{verdict}</b>\n{reason}"
        self.send_alert(text, priority=AlertPriority.IMPORTANT)

    def on_error(self, component: str, message: str) -> None:
        """Alert on system errors.

        Example: ``TinkoffFetcher error: gRPC timeout``
        """
        text = f"\U0001f6a8 {component} error: {message}"
        self.send_alert(text, priority=AlertPriority.CRITICAL)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _persist_alert_before_send(
        self,
        text: str,
        alert_type: str,
        priority: AlertPriority,
        symbol: str | None,
        market_id: str | None,
        parent_id: uuid.UUID | None,
        alert_metadata: dict[str, object] | None,
    ) -> uuid.UUID | None:
        """Insert alerts row with ``delivery_status='queued'`` BEFORE httpx.post.

        Returns ``alert_id`` on success, ``None`` if persistence is unavailable
        or the call failed. NEVER raises — persistence failures must NOT block
        the Telegram send path (PERSIST-05 envelope).
        """
        if self._persistence is None:
            return None
        alert_id = uuid.uuid4()
        timestamp = datetime.now(tz=UTC)
        try:
            self._persistence.persist_alert(
                alert_id,
                timestamp,
                alert_type,
                priority.name,
                text,
                symbol=symbol,
                market_id=market_id,
                parent_id=parent_id,
                delivery_status="queued",
                alert_metadata=alert_metadata,
            )
            self._last_alert_ts[alert_id] = timestamp
        except Exception:
            _log.warning("alert_persist_before_send_failed", exc_info=True)
            return None
        return alert_id

    def _update_alert_status(
        self,
        alert_id: uuid.UUID | None,
        delivery_status: str,
    ) -> None:
        """Update ``delivery_status`` after the httpx response. NEVER raises."""
        if self._persistence is None or alert_id is None:
            return
        ts = self._last_alert_ts.pop(alert_id, None)
        if ts is None:
            return
        try:
            self._persistence.update_alert_status(alert_id, ts, delivery_status)
        except Exception:
            _log.warning("alert_status_update_failed", exc_info=True)

    async def _send(
        self,
        text: str,
        *,
        parse_mode: str = "HTML",
        alert_type: str = "generic",
        priority: AlertPriority = AlertPriority.INFO,
        symbol: str | None = None,
        market_id: str | None = None,
        parent_id: uuid.UUID | None = None,
        alert_metadata: dict[str, object] | None = None,
    ) -> tuple[bool, uuid.UUID | None]:
        """Async POST a message to the Telegram Bot API.

        Uses persistent ``self._client``. Returns ``(ok, alert_id)`` so the
        anomaly path can thread parent_id from raw -> llm follow-up. Persists
        an alerts row BEFORE the httpx.post and updates delivery_status AFTER
        (Phase 57-02 ALRT-03; PERSIST-05 envelope — persistence failures never
        block the send).
        """
        alert_id = self._persist_alert_before_send(
            text,
            alert_type,
            priority,
            symbol,
            market_id,
            parent_id,
            alert_metadata,
        )

        if not self._token:
            self._update_alert_status(alert_id, "sent")
            return (True, alert_id)

        url = f"{_TELEGRAM_API_BASE}{self._token}{_SEND_MESSAGE_PATH}"
        payload: dict[str, str] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        try:
            resp = await self._client.post(url, json=payload)
            if resp.status_code == 429:  # noqa: PLR2004
                retry_after = resp.json().get("parameters", {}).get("retry_after", 30)
                _log.warning("Telegram rate limited", retry_after=retry_after)
                self._update_alert_status(alert_id, "failed")
                return (False, alert_id)
            self._update_alert_status(alert_id, "sent")
            return (True, alert_id)
        except Exception:
            _log.exception("TelegramAlerter failed to send message")
            self._update_alert_status(alert_id, "failed")
            return (False, alert_id)

    def _send_sync(
        self,
        text: str,
        *,
        parse_mode: str = "HTML",
        alert_type: str = "generic",
        priority: AlertPriority = AlertPriority.INFO,
        symbol: str | None = None,
        market_id: str | None = None,
        parent_id: uuid.UUID | None = None,
        alert_metadata: dict[str, object] | None = None,
    ) -> tuple[bool, uuid.UUID | None]:
        """Synchronous POST to Telegram Bot API.

        Used when called from non-async context (APScheduler threads).
        Creates a short-lived httpx.Client per call to avoid event loop issues.
        Same persist-before/update-after envelope as ``_send``.
        """
        alert_id = self._persist_alert_before_send(
            text,
            alert_type,
            priority,
            symbol,
            market_id,
            parent_id,
            alert_metadata,
        )

        if not self._token:
            self._update_alert_status(alert_id, "sent")
            return (True, alert_id)

        url = f"{_TELEGRAM_API_BASE}{self._token}{_SEND_MESSAGE_PATH}"
        payload = {"chat_id": self._chat_id, "text": text, "parse_mode": parse_mode}
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 429:  # noqa: PLR2004
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 30)
                    _log.warning("Telegram rate limited", retry_after=retry_after)
                    self._update_alert_status(alert_id, "failed")
                    return (False, alert_id)
                self._update_alert_status(alert_id, "sent")
                return (True, alert_id)
        except Exception:
            _log.exception("TelegramAlerter sync send failed")
            self._update_alert_status(alert_id, "failed")
            return (False, alert_id)

    def send_alert(
        self,
        message: str,
        *,
        priority: AlertPriority | None = None,
    ) -> None:
        """Send alert safely from any thread context.

        From async context (running event loop): uses create_task with async _send.
        From sync context with known main loop: uses run_coroutine_threadsafe so
        the persistent async httpx.AsyncClient is always used.
        From sync context without known loop: falls back to synchronous httpx.Client.

        Exceptions are always suppressed -- alerts must never crash the caller.
        """
        if not self._token:
            return
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # Async context — use create_task
                if self._queue is not None and priority is not None:
                    _task = loop.create_task(
                        self._queue.enqueue(message, priority),
                    )
                else:
                    _task = loop.create_task(self._send(message))  # type: ignore[arg-type]
            elif self._main_loop is not None and self._main_loop.is_running():
                # Sync context (APScheduler thread) with known main loop —
                # bridge to the uvicorn loop so the async client is reused.
                asyncio.run_coroutine_threadsafe(self._send(message), self._main_loop)
            else:
                # Sync context without a known main loop — use sync httpx.
                self._send_sync(message)
        except Exception:
            _log.exception("TelegramAlerter send_alert failed")
