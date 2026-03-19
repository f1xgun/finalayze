# Phase 5: Integration and Telegram - Research

**Researched:** 2026-03-14
**Domain:** TradingLoop concurrent cycles, Telegram Bot API, daily P&L computation
**Confidence:** HIGH

## Summary

Phase 5 wires the bond cycle into TradingLoop as a concurrent APScheduler job, adds a priority message queue to TelegramAlerter, fixes the daily P&L summary (currently shows zero for bonds), and implements interactive Telegram bot commands via a FastAPI webhook endpoint.

The existing codebase is well-prepared: TradingLoop already accepts `bond_cycle_processor` and schedules it via CronTrigger, TelegramAlerter has 12 alert methods that use fire-and-forget `asyncio.create_task`, and `is_moex_trading_day()` handles both fixed holidays and per-year transfers. The main work is: (1) wrapping TelegramAlerter's `send_alert` with a priority queue + rate limiter, (2) adding MOEX holiday/hours gating inside the bond cycle body, (3) fixing `_daily_reset` to compute per-market P&L with bond layer separation, (4) adding the webhook endpoint and /status + /breakers commands, and (5) verifying all circuit breakers work for both equity and bond layers under concurrent operation.

**Primary recommendation:** Keep httpx for all Telegram API calls (no python-telegram-bot dependency). The existing TelegramAlerter already uses httpx; wrapping it with an asyncio.PriorityQueue + rate limiter is simpler than introducing a full bot framework. For webhook processing, a single FastAPI POST endpoint parsing raw JSON is sufficient.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- 3-tier priority queue: CRITICAL (circuit breaker, errors) > IMPORTANT (fills, stop-loss) > INFO (daily summary, coupon, CBR)
- CRITICAL alerts sent immediately; lower tiers queued with Telegram rate limiting
- During bursts: batch 5+ pending fills into single digest message
- All messages get one retry after 5s on failure (no complex backoff)
- HTML formatting for messages (Telegram parse_mode="HTML")
- /status command returns current portfolio state (positions + P&L)
- /breakers command returns circuit breaker states for all layers
- Read-only commands only -- no trading commands via Telegram
- Webhook transport: add /api/telegram/webhook endpoint to existing FastAPI app
- Auth: chat ID whitelist -- only respond to configured chat_id, reject all others silently
- Shared gRPC AsyncClient between equity and bond cycles, serialized with asyncio.Lock
- Full isolation between cycles: each runs in its own try/except
- Bond cycle frequency: configurable via bond_cycle_minutes setting (default 1440)
- Preflight checks on startup: verify gRPC connectivity, check macro data freshness, validate LayerLedger state, send startup Telegram alert
- Independent degradation: if bond preflight fails, disable bond cycle but keep equity running
- LayerLedger reconciliation runs on every startup
- Separate bond P&L line in daily summary with both currencies
- P&L computation: snapshot diff method -- store portfolio equity at start of day
- Start-of-day equity snapshots persisted in TimescaleDB
- Include top 3 movers in daily summary
- Weekly digest on Sunday evening
- Gate inside cycle body (not APScheduler trigger): bond cycle fires on schedule, first line checks is_moex_holiday()
- Both equity and bond cycles check MOEX holidays
- Gate on market hours too (10:00-18:45 MSK)
- Macro refresh runs 7 days/week regardless (no holiday gate)
- Holiday/hours skips: structlog only, no Telegram alert
- Unexpected skips (e.g., macro data missing): Telegram alert

### Claude's Discretion
- Exact priority queue implementation (asyncio.PriorityQueue vs custom)
- Telegram message templates and formatting details
- Preflight check timeout values
- Weekly digest scheduling (Sunday evening exact time)
- Async lock granularity (per-method vs per-request)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUT-01 | BondCycleProcessor integrated into TradingLoop scheduler | TradingLoop already accepts bond_cycle_processor and schedules via CronTrigger. Need: MOEX holiday gating inside `_bond_cycle()`, concurrent safety with asyncio.Lock, preflight checks, independent degradation |
| AUT-02 | MOEX trading schedule gate (skip non-trading days, respect hours) | `is_moex_trading_day()` exists in `data/moex_calendar.py`. `_is_market_open()` has MOEX hours constants. Need: gate inside `_bond_cycle()` body, equity cycle already gates via `_is_market_open()` |
| AUT-03 | All circuit breakers verified (equity + bond layers) | CircuitBreaker (equity), BondLayerBreaker, AggregateBondBreaker all exist. Need: integration test proving concurrent cycles trigger breakers correctly |
| MON-01 | Telegram bot sends trade alerts (fill, stop-loss, circuit breaker) | TelegramAlerter has all 12 methods. Need: priority queue wrapper, HTML formatting, rate limiting |
| MON-02 | Daily P&L summary fixed (currently shows zero) | `_daily_reset()` computes `equity - baseline` per market but doesn't separate bonds or use correct RUB amounts. Need: equity snapshots in TimescaleDB, bond P&L from LayerLedger, FXRateService for currency conversion, top 3 movers |
| MON-03 | Telegram priority message queue (prevent loss during circuit breaker bursts) | Current fire-and-forget via `create_task` has no queue or rate limiting. Need: asyncio.PriorityQueue with 3 tiers, batching logic for 5+ fills, 20 msg/min rate limiter |
| MON-04 | Coupon receipt alerts via Telegram | `on_coupon_received()` already exists on TelegramAlerter. Need: wire it into BondCycleProcessor coupon reinvestment path |
| MON-05 | CBR meeting alerts with impact analysis | `on_cbr_meeting()` already exists. Need: wire it into `_cbr_day_refresh()` after macro refresh completes |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| httpx | >=0.28.0 | Telegram Bot API HTTP calls | Already used by TelegramAlerter; async-native, no new dependency |
| asyncio.PriorityQueue | stdlib | Priority message queue | Lightweight, fits existing async pattern, no external dependency |
| APScheduler | >=3.10.4 | Bond cycle scheduling | Already used for all TradingLoop cycles |
| FastAPI | >=0.115.0 | Telegram webhook endpoint | Already the API framework; add one POST route |
| SQLAlchemy 2.0 async | existing | Equity snapshot persistence | Reuse MacroSnapshot DB pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| structlog | existing | Structured logging for skipped cycles | All holiday/hours skips logged here |
| pydantic | >=2.10.0 | Webhook update validation, settings | Parse Telegram Update JSON |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw httpx | python-telegram-bot v22 | Full framework with built-in rate limiter, but adds heavy dependency (30+ transitive deps), overkill when we only send messages + handle 2 commands. httpx is already in the stack |
| asyncio.PriorityQueue | asyncio.Queue + manual sorting | PriorityQueue is purpose-built; manual sorting adds complexity for no benefit |
| Webhook | Long polling | Webhook is more efficient for low-frequency bot commands; avoids polling thread |

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
├── core/
│   ├── alerts.py              # TelegramAlerter + TelegramMessageQueue (add queue)
│   ├── trading_loop.py        # Add holiday gating, preflight, concurrent safety
│   ├── bond_cycle.py          # No changes (already complete)
│   ├── models.py              # Add DailyEquitySnapshot model
│   └── telegram_bot.py        # NEW: webhook handler, /status, /breakers commands
├── api/
│   └── v1/
│       └── telegram.py        # NEW: /api/telegram/webhook POST endpoint
├── data/
│   └── moex_calendar.py       # No changes (already complete)
└── markets/
    └── fx_service.py          # No changes (already complete)
```

### Pattern 1: Priority Message Queue
**What:** Wrap TelegramAlerter with an asyncio background task that drains a PriorityQueue
**When to use:** All Telegram message sending
**Example:**
```python
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum

class AlertPriority(IntEnum):
    CRITICAL = 0   # circuit breaker, errors -- sent immediately
    IMPORTANT = 1  # fills, stop-loss
    INFO = 2       # daily summary, coupon, CBR

@dataclass(order=True)
class QueuedMessage:
    priority: int
    timestamp: float = field(compare=False, default_factory=time.monotonic)
    text: str = field(compare=False, default="")
    parse_mode: str = field(compare=False, default="HTML")

class TelegramMessageQueue:
    """Rate-limited priority queue for Telegram messages.

    Respects Telegram's 20 messages/minute per-chat limit.
    CRITICAL messages bypass the queue and send immediately.
    """

    _RATE_LIMIT_PER_MINUTE = 20
    _BATCH_THRESHOLD = 5  # batch fills when >= 5 pending

    def __init__(self, alerter: TelegramAlerter) -> None:
        self._alerter = alerter
        self._queue: asyncio.PriorityQueue[QueuedMessage] = asyncio.PriorityQueue()
        self._sent_timestamps: list[float] = []
        self._drain_task: asyncio.Task[None] | None = None

    async def enqueue(self, text: str, priority: AlertPriority) -> None:
        if priority == AlertPriority.CRITICAL:
            await self._alerter._send(text)  # bypass queue
            return
        await self._queue.put(QueuedMessage(priority=priority.value, text=text))

    async def _drain_loop(self) -> None:
        while True:
            msg = await self._queue.get()
            await self._wait_for_rate_limit()
            await self._alerter._send(msg.text)
            self._record_send()
```

### Pattern 2: Bond Cycle Holiday Gating (Inside Body)
**What:** Check MOEX holidays and market hours as the first operation inside `_bond_cycle()`, not in the APScheduler trigger
**When to use:** Bond cycle and equity cycle MOEX instrument processing
**Example:**
```python
def _bond_cycle(self) -> None:
    """Daily bond trading cycle. Gates on MOEX holidays + hours."""
    if self._bond_processor is None:
        return
    now = self._now()
    from finalayze.data.moex_calendar import is_moex_trading_day
    if not is_moex_trading_day(now.date()):
        _log.info("bond_cycle_skipped_holiday", date=str(now.date()))
        return  # structlog only, no Telegram alert
    if not self._is_market_open("moex", now):
        _log.info("bond_cycle_skipped_hours", time=str(now.time()))
        return
    # ... proceed with run_cycle()
```

### Pattern 3: Telegram Webhook Endpoint
**What:** Single FastAPI POST endpoint that parses Telegram Update JSON, validates chat_id, dispatches commands
**When to use:** Interactive bot commands (/status, /breakers)
**Example:**
```python
from fastapi import APIRouter, Request, HTTPException

router = APIRouter(tags=["telegram"])

@router.post("/api/telegram/webhook")
async def telegram_webhook(request: Request) -> dict[str, str]:
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if secret != settings.telegram_webhook_secret:
        raise HTTPException(status_code=403)
    update = await request.json()
    message = update.get("message", {})
    chat_id = str(message.get("chat", {}).get("id", ""))
    if chat_id not in settings.telegram_allowed_chat_ids:
        return {"ok": "ignored"}  # silent reject
    text = message.get("text", "")
    if text == "/status":
        await handle_status(chat_id)
    elif text == "/breakers":
        await handle_breakers(chat_id)
    return {"ok": "true"}
```

### Pattern 4: Daily P&L with Bond Separation
**What:** Compute P&L per market (US, MOEX equity, MOEX bonds) using start-of-day equity snapshots
**When to use:** `_daily_reset()` method
**Example:**
```python
# Start-of-day snapshot (stored in _daily_reset at end of day for NEXT day)
# and persisted to TimescaleDB via DailyEquitySnapshot model
market_pnl = {
    "us": us_equity - us_baseline,
    "moex_equity": moex_eq_equity - moex_eq_baseline,
    "moex_bonds": sum(ledger.current_equity for ledger in bond_ledgers.values()) - bond_baseline,
}
# Convert to display currencies using FXRateService
total_rub = fx.to_currency(total_usd, "RUB")
```

### Anti-Patterns to Avoid
- **Fire-and-forget without queue:** Current `asyncio.create_task` pattern loses messages during bursts. Always route through the priority queue.
- **Gating in APScheduler trigger:** If you use `CronTrigger(day_of_week='mon-fri')`, you lose control over transferred holidays. Gate inside the cycle body.
- **Sharing mutable state between cycles without lock:** Equity and bond cycles run in different APScheduler threads. The shared gRPC AsyncClient needs `asyncio.Lock` serialization.
- **Creating new httpx.AsyncClient per message:** TelegramAlerter currently creates a new client per `_send()` call. Refactor to use a persistent client (connection pooling).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rate limiting | Custom token bucket | Simple deque of send timestamps + 60s sliding window | Telegram's limit is 20/min per chat -- a timestamp deque is trivial and correct |
| Telegram Update parsing | Full Update model with 30+ fields | Extract only `message.chat.id` and `message.text` from raw JSON | We handle exactly 2 commands; full model is unnecessary |
| Weekly P&L aggregation | Custom rolling window | TimescaleDB `time_bucket` query over DailyEquitySnapshot | TimescaleDB is already in the stack; its time-series aggregation is purpose-built |
| gRPC channel management | Custom connection pool | Single shared AsyncClient with asyncio.Lock | T-Invest gRPC channels are thread-safe for reads; lock only needed for write+read contention |

**Key insight:** The system already has 90% of the pieces (TelegramAlerter, BondCycleProcessor, holiday calendar, FXRateService, circuit breakers). This phase is primarily integration and wiring, not building new subsystems.

## Common Pitfalls

### Pitfall 1: Telegram Rate Limit Exceeded (HTTP 429)
**What goes wrong:** Sending >20 messages/minute to same chat during circuit breaker cascade
**Why it happens:** Circuit breaker trips can generate 5-10 alerts in seconds (per-market trip + per-position liquidation)
**How to avoid:** Priority queue with rate limiter. CRITICAL messages sent immediately (they're rare: 1-2 per event). Lower priority messages queued and drained at <=20/min. Batch 5+ pending fills into digest.
**Warning signs:** HTTP 429 responses with `retry_after` header in logs

### Pitfall 2: Bond P&L Shows Zero
**What goes wrong:** `_daily_reset()` computes `equity - baseline` but bond equity isn't tracked in `_baseline_equities`
**Why it happens:** Bond positions live in LayerLedger, not in broker portfolio equity. The current code only queries broker.get_portfolio() for each market.
**How to avoid:** Separate bond equity computation: sum all LayerLedger.current_equity values. Store baseline separately for bonds.
**Warning signs:** Bond P&L always showing 0 in daily summary despite active positions

### Pitfall 3: gRPC Contention Between Concurrent Cycles
**What goes wrong:** Equity strategy_cycle and bond_cycle both call T-Invest API simultaneously, causing gRPC errors
**Why it happens:** APScheduler runs jobs in different threads, both sharing the same AsyncClient
**How to avoid:** `asyncio.Lock` around all gRPC calls. The lock is per-method (not per-request) since individual API calls are fast (<100ms).
**Warning signs:** gRPC UNAVAILABLE or RESOURCE_EXHAUSTED errors in logs

### Pitfall 4: Webhook Secret Token Not Validated
**What goes wrong:** Anyone can send POST requests to the webhook endpoint, injecting fake commands
**Why it happens:** Forgetting to validate `X-Telegram-Bot-Api-Secret-Token` header
**How to avoid:** Always check the secret token (set via `bot.setWebhook(secret_token=...)`) AND validate chat_id against whitelist. Double validation.
**Warning signs:** Unexpected command executions in logs from unknown chat IDs

### Pitfall 5: Equity Snapshot Not Persisted on Restart
**What goes wrong:** After system restart, P&L shows huge jump because baseline is reset to current equity
**Why it happens:** `_baseline_equities` is in-memory dict, lost on restart
**How to avoid:** Persist start-of-day snapshots to TimescaleDB. On startup, load latest snapshot as baseline. If no snapshot exists for today, take current equity as baseline.
**Warning signs:** Anomalous P&L spikes on first day after restart

### Pitfall 6: httpx.AsyncClient Per-Message Overhead
**What goes wrong:** Creating a new httpx.AsyncClient for every message causes connection overhead and potential resource leaks
**Why it happens:** Current `_send()` uses `async with httpx.AsyncClient() as client:` which creates/destroys per call
**How to avoid:** Use a persistent httpx.AsyncClient stored as instance variable, closed on shutdown
**Warning signs:** Slow message delivery, "too many open files" errors under load

## Code Examples

### Existing TelegramAlerter._send() (to be refactored)
```python
# Source: src/finalayze/core/alerts.py:179-193
# Current: creates new client per message, no rate limiting, no HTML
async def _send(self, text: str) -> None:
    if not self._token:
        return
    url = f"{_TELEGRAM_API_BASE}{self._token}{_SEND_MESSAGE_PATH}"
    payload = {"chat_id": self._chat_id, "text": text}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload, timeout=10)
    except Exception:
        _log.exception("TelegramAlerter failed to send message")
```

### Refactored _send() with HTML and persistent client
```python
async def _send(self, text: str, parse_mode: str = "HTML") -> bool:
    if not self._token:
        return True
    url = f"{_TELEGRAM_API_BASE}{self._token}{_SEND_MESSAGE_PATH}"
    payload = {"chat_id": self._chat_id, "text": text, "parse_mode": parse_mode}
    try:
        resp = await self._client.post(url, json=payload, timeout=10)
        if resp.status_code == 429:
            retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
            _log.warning("telegram_rate_limited", retry_after=retry_after)
            return False  # caller retries
        return True
    except Exception:
        _log.exception("TelegramAlerter failed to send message")
        return False
```

### Existing _bond_cycle() (to be enhanced with holiday gating)
```python
# Source: src/finalayze/core/trading_loop.py:320-329
def _bond_cycle(self) -> None:
    """Daily bond trading cycle across all layers. SYNC."""
    if self._bond_processor is None:
        return
    _log.info("bond_cycle_start")
    try:
        result = self._bond_processor.run_cycle()
        _log.info("bond_cycle_complete", **result.to_log_dict())
    except Exception:
        _log.exception("bond_cycle_failed")
```

### Existing _daily_reset() P&L computation (to be fixed)
```python
# Source: src/finalayze/core/trading_loop.py:1037-1079
# Current: only computes equity - baseline per market, no bond separation,
# passes total_equity as-is (no RUB conversion)
def _daily_reset(self) -> None:
    market_pnl: dict[str, Decimal] = {}
    for market_id, cb in self._circuit_breakers.items():
        portfolio = broker.get_portfolio()
        equity = portfolio.equity
        baseline = self._baseline_equities.get(market_id, equity)
        market_pnl[market_id] = equity - baseline
    self._alerter.on_daily_summary(market_pnl, total_equity)
```

### Existing _is_market_open() with MOEX holiday check
```python
# Source: src/finalayze/core/trading_loop.py:571-597
# Already handles MOEX holidays + hours. Reuse for bond cycle gating.
def _is_market_open(self, market_id: str, dt: datetime) -> bool:
    if dt.weekday() >= _WEEKEND_WEEKDAY:
        return False
    if market_id == "moex":
        from finalayze.data.moex_calendar import is_moex_trading_day
        if not is_moex_trading_day(dt.date()):
            return False
    # ... hours check follows
```

### DailyEquitySnapshot Model (new)
```python
# To be added to src/finalayze/core/models.py
class DailyEquitySnapshot(Base):
    """Start-of-day equity snapshots for P&L computation.

    Persisted to TimescaleDB. Used to survive restarts and for weekly digest.
    """
    __tablename__ = "daily_equity_snapshots"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    equity: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fire-and-forget create_task | Priority queue with rate limiter | This phase | Prevents message loss during bursts |
| Single market_pnl dict | Separate US/MOEX equity/MOEX bonds P&L | This phase | Bond P&L actually shows values |
| In-memory baseline only | TimescaleDB-persisted equity snapshots | This phase | Survives restarts, enables weekly digest |
| No bot interaction | Webhook with /status and /breakers | This phase | Read-only monitoring without opening dashboard |

**Deprecated/outdated:**
- Telegram Bot API < 6.1: No webhook secret token support. Current API (7.0+) uses token-bucket rate limiting with `retry_after` in 429 responses.

## Open Questions

1. **Webhook registration**
   - What we know: Need to call `bot.setWebhook(url=..., secret_token=...)` once during deployment
   - What's unclear: Should this be done in FastAPI lifespan or as a separate CLI command?
   - Recommendation: Add a `scripts/set_telegram_webhook.py` script for manual registration. Don't auto-register on every startup (avoids rate limit on setWebhook).

2. **Bond equity for P&L baseline**
   - What we know: LayerLedger tracks `current_equity` per layer. Need to aggregate across 4 layers.
   - What's unclear: Should bond equity be fetched from broker (live prices) or computed from ledger (entry prices)?
   - Recommendation: Use ledger values (consistent with how bond cycle operates). Live prices would require extra gRPC calls.

3. **Weekly digest exact timing**
   - What we know: Sunday evening (user decision)
   - What's unclear: Exact hour. Sunday 19:00 MSK (16:00 UTC) seems reasonable.
   - Recommendation: Default to `weekly_digest_hour_utc=16` (configurable via Settings)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + pytest-asyncio |
| Config file | pyproject.toml (`[tool.pytest.ini_options]`) |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest --cov -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUT-01 | Bond cycle runs in TradingLoop | unit | `uv run pytest tests/unit/test_trading_loop_bonds.py -x` | Exists (minimal) |
| AUT-02 | MOEX holiday gating skips bond cycle | unit | `uv run pytest tests/unit/test_trading_loop_holidays.py -x` | Exists (equity only) |
| AUT-03 | Circuit breakers fire for both layers | integration | `uv run pytest tests/integration/test_circuit_breaker_integration.py -x` | Exists (equity only) |
| MON-01 | Telegram alerts sent for fills/stops/breakers | unit | `uv run pytest tests/unit/test_telegram_alerter.py -x` | Exists |
| MON-02 | Daily P&L shows correct RUB amounts | unit | `uv run pytest tests/unit/test_daily_pnl.py -x` | Does not exist -- Wave 0 |
| MON-03 | Priority queue batches fills, respects rate limit | unit | `uv run pytest tests/unit/test_telegram_queue.py -x` | Does not exist -- Wave 0 |
| MON-04 | Coupon receipt fires Telegram alert | unit | `uv run pytest tests/unit/test_bond_cycle.py::test_coupon_alert -x` | Does not exist -- Wave 0 |
| MON-05 | CBR meeting fires Telegram alert | unit | `uv run pytest tests/unit/test_trading_loop_bonds.py::test_cbr_alert -x` | Does not exist -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x -q --timeout=30`
- **Per wave merge:** `uv run pytest --cov -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_telegram_queue.py` -- covers MON-03 (priority queue, rate limiting, batching)
- [ ] `tests/unit/test_daily_pnl.py` -- covers MON-02 (P&L computation, currency conversion, bond separation)
- [ ] `tests/unit/test_telegram_webhook.py` -- covers webhook endpoint, command dispatch, auth
- [ ] `tests/unit/test_preflight.py` -- covers AUT-01 preflight checks, independent degradation
- [ ] Extend `test_trading_loop_bonds.py` -- covers AUT-01 bond cycle integration, MON-04, MON-05
- [ ] Extend `test_trading_loop_holidays.py` -- covers AUT-02 bond cycle holiday gating
- [ ] Extend `test_circuit_breaker_integration.py` -- covers AUT-03 bond layer breakers

## Sources

### Primary (HIGH confidence)
- Source code: `src/finalayze/core/alerts.py` -- current TelegramAlerter implementation
- Source code: `src/finalayze/core/trading_loop.py` -- current TradingLoop with bond cycle scheduling
- Source code: `src/finalayze/core/bond_cycle.py` -- BondCycleProcessor.run_cycle()
- Source code: `src/finalayze/data/moex_calendar.py` -- is_moex_trading_day()
- Source code: `config/settings.py` -- current Settings with telegram_bot_token, telegram_chat_id
- Source code: `src/finalayze/core/models.py` -- existing SQLAlchemy models including PortfolioSnapshot

### Secondary (MEDIUM confidence)
- [Telegram Bots FAQ](https://core.telegram.org/bots/faq) -- 20 messages/minute per group chat, 30 messages/second global
- [Telegram Bot API webhook security](https://core.telegram.org/bots/api#setwebhook) -- X-Telegram-Bot-Api-Secret-Token header (API 6.1+)
- Telegram Bot API 7.0+ uses token-bucket with `retry_after` in 429 responses

### Tertiary (LOW confidence)
- None -- all findings verified against source code and official Telegram docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all libraries already in pyproject.toml
- Architecture: HIGH -- patterns derived from existing codebase (MacroCacheService persistence, APScheduler scheduling, TelegramAlerter structure)
- Pitfalls: HIGH -- identified from actual code analysis (fire-and-forget pattern, missing bond P&L, in-memory baselines)

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable -- Telegram Bot API rarely changes, codebase patterns are established)
