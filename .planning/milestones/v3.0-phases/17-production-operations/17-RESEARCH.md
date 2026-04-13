# Phase 17: Production Operations - Research

**Researched:** 2026-03-22
**Domain:** Production operations (kill switch, health monitoring, alert taxonomy, Telegram commands)
**Confidence:** HIGH

## Summary

Phase 17 extends the existing Telegram bot handler and alert infrastructure to deliver four production-operations features: a kill switch that cancels orders + stops scheduler + escalates breakers within 30 seconds, a health monitor with 5-minute heartbeat and 2-miss alerting, 3-tier alert taxonomy wiring (already partially implemented), and two new Telegram commands (/kill with confirmation, /gonogo).

The codebase is well-prepared for this phase. `TelegramBotHandler` already handles `/stop`, `/status`, and `/breakers` commands with webhook-based dispatch. `TelegramAlerter` already has `AlertPriority.CRITICAL` bypassing the queue for immediate delivery. `GoNoGoReporter` exists with full `evaluate()` method. The main work is: (1) a new `KillSwitch` class that orchestrates multi-step shutdown, (2) a `HealthMonitor` with APScheduler heartbeat, (3) extending `TelegramBotHandler` with `/kill` (confirmation flow) and `/gonogo`, and (4) a REST `/health/production` endpoint.

**Primary recommendation:** Build `KillSwitch` as a standalone orchestrator class in `core/kill_switch.py`, `HealthMonitor` in `monitoring/health_monitor.py`, extend existing `TelegramBotHandler` with new commands, and add REST endpoint to `api/v1/system.py`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `KillSwitch` class with `activate()` method -- orchestrates: cancel all broker orders -> stop TradingLoop scheduler -> escalate CircuitBreakers to LIQUIDATE -> send Telegram CRITICAL alert
- Order cancellation via broker -- for each active market (NOTE: `cancel_all_orders()` does NOT exist; must use `get_open_orders()` + `cancel_order()` per order)
- Three triggers: Telegram `/kill` command, REST endpoint, programmatic `KillSwitch.activate()`
- Recovery requires full restart -- kill switch sets persistent flag checked by `main.py` on startup
- `HealthMonitor` class with APScheduler job every 5 minutes
- Checks: broker connectivity (API auth check), data feed freshness (last candle < 30min), TradingLoop alive (cycle count incrementing)
- Missed heartbeat detection: counter increments on check failure, resets on success; 2 consecutive failures -> Telegram alert
- REST `/health/production` endpoint returns JSON with per-component status and overall pass/fail
- Extend existing `TelegramAlerter` with command handler dispatcher -- reuse existing httpx client
- `/kill` requires confirmation reply within 30s ("Type CONFIRM to kill") to prevent accidental activation
- `/gonogo` runs `GoNoGoReporter.evaluate()` from Phase 16, formats as Telegram message with emoji pass/fail per criterion
- Authorization: restrict commands to `FINALAYZE_TELEGRAM_ADMIN_CHAT_ID` env var -- only admin chat can trigger

### Claude's Discretion
- Internal data structures for health check results
- Kill switch persistent flag storage mechanism (file vs DB vs env)
- Telegram bot polling interval and webhook vs polling choice
- Health check timeout values for broker ping and feed freshness

### Deferred Ideas (OUT OF SCOPE)
- Dashboard display of health status -- Phase 18
- REST endpoint for `/sandbox/gonogo` -- Phase 18
- Capital scaling automation -- out of scope for v3.0
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| OPS-01 | Kill switch cancels all open orders at broker, stops TradingLoop, sends Telegram critical alert -- response time <30 seconds | KillSwitch class orchestrating BrokerRouter.get_open_orders() + cancel_order(), TradingLoop.stop(), CircuitBreaker.override_level(LIQUIDATE), TelegramAlerter CRITICAL bypass |
| OPS-02 | Health check heartbeat every 5 minutes, REST `/health/production` endpoint, auto-alert on 2 missed heartbeats | HealthMonitor with APScheduler IntervalTrigger, existing health check helpers in system.py, consecutive failure counter |
| OPS-03 | 3-tier alert taxonomy (critical/warning/info) integrated into TelegramMonitor priority queue to prevent alert fatigue | Already implemented in AlertPriority + TelegramMessageQueue; needs verification that all new alerts use correct priority |
| OPS-04 | Telegram bot `/kill` command triggers kill switch, `/gonogo` command runs gate report | Extend TelegramBotHandler commands dict, add confirmation state machine for /kill, reuse GoNoGoReporter for /gonogo |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| APScheduler | 3.x (already installed) | Health monitor heartbeat scheduling | Already used by TradingLoop; consistent pattern |
| httpx | (already installed) | Telegram API calls | Already used by TelegramAlerter |
| structlog | (already installed) | Structured logging | Project standard |
| pydantic | v2 (already installed) | Health check response models | Project standard for all schemas |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| FastAPI | (already installed) | REST `/health/production` and `/kill` endpoints | API layer |

No new dependencies needed. This phase uses only existing libraries.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
  core/
    kill_switch.py         # KillSwitch orchestrator class
    telegram_bot.py        # Extended with /kill, /gonogo commands
    alerts.py              # No changes needed (3-tier already works)
  monitoring/
    health_monitor.py      # HealthMonitor with APScheduler heartbeat
  api/v1/
    system.py              # Add /health/production and POST /kill endpoints
```

### Pattern 1: KillSwitch as Orchestrator
**What:** Standalone class that coordinates multi-component shutdown in deterministic order.
**When to use:** Kill switch activation (Telegram, REST, programmatic).
**Key design:**
```python
# core/kill_switch.py
@dataclass(frozen=True)
class KillSwitchResult:
    orders_cancelled: int
    scheduler_stopped: bool
    breakers_escalated: int
    alert_sent: bool
    elapsed_seconds: float

class KillSwitch:
    def __init__(
        self,
        broker_router: BrokerRouter,
        trading_loop: TradingLoop,
        circuit_breakers: dict[str, CircuitBreaker],
        alerter: TelegramAlerter,
        flag_path: Path = Path("/tmp/finalayze_killed"),
    ) -> None: ...

    def activate(self, reason: str = "manual") -> KillSwitchResult:
        """Execute kill sequence: cancel orders -> stop scheduler -> escalate breakers -> alert."""
        ...

    @property
    def is_killed(self) -> bool:
        """Check persistent flag."""
        return self._flag_path.exists()
```

### Pattern 2: HealthMonitor with Heartbeat Counter
**What:** APScheduler job that pings components every 5 minutes, tracks consecutive failures.
**When to use:** Continuous production health monitoring.
**Key design:**
```python
# monitoring/health_monitor.py
@dataclass(frozen=True)
class HealthCheckResult:
    broker_ok: bool
    feed_fresh: bool
    loop_alive: bool
    timestamp: datetime
    details: dict[str, str]

class HealthMonitor:
    def __init__(
        self,
        broker_router: BrokerRouter,
        trading_loop: TradingLoop,
        alerter: TelegramAlerter,
        check_interval_seconds: int = 300,
        feed_freshness_minutes: int = 30,
    ) -> None:
        self._consecutive_failures: int = 0
        self._scheduler: BackgroundScheduler | None = None
        ...

    def start(self) -> None:
        """Start APScheduler heartbeat job."""
        ...

    def _heartbeat(self) -> None:
        """Run health checks, update counter, alert on 2 consecutive failures."""
        ...
```

### Pattern 3: Telegram Command Confirmation Flow
**What:** `/kill` requires a CONFIRM reply within 30 seconds. State tracked in TelegramBotHandler.
**When to use:** Destructive operations via Telegram.
**Key design:**
```python
# In TelegramBotHandler:
_pending_kill: dict[str, float] = {}  # chat_id -> timestamp of /kill request

async def handle_kill(self, chat_id: str) -> None:
    """Start kill confirmation flow."""
    self._pending_kill[chat_id] = time.monotonic()
    await self._alerter._send("Type CONFIRM to kill all trading within 30s")

async def handle_update(self, update: dict) -> dict:
    # Check for CONFIRM text from pending kill requests
    if text.upper() == "CONFIRM" and chat_id in self._pending_kill:
        if time.monotonic() - self._pending_kill[chat_id] <= 30:
            del self._pending_kill[chat_id]
            result = self._kill_switch.activate(reason=f"telegram:{chat_id}")
            await self._alerter._send(f"Kill switch activated: {result}")
        else:
            del self._pending_kill[chat_id]
            await self._alerter._send("Confirmation expired. Send /kill again.")
```

### Pattern 4: Persistent Kill Flag (File-based)
**What:** A simple file flag at a configurable path. Exists = system killed. Checked at startup.
**When to use:** Kill switch persistence across restarts.
**Rationale:** File-based is simplest, doesn't depend on DB being up (kill switch may fire precisely because DB is down). Path configurable via settings.

### Anti-Patterns to Avoid
- **Embedding kill switch logic in TradingLoop:** Keep KillSwitch as separate orchestrator. TradingLoop only exposes `stop()`.
- **Using database for kill flag:** DB may be down during emergencies. Use a local file.
- **Blocking Telegram response on full kill sequence:** Send acknowledgment immediately, run kill sequence, then send result. But since <30s SLA is critical, synchronous is fine for this use case.
- **Polling-based Telegram bot:** Project already uses webhook-based approach via FastAPI router. Continue using webhooks, not polling.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Scheduled heartbeat | Custom timer thread | APScheduler IntervalTrigger | Already used by TradingLoop; handles thread safety |
| Telegram bot framework | Full bot framework | Extend existing TelegramBotHandler + webhook | Already wired in main.py, minimal code needed |
| Health check caching | Custom TTL cache | Existing `_health_cache` pattern in system.py | 30s TTL cache already implemented |
| Rate-limited alerts | Custom rate limiter | TelegramMessageQueue with AlertPriority | Already handles CRITICAL bypass + rate limiting |

**Key insight:** Nearly all infrastructure for this phase already exists. The work is orchestration and wiring, not building new foundations.

## Common Pitfalls

### Pitfall 1: Kill Switch Timeout Exceeding 30 Seconds
**What goes wrong:** Network timeouts on broker API calls (cancel_order) cause kill switch to exceed 30s SLA.
**Why it happens:** TinkoffBroker uses gRPC with default timeouts; multiple orders to cancel sequentially.
**How to avoid:** Set aggressive timeout (5s per cancel call). Cancel all orders in parallel if possible. Log timing at each step. Fire Telegram alert FIRST (CRITICAL bypasses queue), then do cleanup.
**Warning signs:** Kill switch unit tests not measuring elapsed time.

### Pitfall 2: Confirmation State Leak
**What goes wrong:** `/kill` confirmation state persists forever if user never replies.
**Why it happens:** `_pending_kill` dict grows unbounded.
**How to avoid:** Clean up expired entries on every `handle_update` call. Set TTL of 30 seconds.
**Warning signs:** Memory growth in long-running bot.

### Pitfall 3: Health Monitor False Positives During Market Closed Hours
**What goes wrong:** Broker connectivity check fails outside trading hours, feed freshness check triggers because no new candles.
**Why it happens:** MOEX trading hours are limited (10:00-18:50 MSK).
**How to avoid:** HealthMonitor should be aware of market hours. Consider only alerting during trading hours, or adjusting feed freshness threshold when market is closed.
**Warning signs:** Telegram alerts flooding at night/weekends.

### Pitfall 4: cancel_all_orders Does Not Exist
**What goes wrong:** CONTEXT.md mentions `broker.cancel_all_orders()` but this method does not exist on BrokerBase or TinkoffBroker.
**Why it happens:** Assumption based on desired API, not actual code.
**How to avoid:** KillSwitch must use `broker.get_open_orders()` to list pending orders, then `broker.cancel_order(order_id)` or `broker.cancel_order_safe(order_id)` for each. Alternatively, add `cancel_all_orders()` convenience method to TinkoffBroker that does this internally.
**Warning signs:** Import errors or AttributeError at runtime.

### Pitfall 5: Circular Import with KillSwitch
**What goes wrong:** KillSwitch in `core/` imports from `execution/` (Layer 5) and `risk/` (Layer 4).
**Why it happens:** `core/` is Layer 0, but KillSwitch needs higher-layer components.
**How to avoid:** Use TYPE_CHECKING imports for type hints, accept dependencies via constructor injection (same pattern as `TradingLoop` and `TelegramBotHandler` which live in core/ but are architecturally Layer 6).
**Warning signs:** Circular import errors during testing.

### Pitfall 6: Health Monitor Scheduler Conflict
**What goes wrong:** HealthMonitor's APScheduler instance conflicts with TradingLoop's scheduler.
**Why it happens:** Both use BackgroundScheduler, potential thread pool exhaustion.
**How to avoid:** HealthMonitor should use its own scheduler instance (lightweight, single job). Alternatively, add health check as a job to TradingLoop's existing scheduler.
**Warning signs:** Scheduler thread starvation, missed heartbeats.

## Code Examples

### Cancel All Orders via BrokerRouter (Verified Pattern)
```python
# KillSwitch._cancel_all_orders()
# BrokerBase has cancel_order(order_id) but NOT cancel_all_orders()
# TinkoffBroker has get_open_orders() -> list[OrderStateResult] and cancel_order_safe(order_id)
def _cancel_all_orders(self) -> int:
    cancelled = 0
    for market_id in self._broker_router.registered_markets:
        broker = self._broker_router.route(market_id)
        if hasattr(broker, "get_open_orders"):
            for order in broker.get_open_orders():
                broker.cancel_order_safe(order.order_id)
                cancelled += 1
    return cancelled
```

### Escalate All Circuit Breakers (Verified Pattern)
```python
# CircuitBreaker.override_level(level) exists for manual level override
from finalayze.risk.circuit_breaker import CircuitLevel

def _escalate_breakers(self) -> int:
    escalated = 0
    for cb in self._circuit_breakers.values():
        cb.override_level(CircuitLevel.LIQUIDATE)
        escalated += 1
    return escalated
```

### TradingLoop Stop (Verified Pattern)
```python
# TradingLoop.stop() shuts down scheduler + async loop + connections
# Already tested in test_telegram_stop_command.py
self._trading_loop.stop()
```

### Existing TelegramBotHandler Command Registration (Verified Pattern)
```python
# From core/telegram_bot.py -- commands dict pattern
self._commands: dict[str, Any] = {
    "/status": self.handle_status,
    "/breakers": self.handle_breakers,
    "/stop": self.handle_stop,
    # Add:
    "/kill": self.handle_kill,
    "/gonogo": self.handle_gonogo,
}
```

### GoNoGoReporter Usage (Verified Pattern)
```python
# From monitoring/go_no_go.py -- requires AsyncSession
from finalayze.monitoring.go_no_go import GoNoGoReporter, GateThresholds
from pathlib import Path

thresholds = GateThresholds.from_yaml(Path("config/gate_thresholds.yaml"))
reporter = GoNoGoReporter(thresholds, market_id="moex")
report = await reporter.evaluate(session)  # needs AsyncSession
# report.verdict: GateVerdict (PROCEED/DEFER/ABORT)
# report.criteria: list[CriterionResult] (name, passed, actual, threshold, unit)
```

### AlertPriority CRITICAL Bypass (Verified Pattern)
```python
# CRITICAL messages bypass the queue entirely -- immediate send
# From TelegramMessageQueue.enqueue():
if priority == AlertPriority.CRITICAL:
    await self._send_with_retry(text)
    return  # no queue
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `/stop` command (stops scheduler only) | `/kill` (cancel orders + stop + escalate + alert) | Phase 17 | Full emergency shutdown vs partial stop |
| No health monitoring | 5-min heartbeat with auto-alerting | Phase 17 | Proactive failure detection |
| Flat Telegram alerts | 3-tier priority (already exists in AlertPriority) | Phase 16 | OPS-03 mostly done; wire correctly |

**Key finding:** OPS-03 (3-tier alert taxonomy) is largely already implemented. `AlertPriority` enum with CRITICAL/IMPORTANT/INFO exists. `TelegramMessageQueue` handles CRITICAL bypass and IMPORTANT batching. The work for OPS-03 is verification and ensuring new alerts (kill switch, health monitor) use correct priorities.

## Open Questions

1. **Admin Chat ID vs Allowed Chat IDs**
   - What we know: CONTEXT.md specifies `FINALAYZE_TELEGRAM_ADMIN_CHAT_ID` for kill command authorization. Settings already has `telegram_allowed_chat_ids: list[str]`.
   - What's unclear: Should admin be a separate setting or reuse existing allowed list?
   - Recommendation: Add `telegram_admin_chat_id: str` setting. `/kill` requires admin chat ID. `/gonogo` and read-only commands use `telegram_allowed_chat_ids`. This separates destructive from read-only authorization.

2. **Health Monitor During Market Closed Hours**
   - What we know: MOEX trades 10:00-18:50 MSK. Feed freshness check will false-positive outside hours.
   - What's unclear: Should health monitor pause outside market hours?
   - Recommendation: Keep monitoring running 24/7 but adjust feed freshness check -- skip feed freshness criterion when market is closed. Broker connectivity should always be checked.

3. **TradingLoop Cycle Counter for "Loop Alive" Check**
   - What we know: TradingLoop has `_cycle_instruments_processed`, `_cycle_signals_generated` etc. counters that reset each cycle.
   - What's unclear: No monotonically increasing cycle counter exists.
   - Recommendation: Add `_total_cycles: int` counter to TradingLoop that increments on each strategy cycle. HealthMonitor compares current vs previous snapshot -- if unchanged after 2 checks (10 min), loop is considered dead.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.x + pytest-asyncio |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_kill_switch.py tests/unit/test_health_monitor.py tests/unit/test_telegram_kill_gonogo.py -x` |
| Full suite command | `uv run pytest tests/ -x --timeout=60` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OPS-01 | Kill switch cancels orders, stops loop, escalates breakers, sends alert <30s | unit | `uv run pytest tests/unit/test_kill_switch.py -x` | Wave 0 |
| OPS-01 | Kill switch elapsed time <30s | unit | `uv run pytest tests/unit/test_kill_switch.py::test_kill_switch_under_30s -x` | Wave 0 |
| OPS-01 | Persistent kill flag blocks restart | unit | `uv run pytest tests/unit/test_kill_switch.py::test_persistent_flag -x` | Wave 0 |
| OPS-02 | Health monitor heartbeat every 5 min | unit | `uv run pytest tests/unit/test_health_monitor.py::test_heartbeat_fires -x` | Wave 0 |
| OPS-02 | 2 consecutive failures trigger alert | unit | `uv run pytest tests/unit/test_health_monitor.py::test_two_miss_alert -x` | Wave 0 |
| OPS-02 | REST /health/production returns JSON | unit | `uv run pytest tests/unit/test_health_monitor.py::test_rest_endpoint -x` | Wave 0 |
| OPS-03 | CRITICAL alerts bypass queue | unit | `uv run pytest tests/unit/test_telegram_queue.py -x` | Exists |
| OPS-03 | Kill switch alert uses CRITICAL priority | unit | `uv run pytest tests/unit/test_kill_switch.py::test_alert_priority -x` | Wave 0 |
| OPS-04 | /kill with confirmation flow | unit | `uv run pytest tests/unit/test_telegram_kill_gonogo.py::test_kill_confirm -x` | Wave 0 |
| OPS-04 | /kill confirmation expires after 30s | unit | `uv run pytest tests/unit/test_telegram_kill_gonogo.py::test_kill_timeout -x` | Wave 0 |
| OPS-04 | /gonogo returns formatted gate report | unit | `uv run pytest tests/unit/test_telegram_kill_gonogo.py::test_gonogo -x` | Wave 0 |
| OPS-04 | Admin-only authorization for /kill | unit | `uv run pytest tests/unit/test_telegram_kill_gonogo.py::test_kill_admin_only -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_kill_switch.py tests/unit/test_health_monitor.py tests/unit/test_telegram_kill_gonogo.py -x`
- **Per wave merge:** `uv run pytest tests/ -x --timeout=60`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_kill_switch.py` -- covers OPS-01 (kill switch orchestration, timing, persistent flag)
- [ ] `tests/unit/test_health_monitor.py` -- covers OPS-02 (heartbeat, consecutive failures, REST endpoint)
- [ ] `tests/unit/test_telegram_kill_gonogo.py` -- covers OPS-04 (/kill confirmation, /gonogo formatting, admin auth)

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/alerts.py` -- TelegramAlerter, AlertPriority, TelegramMessageQueue (read in full)
- `src/finalayze/core/telegram_bot.py` -- TelegramBotHandler with /stop, /status, /breakers (read in full)
- `src/finalayze/core/trading_loop.py` -- TradingLoop.stop(), scheduler, cycle counters (read key sections)
- `src/finalayze/risk/circuit_breaker.py` -- CircuitBreaker.override_level(), CircuitLevel enum (read in full)
- `src/finalayze/execution/tinkoff_broker.py` -- get_open_orders(), cancel_order_safe() (read key sections)
- `src/finalayze/execution/broker_base.py` -- BrokerBase ABC, no cancel_all_orders (read in full)
- `src/finalayze/execution/broker_router.py` -- BrokerRouter.registered_markets, route() (read in full)
- `src/finalayze/monitoring/go_no_go.py` -- GoNoGoReporter.evaluate(), GateReport schema (read in full)
- `src/finalayze/api/v1/system.py` -- Existing health check patterns, _check_tinkoff (read in full)
- `src/finalayze/main.py` -- TradingLoop wiring, TelegramBotHandler creation (read in full)
- `config/settings.py` -- Settings class, telegram fields (read in full)
- `tests/unit/test_telegram_stop_command.py` -- Existing test patterns for bot commands (read in full)

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions on kill switch design and health monitoring approach

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new deps
- Architecture: HIGH -- extending well-understood existing patterns (TelegramBotHandler, APScheduler, health checks)
- Pitfalls: HIGH -- identified from actual code review (cancel_all_orders missing, circular import risk, market hours)

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable domain, internal codebase)
