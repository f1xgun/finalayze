# Architecture Research

**Domain:** Production monitoring and go-live validation for autonomous MOEX trading system
**Researched:** 2026-03-21
**Confidence:** HIGH — based on direct codebase inspection of all 6 layers

## Standard Architecture

### System Overview

Current state (v2.0) — what exists today:

```
Layer 6 (API/Dashboard/Loop)
+--------------------------------------------------------------------+
|  FastAPI REST (20+ endpoints)  | Streamlit Dashboard | TradingLoop  |
|  /api/v1/{health,portfolio,    | auth gate + 5 pages | APScheduler  |
|   trades,signals,risk,ml,news} |                     | 3 cycles:    |
|  Prometheus /metrics           |                     |  news_cycle  |
|  TelegramBotHandler            |                     |  strategy_   |
|   (/status, /breakers, /stop)  |                     |   cycle      |
|  MetricsCollector              |                     |  daily_reset |
+--------------------------------------------------------------------+
                    | reads from / calls into
Layer 5 (Execution)
+--------------------------------------------------------------------+
|  BrokerRouter -> TinkoffBroker (live/sandbox)                      |
|               -> AlpacaBroker                                      |
|               -> SimulatedBroker                                   |
|  SandboxPortfolioTracker (shadow ledger: coupons + dividends)      |
|  RetryPolicy, impact.py                                            |
+--------------------------------------------------------------------+
Layer 4 (Strategy/Risk)
+--------------------------------------------------------------------+
|  StrategyCombiner + ADX routing | CircuitBreaker (L1/L2/L3)       |
|  8 strategies (5 enabled)       | PreTradeChecker (11 checks)      |
|  Optuna presets per segment     | PositionSizingPipeline           |
+--------------------------------------------------------------------+
Layer 0-3 (Core/Config/Data/Analysis/ML)
+--------------------------------------------------------------------+
|  schemas.py, events.py        | TinkoffFetcher, MoexISSFetcher     |
|  ValidationLogger (JSONL)     | MacroCacheService, CBRFetcher      |
|  TelegramAlerter (3 priority) | ML ensemble (XGB+LGBM+CatBoost)   |
|  alerts.py                    | CombinedNewsAnalyzer               |
+--------------------------------------------------------------------+
```

Target state (v3.0) — new components annotated:

```
Layer 6 (API/Dashboard/Loop)
+--------------------------------------------------------------------+
|  [EXISTING]               [NEW]                  [MODIFIED]        |
|  FastAPI REST             monitoring/             TradingLoop       |
|  TelegramBotHandler        sandbox_monitor.py     + _health_pulse   |
|  Streamlit Dashboard       gonogo.py              + kill_switch     |
|  TradingLoop               anomaly.py            TelegramBotHandler |
|  Prometheus /metrics       health_monitor.py      + /gonogo        |
|                            kill_switch.py         + /health        |
|                           api/v1/sandbox.py       + /killswitch    |
|                           MetricsCollector        MetricsCollector  |
|                            (extended)             (5 new gauges)   |
+--------------------------------------------------------------------+
Layer 4 (Strategy/Risk) — minimal parameter additions
+--------------------------------------------------------------------+
|  [EXISTING]                         [MODIFIED]                     |
|  CircuitBreaker, PreTradeChecker    accept RolloutPhase param       |
|  PositionSizingPipeline             (tighter limits for minimal     |
|  SandboxPortfolioTracker             capital phase)                 |
+--------------------------------------------------------------------+
Layer 0-1 (Core Schemas + Config) — new types only
+--------------------------------------------------------------------+
|  [EXISTING]                         [NEW]                          |
|  CycleLogEntry, ValidationLogger    GoNoGoReport, GoNoGoGateResult  |
|  PortfolioState, TradeResult        SandboxMetricSnapshot          |
|  Signal, Candle                     HealthCheckResult              |
|  config/settings.py                 config/rollout.py              |
|  config/modes.py                    RolloutPhase (named profiles)   |
+--------------------------------------------------------------------+
```

### Component Responsibilities

| Component | New/Modified/Existing | Responsibility | Layer |
|-----------|----------------------|----------------|-------|
| `SandboxMonitorService` | NEW | Collect sandbox metrics on a schedule: equity curve, P&L, fill rate, slippage, uptime. Persist to TimescaleDB. | 6 |
| `GoNoGoReporter` | NEW | Evaluate gate thresholds against collected metrics. Produce structured pass/fail report with per-gate detail. Pure function: no side effects. | 6 |
| `GradualRolloutConfig` / `RolloutPhase` | NEW | Named config profiles (minimal_50k, scale_200k, target_500k) with per-profile risk parameter overrides. Mode-gated: only active in real mode. | 1 |
| `ProductionHealthMonitor` | NEW | Periodic health pulse: broker ping, data feed freshness, component status. Alerts on degradation. | 6 |
| `AnomalyDetector` | NEW | Z-score alerting on sudden equity moves, fill rate drops, abnormal signal frequency. | 6 |
| `KillSwitch` | NEW | Immediate halt via Telegram /killswitch or API. Stops scheduler, cancels open orders, escalates circuit breakers, alerts. | 6 |
| `/api/v1/sandbox.py` | NEW | REST endpoints: GET /sandbox/metrics, GET /sandbox/gonogo | 6 |
| `TradingLoop` | MODIFIED | Add `_health_pulse_cycle()` APScheduler job. Add `activate_kill_switch()`. | 6 |
| `TelegramBotHandler` | MODIFIED | Add /gonogo, /health, /killswitch commands to existing dispatch map. | 6 |
| `MetricsCollector` | MODIFIED | Add ~5 new Prometheus gauges: fill_rate, signal_divergence, health_pulse_status, kill_switch_active, rollout_phase. | 6 |
| `CircuitBreaker` | MODIFIED | Accept optional `rollout_phase: RolloutPhase` constructor param for threshold overrides. None = existing behavior. | 4 |
| `PreTradeChecker` | MODIFIED | Accept optional `rollout_phase: RolloutPhase` to override max_position_pct. None = existing behavior. | 4 |
| `ValidationLogger` / `CycleLogEntry` | MODIFIED | Extend CycleLogEntry with 3 optional fields: fill_rate_pct, avg_slippage_bps, signal_divergence_pct. Defaults None. | 0 |
| `GoNoGoReport` schema | NEW | Frozen Pydantic model: gate results list, overall pass/fail, recommendation, raw metrics. | 0 |
| `SandboxMetricSnapshot` schema | NEW | Frozen Pydantic model: daily aggregate (date, uptime_pct, fill_rate_pct, avg_slippage_bps, drawdown_pct, signal_divergence_pct). | 0 |

## Recommended Project Structure

New files only (additions to existing tree):

```
src/finalayze/
|
+-- core/
|   +-- schemas.py              # MODIFIED: GoNoGoReport, SandboxMetricSnapshot,
|   |                           #   HealthCheckResult (new frozen models)
|   +-- validation_logger.py    # MODIFIED: CycleLogEntry +3 optional fields
|
+-- monitoring/                 # NEW top-level module (Layer 6, same plane as api/)
|   +-- __init__.py
|   +-- CLAUDE.md
|   +-- sandbox_monitor.py      # SandboxMonitorService
|   +-- gonogo.py               # GoNoGoReporter (pure function)
|   +-- anomaly.py              # AnomalyDetector (z-score alerting)
|   +-- health_monitor.py       # ProductionHealthMonitor (periodic pulse)
|   +-- kill_switch.py          # KillSwitch (halt-all sequence)
|
+-- api/v1/
|   +-- sandbox.py              # NEW: GET /sandbox/metrics, GET /sandbox/gonogo
|   +-- system.py               # MODIFIED: add /health/production endpoint
|   +-- router.py               # MODIFIED: include_router(sandbox.router)
|
+-- config/
|   +-- rollout.py              # NEW: RolloutPhase dataclasses, ROLLOUT_PHASES dict,
|                               #   env var resolver (FINALAYZE_ROLLOUT_PHASE)
|
alembic/versions/
+-- XXXX_add_sandbox_metrics.py # NEW migration: sandbox_metrics hypertable
```

### Structure Rationale

- **`monitoring/` as a new top-level module:** Keeps sandbox monitoring, go/no-go, anomaly detection, and health pulse grouped by concern. All are Layer 6 — they read from layers 0-5 but nothing reads from them. Avoids polluting `core/`, `api/`, or `execution/` with cross-cutting concerns. Mirrors the existing separation: `api/` handles HTTP interface, `monitoring/` handles operational intelligence.

- **`config/rollout.py` at Layer 1:** Rollout profiles are pure configuration — named presets of risk parameters. Belongs with `settings.py` and `segments.py`. Consumed by Layer 4 (CircuitBreaker, PreTradeChecker) and Layer 6 (TradingLoop). No upward imports. Validated at startup via Pydantic.

- **Schema additions in `core/schemas.py`:** All new data types are frozen Pydantic models at Layer 0, importable from any layer without violating the dependency hierarchy.

- **`ValidationLogger` extended in-place:** CycleLogEntry is already the per-cycle data carrier. Adding optional fields (default None) is backward compatible. Existing JSONL files remain parseable by old and new code.

## Architectural Patterns

### Pattern 1: Metric Collection via Existing ValidationLogger + DB Persistence

**What:** `SandboxMonitorService` reads from `ValidationLogger` (JSONL), aggregates daily metrics, and persists `SandboxMetricSnapshot` to TimescaleDB. The JSONL log is the primary write path (fast, append-only). TimescaleDB is the reporting read path (queryable, time-series optimized).

**When to use:** Throughout the sandbox validation period (2-4 weeks before go-live). Runs daily as an APScheduler job or on-demand via API.

**Trade-offs:** Two storage layers add slight complexity, but they serve different purposes. JSONL is near-zero-latency for TradingLoop writes. TimescaleDB supports time-range queries for trend analysis and dashboard display. Not using only TimescaleDB avoids adding a synchronous DB write to the strategy cycle hot path.

**Example:**
```python
# monitoring/sandbox_monitor.py (Layer 6)
class SandboxMonitorService:
    def __init__(
        self,
        validation_logger: ValidationLogger,
        broker: SandboxPortfolioTracker,
        session_factory: AsyncSessionFactory,
    ) -> None: ...

    async def collect_daily_snapshot(self, date: date) -> SandboxMetricSnapshot:
        entries = self._validation_logger.get_entries()
        today_entries = [e for e in entries if e.timestamp.date() == date]
        snapshot = SandboxMetricSnapshot(
            date=date,
            uptime_pct=self._compute_uptime(today_entries),
            fill_rate_pct=self._compute_fill_rate(today_entries),
            avg_slippage_bps=self._compute_slippage(today_entries),
            drawdown_pct=float(self._broker.shadow_portfolio().equity),
            signal_divergence_pct=self._compute_divergence(today_entries),
        )
        await self._persist(snapshot)
        return snapshot
```

### Pattern 2: Go/No-Go as Pure Evaluation Function

**What:** `GoNoGoReporter.evaluate(snapshots: list[SandboxMetricSnapshot]) -> GoNoGoReport` is a pure function — no side effects, no broker calls, no DB writes. Takes collected snapshots, applies threshold rules, returns a structured report.

**When to use:** Called by `GET /api/v1/sandbox/gonogo` on demand and by `/gonogo` Telegram command. The operator reads the report and makes the human decision to proceed.

**Trade-offs:** Pure function is trivially testable (parametrize threshold values). Requires SandboxMonitorService to have already persisted the data. The reporter never reads from the broker directly — eliminates runtime coupling to live systems.

**Example:**
```python
# core/schemas.py (Layer 0)
class GoNoGoGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    gate: str            # "uptime", "fill_rate", "drawdown", "signal_divergence"
    threshold: float
    actual: float
    passed: bool
    detail: str

class GoNoGoReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_at: datetime
    period_days: int
    overall_pass: bool
    gates: list[GoNoGoGateResult]
    recommendation: str  # "PROCEED", "DEFER", "ABORT"
```

### Pattern 3: Kill Switch as TradingLoop State Flag

**What:** `KillSwitch.activate(reason)` sets a shared `threading.Event` on TradingLoop (reusing the existing `_stop_event` pattern), cancels broker orders, escalates circuit breakers to LIQUIDATE, and sends a CRITICAL Telegram alert.

**When to use:** Emergency only. One-way: manual recovery required via mode transition API.

**Trade-offs:** `threading.Event` is the mechanism TradingLoop already uses for graceful shutdown. Reusing it avoids new synchronization primitives. The key addition is order cancellation before shutdown — without it, pending orders remain live at the broker.

**Example:**
```python
# core/trading_loop.py (MODIFIED)
class TradingLoop:
    def activate_kill_switch(self, reason: str) -> None:
        """Immediately halt all trading. Manual recovery required."""
        self._stop_event.set()
        self._kill_switch_active = True
        # Escalate all circuit breakers
        for cb in self._circuit_breakers.values():
            cb.override_level(CircuitLevel.LIQUIDATE)
        # Cancel pending orders via BrokerRouter
        for market_id, broker in self._broker_router.brokers.items():
            try:
                broker.cancel_all_pending()
            except Exception:
                _log.exception("kill_switch.cancel_failed", market=market_id)
        # Alert (fire-and-forget into async queue)
        asyncio.run_coroutine_threadsafe(
            self._alerter.send_critical(f"KILL SWITCH ACTIVATED: {reason}"),
            self._async_loop,
        )
```

### Pattern 4: Gradual Rollout as Named Config Profiles

**What:** `config/rollout.py` defines frozen `RolloutPhase` dataclasses with risk parameter overrides. Active phase resolved from `FINALAYZE_ROLLOUT_PHASE` env var at startup. `PreTradeChecker` and `CircuitBreaker` accept an optional phase to override their limits.

**When to use:** After go/no-go gate passes. Start `minimal_50k` (3% max position, 1% daily loss, 2% DD auto-stop), scale to `target_500k` after N profitable weeks via env var change + restart.

**Trade-offs:** Config-driven means no code changes for phase transitions. Validated by Pydantic at startup. Avoids runtime mutation of live risk parameters. The None-default on both risk components means zero behavioral change for all existing tests and sandbox mode.

**Example:**
```python
# config/rollout.py (Layer 1 — NEW)
from __future__ import annotations
import dataclasses
from decimal import Decimal

@dataclasses.dataclass(frozen=True)
class RolloutPhase:
    name: str
    capital_rub: Decimal
    max_position_pct: Decimal    # 0.03 for minimal vs 0.05 for target
    daily_loss_limit_pct: Decimal
    dd_auto_stop_pct: Decimal

ROLLOUT_PHASES: dict[str, RolloutPhase] = {
    "minimal_50k":  RolloutPhase("minimal_50k",   Decimal("50000"),  Decimal("0.03"), Decimal("0.01"), Decimal("0.02")),
    "scale_200k":   RolloutPhase("scale_200k",   Decimal("200000"),  Decimal("0.04"), Decimal("0.02"), Decimal("0.05")),
    "target_500k":  RolloutPhase("target_500k",  Decimal("500000"),  Decimal("0.05"), Decimal("0.03"), Decimal("0.10")),
}

def get_active_phase() -> RolloutPhase | None:
    import os
    name = os.environ.get("FINALAYZE_ROLLOUT_PHASE")
    return ROLLOUT_PHASES.get(name) if name else None
```

## Data Flow

### Sandbox Monitoring Collection Flow

```
TradingLoop._strategy_cycle()
    |
    | (per cycle, existing)
    v
ValidationLogger.log_cycle(CycleLogEntry)  -->  results/validation/cycles.jsonl
    |
    | (daily, new APScheduler job OR on-demand via API)
    v
SandboxMonitorService.collect_daily_snapshot(date)
    |-- reads: ValidationLogger.get_entries() [JSONL]
    |-- reads: SandboxPortfolioTracker.shadow_portfolio()
    v
SandboxMetricSnapshot
    |-- persists to: TimescaleDB sandbox_metrics hypertable
    |-- records to: MetricsCollector.record_sandbox_snapshot()
```

### Go/No-Go Report Flow

```
GET /api/v1/sandbox/gonogo  OR  Telegram /gonogo
    |
    v
SandboxMonitorService.get_snapshots(period_days=14)  [reads TimescaleDB]
    |
    v
GoNoGoReporter.evaluate(snapshots)  [pure function]
    |-- gate 1: uptime_pct >= 0.99? (MOEX market hours covered)
    |-- gate 2: fill_rate_pct >= 0.95? (filled / submitted orders)
    |-- gate 3: max_drawdown_pct < 0.05? (peak-to-trough on shadow_equity)
    |-- gate 4: signal_divergence_pct < 0.50? (distribution vs backtest)
    v
GoNoGoReport(overall_pass, gates, recommendation)
    |-- returns as JSON  (API endpoint)
    |-- formats as Telegram message  (bot command)
```

### Production Health Pulse Flow

```
TradingLoop._health_pulse_cycle()  [NEW APScheduler job, every 5 min]
    |
    v
ProductionHealthMonitor.check()
    |-- TinkoffBroker.get_portfolio()  (broker liveness, 5s timeout)
    |-- ValidationLogger: last entry timestamp < 15 min? (feed freshness)
    |-- CircuitBreaker.level  (risk state)
    |-- Redis ping  (cache health)
    v
HealthCheckResult(status, components, degraded_components)
    |-- if degraded: TelegramAlerter.send_critical() [IMPORTANT priority]
    |-- always: MetricsCollector.record_health_pulse()
    v
Prometheus: finalayze_health_pulse_status{component="..."}
```

### Kill Switch Activation Flow

```
Telegram /killswitch  OR  POST /api/v1/system/killswitch
    |
    v
TradingLoop.activate_kill_switch(reason)
    |-- (1) _stop_event.set()              [halts scheduler from accepting new cycles]
    |-- (2) _kill_switch_active = True     [prevents restart]
    |-- (3) CircuitBreakers -> LIQUIDATE   [existing liquidation path runs if cycle in progress]
    |-- (4) BrokerRouter.cancel_pending()  [cancel open orders at broker]
    |-- (5) TelegramAlerter.send_critical("KILL SWITCH: {reason}")
    |-- (6) MetricsCollector.set_kill_switch_active(True)
    v
Prometheus alert: finalayze_kill_switch_active == 1  -->  Alertmanager  -->  Telegram
```

## Integration Points

### New vs. Existing: Clear Boundaries

| New Component | Reads From (existing) | Writes To (existing) | Does NOT touch |
|---|---|---|---|
| `SandboxMonitorService` | `ValidationLogger`, `SandboxPortfolioTracker` | TimescaleDB, `MetricsCollector` | TradingLoop directly, strategies, risk |
| `GoNoGoReporter` | TimescaleDB snapshots | Nothing (pure function) | Broker, execution, strategies |
| `ProductionHealthMonitor` | `TinkoffBroker`, `ValidationLogger`, Redis, `CircuitBreaker` | `TelegramAlerter`, `MetricsCollector` | Strategies, ML, data fetchers |
| `KillSwitch` | Nothing | `TradingLoop._stop_event`, `CircuitBreaker`, `BrokerRouter`, `TelegramAlerter` | Data layer, ML, API read paths |
| `GradualRolloutConfig` | env var `FINALAYZE_ROLLOUT_PHASE` | `PreTradeChecker`, `CircuitBreaker` (at construction only) | Nothing above Layer 4 |
| `/api/v1/sandbox` | `SandboxMonitorService`, `GoNoGoReporter` | Nothing (read-only endpoints) | TradingLoop directly |

### Modified Existing Components

| Component | Modification | Regression Risk |
|---|---|---|
| `CycleLogEntry` in `validation_logger.py` | Add 3 optional fields with None defaults. Old JSONL files still parse. | LOW |
| `TradingLoop` | Add 1 new APScheduler job (`_health_pulse_cycle`). Add `activate_kill_switch()` method. | MEDIUM — scheduler modification; must not affect existing news/strategy/daily_reset cycles. Test scheduling isolation. |
| `TelegramBotHandler` | Add 3 new commands to `_commands` dict. Existing commands unaffected. | LOW — additive dispatch table extension |
| `PreTradeChecker` | Add `rollout_phase: RolloutPhase | None = None` param. None preserves existing behavior. | LOW — optional param, all existing call sites pass None implicitly |
| `CircuitBreaker` | Add `rollout_phase: RolloutPhase | None = None` constructor param. | LOW — optional param |
| `MetricsCollector` | Add ~5 new static gauge registrations. No existing gauge touched. | LOW — Prometheus registry is additive |
| `api/v1/router.py` | `include_router(sandbox.router)`. Existing routes unaffected. | LOW |

### External Service Boundaries

| Service | Integration Pattern | v3.0 Notes |
|---------|---------------------|------------|
| Tinkoff Sandbox gRPC | Existing `TinkoffBroker(sandbox=True)` via `SandboxPortfolioTracker` | No new integration; SandboxMonitorService reads from SandboxPortfolioTracker only |
| Tinkoff Live gRPC | Existing `TinkoffBroker(sandbox=False)` | Kill switch calls `cancel_order()` on all open orders via BrokerRouter |
| TimescaleDB | New `sandbox_metrics` hypertable via existing async SQLAlchemy session factory | One new Alembic migration; uses existing `get_async_session_factory()` |
| Telegram Bot API | Existing `TelegramAlerter` priority queue + new bot commands | 3 new commands: /gonogo, /health, /killswitch; existing /status, /breakers, /stop unchanged |
| Prometheus | Extended `MetricsCollector` with ~5 new gauges | Existing /metrics endpoint; no new scrape configuration |

## Anti-Patterns

### Anti-Pattern 1: Monitoring Logic Embedded in TradingLoop

**What people do:** Add go/no-go evaluation, metric aggregation, and health pulse directly inside TradingLoop's `strategy_cycle` or `daily_reset`.

**Why it's wrong:** TradingLoop is already a 500+ line orchestrator with 20+ injected dependencies. Adding monitoring concerns creates untestable code, circular coupling between execution and reporting, and scheduler contention (a slow health check delays trade execution on the same thread pool).

**Do this instead:** `SandboxMonitorService` and `ProductionHealthMonitor` are standalone services with separate APScheduler jobs. TradingLoop only calls `validation_logger.log_cycle()` (already implemented) — monitoring services read from the logger independently via `get_entries()`.

### Anti-Pattern 2: Automated Go/No-Go Block at Real Mode Startup

**What people do:** Evaluate go/no-go thresholds inside `ModeManager.transition_to(REAL)`, refusing to start if gates don't pass.

**Why it's wrong:** The go/no-go decision is a human decision, not a machine gate. Automated blocking at startup creates a chicken-and-egg problem. It also hides the root cause of failures. The existing `real_confirmed` environment variable guard is the correct automated safety.

**Do this instead:** Go/no-go is an on-demand report via `GET /api/v1/sandbox/gonogo`. The operator reviews it, then initiates the mode transition via `POST /api/v1/mode {mode: "real", confirm_token: "..."}`. The confirmation token gate already enforces human intent.

### Anti-Pattern 3: Kill Switch That Only Stops the Scheduler

**What people do:** Kill switch calls `scheduler.shutdown()` and considers the job done.

**Why it's wrong:** Pending orders remain live at the broker. In MOEX, a limit order placed during the morning session remains valid until market close. A network partition between scheduler shutdown and order cancellation leaves uncontrolled positions overnight.

**Do this instead:** Kill switch follows a strict sequence: (1) stop accepting new cycles, (2) escalate circuit breakers to LIQUIDATE, (3) cancel all pending orders via `BrokerRouter`, (4) alert, (5) shut down scheduler. Step 3 is the critical step. If the broker cancel fails, log and continue — the LIQUIDATE circuit state ensures no new orders are placed.

### Anti-Pattern 4: Signal Divergence Computed Per-Bar Against Backtest Files

**What people do:** On every strategy cycle, load backtest CSV output, find the row matching today's date, compare sandbox signal against backtest signal.

**Why it's wrong:** Backtest output files are not indexed for live lookup. Parsing CSV/JSONL on every cycle (every few minutes) adds latency, creates fragile file path coupling, and the comparison is meaningless at bar level (sandbox uses live prices, backtest uses historical — they will always differ on individual bars).

**Do this instead:** Signal divergence is a daily aggregate metric computed by `SandboxMonitorService`. It compares the rolling distribution of signal directions (fraction BUY / SELL / HOLD) in sandbox cycles over the past N days against backtest summary statistics stored in `results/iterations/`. The divergence metric detects structural regime shifts, not individual bar differences.

## Suggested Build Order

The build order is dictated by the dependency chain — each phase unlocks the next with no circular dependencies.

### Phase 1: Schema and Config Foundation (zero behavior change)

1. `core/schemas.py` — add `GoNoGoReport`, `GoNoGoGateResult`, `SandboxMetricSnapshot`, `HealthCheckResult` schemas
2. `core/validation_logger.py` — extend `CycleLogEntry` with 3 optional None-default fields
3. `config/rollout.py` — `RolloutPhase` dataclasses, `ROLLOUT_PHASES` dict, `get_active_phase()` resolver
4. Alembic migration — `sandbox_metrics` hypertable (date, uptime_pct, fill_rate_pct, avg_slippage_bps, drawdown_pct, signal_divergence_pct)

Rationale: Schemas must exist before any service can produce or consume them. Rollout config must exist before risk layer can be modified. All existing tests pass unchanged — no behavior change.

### Phase 2: Risk Layer Rollout Wiring (tightened limits)

5. `risk/circuit_breaker.py` — accept optional `rollout_phase` for threshold overrides
6. `risk/pre_trade_check.py` — accept optional `rollout_phase` for `max_position_pct`, `daily_loss_limit`
7. Unit tests for rollout-phase-aware risk limits (parametrize phases)

Rationale: Risk limits must be validated as tighter before any production deployment. Isolated to Layer 4 with no Layer 5/6 dependencies. Tests confirm that `minimal_50k` profile enforces 3% max position vs the default 5%.

### Phase 3: Monitoring Services

8. `monitoring/sandbox_monitor.py` — `SandboxMonitorService` (metric collection, DB persistence)
9. `monitoring/gonogo.py` — `GoNoGoReporter` (pure threshold evaluation)
10. `monitoring/anomaly.py` — `AnomalyDetector` (z-score alerting)
11. `monitoring/health_monitor.py` — `ProductionHealthMonitor` (periodic pulse)
12. Tests: SandboxMonitorService (DB integration with async fixtures), GoNoGoReporter (unit, pure function parametrized by thresholds)

Rationale: Services depend on schemas from Phase 1 and read from existing ValidationLogger + SandboxPortfolioTracker. Pure services with no effect on TradingLoop — safe to build and test in isolation.

### Phase 4: API Endpoints

13. `api/v1/sandbox.py` — `GET /sandbox/metrics`, `GET /sandbox/gonogo`
14. `api/v1/system.py` — extend with `/health/production`
15. `api/v1/router.py` — `include_router(sandbox.router)`
16. API integration tests

Rationale: Endpoints are thin wrappers over monitoring services. Services must exist first. Follows the existing sub-router pattern — lowest-risk change in the API layer.

### Phase 5: TradingLoop and Telegram Extensions (highest-risk change, done last)

17. `monitoring/kill_switch.py` — `KillSwitch` class
18. `core/trading_loop.py` — add `_health_pulse_cycle()` APScheduler job, `activate_kill_switch()`, wire rollout phase at construction
19. `core/telegram_bot.py` — add `/gonogo`, `/health`, `/killswitch` commands
20. Integration tests for kill switch sequence (mock broker, assert cancel_order called, assert LIQUIDATE set)

Rationale: TradingLoop modification is the highest-regression-risk change — done last so all dependencies are tested. Kill switch sequence tested with mocked broker before wiring into the live scheduler. APScheduler job isolation tested to confirm no interference with existing cycles.

### Phase 6: Streamlit Dashboard Pages (optional)

21. `dashboard/pages/sandbox_validation.py` — sandbox metrics timeline, go/no-go gate status
22. `dashboard/pages/production_health.py` — health pulse history, kill switch status

Rationale: Dashboard pages are read-only API consumers. Can be built any time after Phase 4 without affecting trading functionality. Mark as optional for the milestone — Telegram commands provide equivalent operational visibility.

## Sources

- Direct inspection: `src/finalayze/` codebase — all 6 layers, all modified files
- `src/finalayze/core/validation_logger.py` — existing CycleLogEntry schema and JSONL write pattern
- `src/finalayze/execution/sandbox_tracker.py` — SandboxPortfolioTracker shadow ledger and BrokerBase interface
- `src/finalayze/core/trading_loop.py` — APScheduler cycle structure, _stop_event threading pattern, async_loop pattern
- `src/finalayze/core/telegram_bot.py` — command dispatch table, whitelist pattern
- `src/finalayze/api/metrics.py` — existing Prometheus gauge/counter/histogram registrations
- `src/finalayze/risk/circuit_breaker.py` — L1/L2/L3 threshold, sticky escalation, override_level()
- `src/finalayze/risk/pre_trade_check.py` — 11-check pipeline, max_position_pct parameter
- `docs/operations/GO_LIVE_CHECKLIST.md` — existing go-live procedure and real_confirmed guard
- `docs/operations/MONITORING.md` — existing Prometheus metrics and alert rules
- `.planning/PROJECT.md` — v3.0 milestone requirements (uptime >= 99%, fill rate >= 95%, DD < 5%, signal divergence < 50%)

---
*Architecture research for: production monitoring and go-live validation for autonomous MOEX trading system*
*Researched: 2026-03-21*
