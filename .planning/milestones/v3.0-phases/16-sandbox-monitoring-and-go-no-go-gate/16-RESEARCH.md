# Phase 16: Sandbox Monitoring and Go/No-Go Gate - Research

**Researched:** 2026-03-21
**Domain:** TimescaleDB metric persistence, anomaly detection, gate evaluation
**Confidence:** HIGH

## Summary

Phase 16 adds three capabilities to the existing TradingLoop: (1) a SandboxMonitorService that persists per-cycle metrics to a new `sandbox_metrics` TimescaleDB hypertable, (2) slippage capture in `_submit_order` comparing fill price against expected price, and (3) a GoNoGoReporter that evaluates 8 configurable thresholds to produce a PROCEED/DEFER/ABORT decision. Anomaly detection runs post-cycle with Telegram alerts.

The codebase already has all foundational patterns: `_persist_equity_snapshots` for async DB writes via `_run_async`, `CycleLogEntry` / `ValidationLogger` for post-cycle data collection, `TelegramAlerter` with priority queue for alerting, `MetricsCollector` for Prometheus metrics, and migration 003 for TimescaleDB hypertable creation. The work is primarily integration -- wiring new services into existing hook points.

**Primary recommendation:** Build SandboxMonitorService as a standalone service class (not embedded in TradingLoop), called from `_strategy_cycle` finally block. GoNoGoReporter is a pure function evaluator with no cycle coupling -- called on-demand only.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- SandboxMonitorService persists metrics to a new `sandbox_metrics` TimescaleDB hypertable (Alembic migration 005), following the existing `portfolio_snapshots` pattern
- Collection happens via post-cycle hook in `_strategy_cycle` finally block -- every cycle, zero-config
- Slippage captured in `_submit_order` by computing `(fill_price - expected_price) / expected_price * 10000` bps using `candles[-1].close` as expected price
- Uptime tracked via heartbeat counter in SandboxMonitorService -- increment per successful cycle, gap detection = downtime
- Gate evaluation is on-demand via `GoNoGoReporter.evaluate()` -- called from REST endpoint and Telegram `/gonogo` command
- 3-tier result: PROCEED / DEFER / ABORT with per-criterion pass/fail breakdown
- Thresholds derived from walk-forward backtest stats in `results/iterations/history.jsonl`, computed as percentile bands, stored in `config/gate_thresholds.yaml`
- Minimum 5 trading days of sandbox data required before gate can return PROCEED
- Rolling z-score (window=20 cycles) for drawdown anomalies; threshold-based for fill rate (<90%) and slippage (>50bps)
- 30-minute cooldown per metric to prevent alert fatigue from repeated threshold breaches
- Anomaly checking runs post-cycle in SandboxMonitorService after each metric persist
- Alerts via existing TelegramAlerter with `AlertPriority.CRITICAL`, new `on_anomaly_detected(metric, value, threshold)` method

### Claude's Discretion
- Internal data model for `SandboxMetricRow` (DB columns, indexes)
- GoNoGoReporter internal evaluation logic and criterion ordering
- Z-score window size tuning (20 suggested, can adjust)
- Exact gate threshold percentile values from backtest distribution

### Deferred Ideas (OUT OF SCOPE)
- REST endpoint `/sandbox/gonogo` -- Phase 18 (Dashboard and API Integration)
- Streamlit sandbox dashboard page -- Phase 18
- Telegram `/gonogo` command -- Phase 17 (Production Operations)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MON-01 | SandboxMonitorService collects per-cycle metrics (trades, P&L, equity, fill rate, uptime) and persists to TimescaleDB | Existing `_persist_equity_snapshots` async pattern + migration 003 hypertable pattern; new `sandbox_metrics` table with Alembic 005 |
| MON-02 | Slippage capture records expected_price at signal time vs fill_price at execution, computes realized slippage in bps | `_submit_order` at line 1445 already has fill_price; `candles[-1].close` available as expected price; formula locked in CONTEXT |
| MON-04 | Anomaly detector triggers Telegram alerts on drawdown spikes (>2-sigma), fill rate drops (<90%), and slippage outliers (>50bps) | `TelegramAlerter` has `AlertPriority.CRITICAL` with queue bypass; rolling z-score is stdlib math (no external deps) |
| GATE-01 | GoNoGoReporter evaluates formalized thresholds (uptime >=99%, fill rate >=95%, max DD <5%, trades >=5/5days, signal divergence <50%) | Pure evaluator reading from `sandbox_metrics` table; Pydantic model for structured report |
| GATE-02 | Gate thresholds derived from walk-forward backtest distribution percentiles, not hardcoded | `results/iterations/history.jsonl` has wf_sharpe, wf_max_drawdown, trade_count; YAML config at `config/gate_thresholds.yaml` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.0 (async) | ORM for sandbox_metrics table | Already used project-wide |
| Alembic | existing | Migration 005 for sandbox_metrics hypertable | Existing migration pattern (001-004) |
| TimescaleDB | existing | Hypertable for time-series metric storage | Already used for portfolio_snapshots |
| Pydantic v2 | existing | GoNoGoReport, SandboxMetricRow schemas | Project standard for all schemas |
| PyYAML | existing | gate_thresholds.yaml loading | Lightweight config file format |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | existing | Percentile computation from history.jsonl | Threshold derivation script |
| prometheus_client | existing | Sandbox-specific Prometheus gauges | Extend MetricsCollector |

No new dependencies required. Everything uses existing project libraries.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
├── monitoring/
│   ├── __init__.py
│   ├── sandbox_monitor.py     # SandboxMonitorService
│   ├── anomaly_detector.py    # AnomalyDetector (z-score + threshold)
│   └── go_no_go.py            # GoNoGoReporter + GateResult schema
config/
│   └── gate_thresholds.yaml   # Threshold config (generated from backtest)
alembic/versions/
│   └── 005_sandbox_metrics.py # TimescaleDB hypertable migration
scripts/
│   └── derive_gate_thresholds.py  # One-shot: read history.jsonl -> YAML
```

### Pattern 1: Standalone Service with Post-Cycle Hook

**What:** SandboxMonitorService is a standalone class, not embedded in TradingLoop. TradingLoop calls it from the `_strategy_cycle` finally block.

**When to use:** Always -- matches STATE.md decision "Monitoring services standalone (not embedded in TradingLoop)"

**Example:**
```python
# src/finalayze/monitoring/sandbox_monitor.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

@dataclass(frozen=True)
class CycleMetrics:
    """Metrics collected from a single strategy cycle."""
    timestamp: datetime
    trade_count: int
    pnl_rub: Decimal
    equity_rub: Decimal
    fill_rate: float        # orders_filled / orders_submitted (0-1)
    uptime_cycles: int      # cumulative successful cycles
    signals_generated: int
    errors_caught: int
    max_slippage_bps: float
    avg_slippage_bps: float

class SandboxMonitorService:
    """Collects and persists sandbox cycle metrics to TimescaleDB."""

    def __init__(self, alerter: TelegramAlerter | None = None) -> None:
        self._cycle_count: int = 0
        self._anomaly_detector = AnomalyDetector(alerter=alerter)
        self._slippage_buffer: list[float] = []  # per-cycle accumulator

    def record_slippage(self, slippage_bps: float) -> None:
        """Called from _submit_order for each fill."""
        self._slippage_buffer.append(slippage_bps)

    def on_cycle_complete(self, metrics: CycleMetrics) -> None:
        """Post-cycle hook: persist + check anomalies."""
        self._cycle_count += 1
        self._persist_metrics(metrics)
        self._anomaly_detector.check(metrics)
        self._slippage_buffer.clear()
```

### Pattern 2: TradingLoop Integration Point

**What:** Minimal TradingLoop changes -- add SandboxMonitorService as optional dependency, call from finally block.

**Example:**
```python
# In TradingLoop.__init__:
self._sandbox_monitor: SandboxMonitorService | None = sandbox_monitor

# In _strategy_cycle finally block (after existing CycleLogEntry):
if self._sandbox_monitor is not None:
    metrics = CycleMetrics(
        timestamp=self._now(),
        trade_count=self._cycle_orders_filled,
        pnl_rub=Decimal(str(equity_rub - baseline_rub)),
        equity_rub=Decimal(str(equity_rub)),
        fill_rate=(self._cycle_orders_filled / max(self._cycle_orders_submitted, 1)),
        uptime_cycles=self._sandbox_monitor.cycle_count,
        signals_generated=self._cycle_signals_generated,
        errors_caught=self._cycle_errors_caught,
        max_slippage_bps=max(self._sandbox_monitor.slippage_buffer, default=0.0),
        avg_slippage_bps=(sum(self._sandbox_monitor.slippage_buffer) / max(len(self._sandbox_monitor.slippage_buffer), 1)),
    )
    self._sandbox_monitor.on_cycle_complete(metrics)
```

### Pattern 3: Slippage Capture in _submit_order

**What:** Compute slippage at fill time, record to monitor and Prometheus.

**Example:**
```python
# In _submit_order, after result.filled check (line ~1466):
expected_price = candles[-1].close if candles else None
if result.fill_price is not None and expected_price is not None and expected_price > 0:
    slippage_bps = float(
        (result.fill_price - expected_price) / expected_price * 10000
    )
else:
    slippage_bps = 0.0

MetricsCollector.record_trade(
    market=market_id, side=order.side.lower(),
    slippage_bps=slippage_bps, fill_latency_seconds=0.0,
)
if self._sandbox_monitor is not None:
    self._sandbox_monitor.record_slippage(slippage_bps)
```

### Pattern 4: GoNoGoReporter as Pure Evaluator

**What:** Reads sandbox_metrics from DB, evaluates against config thresholds, returns structured report.

**Example:**
```python
# src/finalayze/monitoring/go_no_go.py
from enum import StrEnum

class GateVerdict(StrEnum):
    PROCEED = "PROCEED"
    DEFER = "DEFER"
    ABORT = "ABORT"

@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    actual: float
    threshold: float
    unit: str

@dataclass(frozen=True)
class GateReport:
    verdict: GateVerdict
    criteria: list[CriterionResult]
    sandbox_days: int
    evaluated_at: datetime
    reason: str

class GoNoGoReporter:
    def __init__(self, thresholds: GateThresholds) -> None:
        self._thresholds = thresholds

    async def evaluate(self) -> GateReport:
        """On-demand evaluation -- reads sandbox_metrics from DB."""
        metrics = await self._load_recent_metrics()
        if self._sandbox_days(metrics) < 5:
            return GateReport(verdict=GateVerdict.DEFER, ...)
        criteria = [
            self._check_uptime(metrics),
            self._check_fill_rate(metrics),
            self._check_max_drawdown(metrics),
            self._check_trade_count(metrics),
            self._check_signal_frequency(metrics),
            self._check_critical_errors(metrics),
            self._check_slippage(metrics),
            self._check_signal_divergence(metrics),
        ]
        # ABORT if any critical criterion fails
        # DEFER if any non-critical fails
        # PROCEED if all pass
```

### Pattern 5: Anomaly Detection with Cooldown

**What:** Rolling z-score for drawdown, simple thresholds for fill rate and slippage, with per-metric cooldown.

**Example:**
```python
class AnomalyDetector:
    _COOLDOWN_SECONDS = 1800  # 30 minutes

    def __init__(self, alerter: TelegramAlerter | None = None, window: int = 20):
        self._window = window
        self._alerter = alerter
        self._drawdown_history: deque[float] = deque(maxlen=window)
        self._last_alert: dict[str, float] = {}  # metric -> monotonic timestamp

    def check(self, metrics: CycleMetrics) -> None:
        self._check_drawdown_zscore(metrics.drawdown_pct)
        self._check_fill_rate(metrics.fill_rate)
        self._check_slippage(metrics.max_slippage_bps)

    def _is_cooled_down(self, metric: str) -> bool:
        last = self._last_alert.get(metric, 0.0)
        return (time.monotonic() - last) >= self._COOLDOWN_SECONDS
```

### Anti-Patterns to Avoid
- **Embedding monitor logic in TradingLoop:** TradingLoop is already 1900+ lines. Keep SandboxMonitorService as standalone class, pass as optional dependency.
- **Hardcoding gate thresholds:** GATE-02 explicitly requires backtest-derived thresholds. Use YAML config generated from history.jsonl.
- **Alerting without cooldown:** Without cooldown, a sustained anomaly generates alerts every cycle (every 60 minutes for strategy_cycle). 30-minute cooldown prevents fatigue.
- **Synchronous DB writes in finally block:** Use `_run_async` pattern already established for `_persist_equity_snapshots`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Time-series storage | Custom file-based metric store | TimescaleDB hypertable | Already in stack, handles retention, compression, aggregation |
| Z-score computation | External stats library | `statistics.mean()` + `statistics.stdev()` from stdlib | Window of 20 values; no numpy needed for this |
| Alert rate limiting | Custom deduplication | Existing `TelegramMessageQueue` rate limiter + per-metric cooldown dict | Queue already handles 20 msg/min limit |
| Async DB writes from sync context | New event loop per write | Existing `_run_async()` bridge in TradingLoop | Already proven pattern in codebase |
| Config file loading | Custom parser | PyYAML `yaml.safe_load()` | Standard, already in dependency tree |

**Key insight:** This phase has zero new external dependencies. All patterns exist in the codebase -- the work is integration and new service classes using established conventions.

## Common Pitfalls

### Pitfall 1: Slippage Measurement in Sandbox
**What goes wrong:** Tinkoff sandbox fills at 100% rate with synthetic prices. Slippage will always be near-zero.
**Why it happens:** Sandbox is simulated -- no real orderbook depth or market impact.
**How to avoid:** The CONTEXT.md and STATE.md note this. Capture slippage using the formula `(fill_price - expected_price) / expected_price * 10000` bps. In sandbox, `candles[-1].close` is the expected price. The measurement infrastructure is correct even if sandbox values are unrealistic. Consider MOEX ISS mid-price comparison for more realistic measurement (noted as blocker in STATE.md).
**Warning signs:** All slippage readings are exactly 0.0 bps.

### Pitfall 2: Division by Zero in Fill Rate
**What goes wrong:** `fill_rate = orders_filled / orders_submitted` when no orders submitted.
**Why it happens:** Cycles with no signals produce zero orders.
**How to avoid:** Guard with `max(orders_submitted, 1)` or define fill_rate = 1.0 when no orders (no orders = no failures).

### Pitfall 3: Gate Evaluation Before Sufficient Data
**What goes wrong:** GoNoGoReporter returns PROCEED after 1 day of data.
**Why it happens:** Thresholds are met trivially with sparse data.
**How to avoid:** CONTEXT.md locks minimum 5 trading days. Gate MUST return DEFER if insufficient data, regardless of metric values.

### Pitfall 4: Drawdown Z-Score with Insufficient History
**What goes wrong:** Z-score computation with < 3 data points produces meaningless results.
**Why it happens:** Rolling window needs warm-up period.
**How to avoid:** Only compute z-score after window has >= 3 entries. Skip anomaly check during warm-up.

### Pitfall 5: Thread Safety in Slippage Buffer
**What goes wrong:** `_submit_order` is called from APScheduler thread; `on_cycle_complete` also from APScheduler thread.
**Why it happens:** Both run in the same `_strategy_cycle` call, so they're actually sequential within a single thread. But must verify this assumption.
**How to avoid:** Both `_submit_order` and `_strategy_cycle` run on the same APScheduler executor thread (strategy_cycle calls _submit_order). No concurrent access. Document this assumption.

### Pitfall 6: Migration Ordering
**What goes wrong:** Migration 005 depends on 004 but existing 004 may not be applied.
**Why it happens:** Migration 004 (daily_equity_snapshots) is in the untracked files list.
**How to avoid:** Ensure `down_revision = "004"` in migration 005. Test with `alembic upgrade head`.

## Code Examples

### sandbox_metrics Table Schema (Migration 005)
```python
# alembic/versions/005_sandbox_metrics.py
def upgrade() -> None:
    op.create_table(
        "sandbox_metrics",
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_id", sa.String(10), nullable=False),
        sa.Column("trade_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pnl_rub", sa.Numeric(14, 4), nullable=True),
        sa.Column("equity_rub", sa.Numeric(14, 4), nullable=False),
        sa.Column("fill_rate", sa.Numeric(5, 4), nullable=True),
        sa.Column("uptime_cycles", sa.Integer, nullable=False, server_default="0"),
        sa.Column("signals_generated", sa.Integer, nullable=False, server_default="0"),
        sa.Column("errors_caught", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_slippage_bps", sa.Numeric(8, 2), nullable=True),
        sa.Column("avg_slippage_bps", sa.Numeric(8, 2), nullable=True),
        sa.Column("drawdown_pct", sa.Numeric(7, 4), nullable=True),
        sa.PrimaryKeyConstraint("timestamp", "market_id"),
    )
    op.execute(
        "SELECT create_hypertable('sandbox_metrics', 'timestamp', if_not_exists => TRUE)"
    )
```

### ORM Model
```python
# In src/finalayze/core/models.py
class SandboxMetricRow(Base):
    """Per-cycle sandbox metrics persisted to TimescaleDB hypertable."""
    __tablename__ = "sandbox_metrics"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pnl_rub: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    equity_rub: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    fill_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    uptime_cycles: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    signals_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors_caught: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_slippage_bps: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    avg_slippage_bps: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(7, 4))
```

### gate_thresholds.yaml Format
```yaml
# config/gate_thresholds.yaml
# Generated by scripts/derive_gate_thresholds.py from walk-forward history
gate:
  min_sandbox_days: 5
  criteria:
    uptime_pct:
      threshold: 99.0
      critical: true
      source: "p10 of backtest uptime distribution"
    fill_rate_pct:
      threshold: 95.0
      critical: true
      source: "p10 of fill rate distribution"
    max_drawdown_pct:
      threshold: 5.0
      critical: true
      source: "p90 of wf_max_drawdown from history.jsonl"
    min_trades_5d:
      threshold: 5
      critical: false
      source: "p10 of trade_count / periods"
    signal_frequency_per_day:
      threshold: 1.0
      critical: false
      source: "minimum viable signal generation"
    critical_errors_pct:
      threshold: 1.0
      critical: true
      source: "< 1% error rate in cycles"
    max_slippage_bps:
      threshold: 50.0
      critical: false
      source: "p95 of observed slippage"
    signal_divergence_pct:
      threshold: 50.0
      critical: false
      source: "sandbox vs backtest signal agreement"
```

### TelegramAlerter Extension
```python
# Added to alerts.py TelegramAlerter class
def on_anomaly_detected(
    self, metric: str, value: float, threshold: float
) -> None:
    """Alert on sandbox anomaly detection."""
    text = (
        f"\U0001f6a8 Sandbox anomaly: <b>{metric}</b> "
        f"= <code>{value:.2f}</code> (threshold: {threshold:.2f})"
    )
    self.send_alert(text, priority=AlertPriority.CRITICAL)

def on_go_nogo_decision(self, verdict: str, reason: str) -> None:
    """Alert on go/no-go gate evaluation result."""
    emoji = {"PROCEED": "\u2705", "DEFER": "\u23f3", "ABORT": "\u274c"}.get(verdict, "\u2753")
    text = f"{emoji} Go/No-Go: <b>{verdict}</b>\n{reason}"
    self.send_alert(text, priority=AlertPriority.IMPORTANT)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| JSONL file logging only | TimescaleDB hypertable persistence | This phase | Queryable metrics, retention policies, aggregation |
| No slippage tracking | Per-order slippage in bps | This phase | Execution quality measurement |
| Informal go/no-go | 8-criterion structured report | This phase | Reproducible production readiness assessment |
| Hardcoded thresholds | Backtest-derived percentile thresholds | This phase | Data-driven, adjustable gate criteria |

## Open Questions

1. **Signal Divergence Measurement**
   - What we know: GoNoGoReporter needs to evaluate "signal divergence < 50%"
   - What's unclear: How to compare sandbox signals vs backtest signals in a meaningful way (different time periods, different data)
   - Recommendation: Define divergence as "% of sandbox signals that would NOT have been generated by the same strategy on historical data for the same period" -- but this requires replaying backtest on the same dates. Simpler alternative: compare signal frequency distribution (signals/day) between sandbox and backtest. If sandbox generates 2x or 0.5x the signals of backtest, divergence is high.

2. **Threshold Derivation from history.jsonl**
   - What we know: history.jsonl has `wf_sharpe`, `wf_max_drawdown`, `trade_count`, `verdict`
   - What's unclear: It doesn't have fill_rate, uptime, slippage, or signal_frequency
   - Recommendation: For metrics NOT in history.jsonl, use reasonable defaults documented in gate_thresholds.yaml with `source: "engineering default"`. Only `max_drawdown_pct` and `trade_count` can be genuinely backtest-derived.

3. **Drawdown Computation Baseline**
   - What we know: `_compute_drawdown_pct` uses `_peak_equity_rub` which resets on restart
   - What's unclear: Should sandbox_metrics store peak equity so GoNoGoReporter can compute max drawdown over the full observation period?
   - Recommendation: Store `drawdown_pct` per cycle in sandbox_metrics. GoNoGoReporter computes max(drawdown_pct) over the observation window. Peak equity is TradingLoop's concern.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_sandbox_monitor.py tests/unit/test_go_no_go.py tests/unit/test_anomaly_detector.py -x` |
| Full suite command | `uv run pytest tests/ -x --timeout=60` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MON-01 | SandboxMonitorService persists metrics to DB | unit | `uv run pytest tests/unit/test_sandbox_monitor.py::test_persist_metrics -x` | Wave 0 |
| MON-01 | CycleMetrics dataclass has all required fields | unit | `uv run pytest tests/unit/test_sandbox_monitor.py::test_cycle_metrics_fields -x` | Wave 0 |
| MON-02 | Slippage capture computes bps correctly | unit | `uv run pytest tests/unit/test_sandbox_monitor.py::test_slippage_computation -x` | Wave 0 |
| MON-02 | Slippage recorded to SandboxMonitorService buffer | unit | `uv run pytest tests/unit/test_sandbox_monitor.py::test_slippage_recording -x` | Wave 0 |
| MON-04 | Drawdown z-score anomaly triggers alert | unit | `uv run pytest tests/unit/test_anomaly_detector.py::test_drawdown_zscore_alert -x` | Wave 0 |
| MON-04 | Fill rate below 90% triggers alert | unit | `uv run pytest tests/unit/test_anomaly_detector.py::test_fill_rate_alert -x` | Wave 0 |
| MON-04 | Slippage above 50bps triggers alert | unit | `uv run pytest tests/unit/test_anomaly_detector.py::test_slippage_alert -x` | Wave 0 |
| MON-04 | 30-minute cooldown prevents duplicate alerts | unit | `uv run pytest tests/unit/test_anomaly_detector.py::test_cooldown -x` | Wave 0 |
| GATE-01 | GoNoGoReporter evaluates 8 criteria | unit | `uv run pytest tests/unit/test_go_no_go.py::test_evaluate_all_criteria -x` | Wave 0 |
| GATE-01 | DEFER returned when < 5 sandbox days | unit | `uv run pytest tests/unit/test_go_no_go.py::test_defer_insufficient_data -x` | Wave 0 |
| GATE-01 | ABORT on critical criterion failure | unit | `uv run pytest tests/unit/test_go_no_go.py::test_abort_critical_fail -x` | Wave 0 |
| GATE-01 | PROCEED when all criteria pass | unit | `uv run pytest tests/unit/test_go_no_go.py::test_proceed_all_pass -x` | Wave 0 |
| GATE-02 | Thresholds loaded from gate_thresholds.yaml | unit | `uv run pytest tests/unit/test_go_no_go.py::test_load_yaml_thresholds -x` | Wave 0 |
| GATE-02 | derive_gate_thresholds script reads history.jsonl | unit | `uv run pytest tests/unit/test_go_no_go.py::test_derive_thresholds -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_sandbox_monitor.py tests/unit/test_go_no_go.py tests/unit/test_anomaly_detector.py -x`
- **Per wave merge:** `uv run pytest tests/ -x --timeout=60`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_sandbox_monitor.py` -- covers MON-01, MON-02
- [ ] `tests/unit/test_anomaly_detector.py` -- covers MON-04
- [ ] `tests/unit/test_go_no_go.py` -- covers GATE-01, GATE-02

## Sources

### Primary (HIGH confidence)
- `src/finalayze/core/trading_loop.py` -- _strategy_cycle finally block (lines 1004-1052), _submit_order (lines 1445-1507), _persist_equity_snapshots (lines 1803-1847)
- `src/finalayze/core/alerts.py` -- TelegramAlerter API, AlertPriority enum, TelegramMessageQueue
- `src/finalayze/core/models.py` -- ORM model patterns (PortfolioSnapshot, DailyEquitySnapshot)
- `src/finalayze/core/validation_logger.py` -- CycleLogEntry dataclass pattern
- `src/finalayze/api/metrics.py` -- MetricsCollector static facade pattern
- `alembic/versions/003_portfolio_snapshots.py` -- TimescaleDB hypertable migration pattern
- `config/settings.py` -- Settings class, telegram_bot_token/chat_id fields
- `results/iterations/history.jsonl` -- wf_sharpe, wf_max_drawdown, trade_count fields
- `.planning/phases/16-sandbox-monitoring-and-go-no-go-gate/16-CONTEXT.md` -- locked decisions
- `.planning/STATE.md` -- "Monitoring services standalone" decision, sandbox slippage blocker

### Secondary (MEDIUM confidence)
- TimescaleDB hypertable documentation -- create_hypertable with if_not_exists pattern

### Tertiary (LOW confidence)
- Signal divergence measurement approach -- no established pattern in codebase, recommendation is speculative

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, zero new dependencies
- Architecture: HIGH -- follows established codebase patterns (standalone service, post-cycle hook, async DB persistence)
- Pitfalls: HIGH -- identified from direct code inspection of _submit_order and _strategy_cycle
- Gate thresholds: MEDIUM -- history.jsonl only has 3 of 8 needed metrics; remaining thresholds must be engineering defaults
- Signal divergence: LOW -- no established pattern; recommend simpler frequency-based comparison

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable domain -- internal services, no external API dependencies)
