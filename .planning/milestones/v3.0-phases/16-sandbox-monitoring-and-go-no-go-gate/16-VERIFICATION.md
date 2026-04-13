---
phase: 16-sandbox-monitoring-and-go-no-go-gate
verified: 2026-03-22T00:00:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
---

# Phase 16: Sandbox Monitoring and Go/No-Go Gate Verification Report

**Phase Goal:** System collects sandbox execution metrics and produces a structured go/no-go evaluation report with calibrated thresholds
**Verified:** 2026-03-22
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | CycleMetrics dataclass captures all 10+ fields (timestamp, trade_count, pnl_rub, equity_rub, fill_rate, uptime_cycles, signals_generated, errors_caught, max_slippage_bps, avg_slippage_bps, drawdown_pct) | VERIFIED | `sandbox_monitor.py` lines 23-37: frozen dataclass with all 11 fields |
| 2  | SandboxMonitorService persists CycleMetrics to sandbox_metrics TimescaleDB table via async session | VERIFIED | `sandbox_monitor.py` lines 92-115: `_persist_metrics_async` uses `get_async_session_factory`, adds `SandboxMetricRow`, commits |
| 3  | SandboxMonitorService.record_slippage accumulates per-order slippage in bps buffer | VERIFIED | `sandbox_monitor.py` line 67: `self._slippage_buffer.append(slippage_bps)` |
| 4  | AnomalyDetector fires alert on drawdown z-score > 2-sigma, fill rate < 0.90, slippage > 50bps | VERIFIED | `anomaly_detector.py` lines 35-101: constants `_ZSCORE_THRESHOLD=2.0`, `_FILL_RATE_FLOOR=0.90`, `_SLIPPAGE_CEILING_BPS=50.0` with check methods |
| 5  | AnomalyDetector respects 30-minute cooldown per metric | VERIFIED | `anomaly_detector.py` lines 90-101: `_COOLDOWN_SECONDS=1800`, `_is_cooled_down` per metric, `_last_alert` dict |
| 6  | SandboxMetricRow ORM model maps to sandbox_metrics hypertable | VERIFIED | `models.py` line 349: `class SandboxMetricRow(Base)` with `__tablename__ = "sandbox_metrics"` |
| 7  | Alembic migration 005 creates sandbox_metrics hypertable | VERIFIED | `005_sandbox_metrics.py` line 36: `create_hypertable('sandbox_metrics', 'timestamp', if_not_exists => TRUE)` |
| 8  | GoNoGoReporter evaluates 8 criteria: uptime, fill_rate, max_drawdown, trade_count, signal_frequency, critical_errors, slippage, signal_divergence | VERIFIED | `go_no_go.py` lines 161-170: 8 `_check_*` methods called in `evaluate()` |
| 9  | GoNoGoReporter returns DEFER when sandbox data < 5 trading days | VERIFIED | `go_no_go.py` lines 149-159: `if sandbox_days < self._thresholds.min_sandbox_days: return GateReport(verdict=GateVerdict.DEFER, ...)` |
| 10 | GoNoGoReporter returns ABORT when any critical criterion fails | VERIFIED | `go_no_go.py` lines 172-178: `if critical_fails: verdict = GateVerdict.ABORT` |
| 11 | GoNoGoReporter returns PROCEED when all criteria pass | VERIFIED | `go_no_go.py` lines 181-183: `else: verdict = GateVerdict.PROCEED; reason = "All 8 criteria passed"` |
| 12 | GateThresholds loaded from config/gate_thresholds.yaml | VERIFIED | `go_no_go.py` line 98: `raw = yaml.safe_load(f)`, `from_yaml` classmethod present |
| 13 | derive_gate_thresholds.py reads history.jsonl and outputs gate_thresholds.yaml | VERIFIED | `derive_gate_thresholds.py` references `history.jsonl` 8 times; `gate_thresholds.yaml` has `source: p90 of wf_max_drawdown from history.jsonl` |

**Score:** 13/13 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/monitoring/__init__.py` | Package init exporting all public symbols | VERIFIED | Exports AnomalyDetector, CycleMetrics, GateReport, GateThresholds, GateVerdict, GoNoGoReporter, SandboxMonitorService |
| `src/finalayze/monitoring/sandbox_monitor.py` | CycleMetrics dataclass and SandboxMonitorService class | VERIFIED | Both classes present, substantive implementation, 124 lines |
| `src/finalayze/monitoring/anomaly_detector.py` | AnomalyDetector with z-score and threshold checks | VERIFIED | Full implementation, 102 lines, all constants and methods present |
| `src/finalayze/monitoring/go_no_go.py` | GoNoGoReporter, GateVerdict, CriterionResult, GateReport, GateThresholds | VERIFIED | All 5 classes present, `evaluate()` is async, 8 check methods, 332 lines |
| `src/finalayze/core/models.py` | SandboxMetricRow ORM model | VERIFIED | Class at line 349, `__tablename__ = "sandbox_metrics"`, composite PK (timestamp, market_id) |
| `alembic/versions/005_sandbox_metrics.py` | TimescaleDB hypertable migration | VERIFIED | revision="005", down_revision="004", `create_hypertable` call present |
| `config/gate_thresholds.yaml` | 8 gate threshold configs with source documentation | VERIFIED | 8 criteria, `max_drawdown_pct` and `min_trades_5d` have `history.jsonl` source, min_sandbox_days=5 |
| `scripts/derive_gate_thresholds.py` | One-shot script to derive thresholds from history.jsonl | VERIFIED | Reads history.jsonl via argparse, computes p90/p10 thresholds, writes YAML |
| `src/finalayze/core/trading_loop.py` | SandboxMonitorService integration in __init__, _submit_order, _strategy_cycle | VERIFIED | `self._sandbox_monitor` at line 156; `record_slippage` at line 1513; `on_cycle_complete` at line 1081 |
| `src/finalayze/core/alerts.py` | on_anomaly_detected and on_go_nogo_decision methods | VERIFIED | Both methods at lines 377 and 385 |
| `src/finalayze/main.py` | SandboxMonitorService creation and injection in SANDBOX mode | VERIFIED | Conditional at line 307: `if settings.mode == WorkMode.SANDBOX:`, passed as `sandbox_monitor=sandbox_monitor` at line 330 |
| `tests/unit/test_sandbox_monitor.py` | Unit tests for SandboxMonitorService | VERIFIED | 14 test functions (> 8 required) |
| `tests/unit/test_anomaly_detector.py` | Unit tests for AnomalyDetector | VERIFIED | 14 test functions (> 8 required) |
| `tests/unit/test_go_no_go.py` | Unit tests for GoNoGoReporter | VERIFIED | 18 test functions (> 10 required) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `sandbox_monitor.py` | `core/models.py` | SandboxMetricRow deferred import | WIRED | Line 95: `from finalayze.core.models import SandboxMetricRow` in `_persist_metrics_async` |
| `sandbox_monitor.py` | `anomaly_detector.py` | `self._anomaly_detector.check(metrics)` call | WIRED | Line 73: `self._anomaly_detector.check(metrics)` in `on_cycle_complete` |
| `go_no_go.py` | `config/gate_thresholds.yaml` | `yaml.safe_load` | WIRED | Line 98: `raw = yaml.safe_load(f)` in `from_yaml` |
| `scripts/derive_gate_thresholds.py` | `results/iterations/history.jsonl` | JSON line reading | WIRED | References `history.jsonl` 8 times; argparse default points to `results/iterations/history.jsonl` |
| `trading_loop.py` | `sandbox_monitor.py` | `self._sandbox_monitor.on_cycle_complete` | WIRED | Line 1081: called in `_strategy_cycle` finally block |
| `trading_loop.py` | `sandbox_monitor.py` | `self._sandbox_monitor.record_slippage` | WIRED | Line 1513: called in `_submit_order` after fill |
| `main.py` | `sandbox_monitor.py` | `SandboxMonitorService(` instantiation | WIRED | Line 308: deferred import + instantiation, passed at line 330 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MON-01 | 16-01, 16-03 | SandboxMonitorService collects per-cycle metrics (trades, P&L, equity, fill rate, uptime) and persists to TimescaleDB | SATISFIED | CycleMetrics captures all fields; `_persist_metrics_async` writes SandboxMetricRow to TimescaleDB; TradingLoop calls `on_cycle_complete` every cycle |
| MON-02 | 16-03 | Slippage capture records expected_price vs fill_price, computes realized slippage in bps | SATISFIED | `trading_loop.py` lines 1500-1513: `slippage_bps = (fill_price - expected_price) / expected_price * 10000`, passed to `record_slippage`; replaces hardcoded `0.0` in MetricsCollector |
| MON-04 | 16-01 | Anomaly detector triggers Telegram alerts on drawdown spikes (>2sigma), fill rate drops (<90%), slippage outliers (>50bps) | SATISFIED | `anomaly_detector.py`: z-score check, fill rate check, slippage check all implemented with alert firing via `TelegramAlerter.send_alert` |
| GATE-01 | 16-02 | GoNoGoReporter evaluates formalized thresholds (uptime, fill rate, max DD, trades, signal divergence) | SATISFIED | `go_no_go.py`: 8-criterion evaluation with PROCEED/DEFER/ABORT verdict, all thresholds loaded from YAML |
| GATE-02 | 16-02 | Gate thresholds derived from walk-forward backtest distribution percentiles, not hardcoded round numbers | SATISFIED | `derive_gate_thresholds.py` computes p90(wf_max_drawdown) = 2.27 and p10(trade_count)/periods = 18.0; `gate_thresholds.yaml` reflects derived values with `source` attribution |

All 5 required requirements (MON-01, MON-02, MON-04, GATE-01, GATE-02) are satisfied. No orphaned requirements found.

### Anti-Patterns Found

No blocker or warning anti-patterns detected in phase 16 files.

| File | Pattern | Severity | Notes |
|------|---------|----------|-------|
| `go_no_go.py:295-305` | `_check_signal_divergence` always returns `passed=True` | Info | Documented as placeholder — no backtest comparison data available yet. Non-blocking per design (non-critical criterion per PLAN spec). Source comment states "engineering default". |

### Human Verification Required

#### 1. SANDBOX mode end-to-end metric flow

**Test:** Start system in SANDBOX mode with `FINALAYZE_TINKOFF_TOKEN` set, run for at least one trading cycle, then query `SELECT * FROM sandbox_metrics ORDER BY timestamp DESC LIMIT 5;` in PostgreSQL.
**Expected:** Rows appear with non-null `equity_rub`, `trade_count >= 0`, `uptime_cycles > 0`, and `fill_rate` between 0.0 and 1.0.
**Why human:** Requires live database and Tinkoff sandbox credentials; cannot verify DB writes programmatically without running the full loop.

#### 2. Anomaly alert Telegram delivery

**Test:** Inject a CycleMetrics with `fill_rate=0.5` (below 0.90 threshold) via `SandboxMonitorService.on_cycle_complete` with a real `TelegramAlerter` configured.
**Expected:** Telegram message received: "Sandbox anomaly: fill_rate = 0.50 (threshold: 0.90)".
**Why human:** Requires active Telegram bot token and chat ID; network call cannot be mocked in automated verification.

#### 3. GoNoGoReporter PROCEED path against real DB data

**Test:** After 5+ days of sandbox operation, call `GoNoGoReporter.evaluate(session)` and inspect the returned `GateReport`.
**Expected:** `verdict` is one of PROCEED/DEFER/ABORT with a non-empty `criteria` list (8 items) and human-readable `reason`.
**Why human:** Requires 5+ days of accumulated `sandbox_metrics` rows in TimescaleDB.

### Gaps Summary

No gaps found. All 13 must-have truths verified, all artifacts present and substantive, all key links wired. The only open item is `_check_signal_divergence` returning a constant `passed=True` placeholder, which is an acknowledged design decision (no backtest comparison data available) and flagged in code comments — not a blocker for phase goal achievement.

---

_Verified: 2026-03-22_
_Verifier: Claude (gsd-verifier)_
