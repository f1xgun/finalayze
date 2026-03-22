---
phase: 22-dependency-layer-cleanup
verified: 2026-03-23T00:00:00Z
status: passed
score: 6/6 must-haves verified
gaps: []
human_verification: []
---

# Phase 22: Dependency Layer Cleanup Verification Report

**Phase Goal:** core/ contains only Layer 0 types and schemas; orchestration logic lives in a dedicated module; dead infrastructure is removed
**Verified:** 2026-03-23
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | trading_loop.py and bond_cycle.py are importable from finalayze.orchestration | VERIFIED | `from finalayze.orchestration.trading_loop import TradingLoop` and `from finalayze.orchestration.bond_cycle import BondCycleProcessor` both succeed at runtime. Module identity confirmed: `TradingLoop is TL2 -> True`. |
| 2 | telegram_bot.py and alerts.py reside under api/; core/ does not contain API/dashboard layer code | VERIFIED | `src/finalayze/api/alerts.py` and `src/finalayze/api/telegram_bot.py` exist with full implementations. core/ shims are 14-line sys.modules aliases. `from finalayze.api.alerts import TelegramAlerter` resolves correctly. |
| 3 | core/ shims resolve to canonical modules (backward compat) | VERIFIED | sys.modules aliasing approach confirmed. `TelegramAlerter is TA2 -> True`. All four shims (trading_loop.py, bond_cycle.py, alerts.py, telegram_bot.py) are 14-16 line wrappers that register the canonical module under the old name. |
| 4 | MetricsCollector is injected into TradingLoop via constructor — no deferred imports inside method bodies | VERIFIED | `TradingLoop.__init__` accepts `metrics_collector` as last parameter. Only `TYPE_CHECKING` guard import of MetricsCollector at line 45. Zero occurrences of `from finalayze.api.metrics import MetricsCollector` inside method bodies. main.py passes `MetricsCollector` at construction site (line 463). |
| 5 | backtest/ and monitoring/ modules have layer assignments documented | VERIFIED | `src/finalayze/monitoring/CLAUDE.md` exists, declares "Layer 6 -- Monitoring". `src/finalayze/backtest/CLAUDE.md` updated to "Cross-cutting test infrastructure (Layer 4-5)" with explicit import rules. |
| 6 | Dead event bus streams removed; stub API endpoints return 501 | VERIFIED | `EventBus` has only `STREAM_COUPONS = "coupons"`. STREAM_MARKET_DATA, STREAM_SIGNALS, STREAM_EXECUTION not present in any src/ file. All 6 stub endpoints (/signals, /strategies/performance, /trades, /trades/analytics, /news, /ml/status) return HTTP 501 with `{"detail": "Not yet implemented"}` — confirmed via FastAPI TestClient. |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/orchestration/__init__.py` | New orchestration module package | VERIFIED | Exists (8 lines, minimal package init) |
| `src/finalayze/orchestration/trading_loop.py` | TradingLoop class in correct layer | VERIFIED | Full implementation, 2000+ lines. TYPE_CHECKING guard for MetricsCollector. No deferred api.metrics imports. |
| `src/finalayze/orchestration/bond_cycle.py` | BondCycleProcessor in correct layer | VERIFIED | Full implementation. |
| `src/finalayze/orchestration/CLAUDE.md` | Layer documentation for orchestration module | VERIFIED | Declares "Layer 5 -- Orchestration". Documents injection pattern for L6 dependencies. |
| `src/finalayze/api/alerts.py` | TelegramAlerter in correct Layer 6 location | VERIFIED | Full implementation, not a shim. |
| `src/finalayze/api/telegram_bot.py` | TelegramBotHandler in correct Layer 6 location | VERIFIED | Full implementation, not a shim. |
| `src/finalayze/core/trading_loop.py` | Backward-compat shim | VERIFIED | 16-line sys.modules alias to orchestration.trading_loop |
| `src/finalayze/core/bond_cycle.py` | Backward-compat shim | VERIFIED | 16-line sys.modules alias to orchestration.bond_cycle |
| `src/finalayze/core/alerts.py` | Backward-compat shim | VERIFIED | 14-line sys.modules alias to api.alerts |
| `src/finalayze/core/telegram_bot.py` | Backward-compat shim | VERIFIED | 14-line sys.modules alias to api.telegram_bot |
| `src/finalayze/monitoring/CLAUDE.md` | Layer 6 documentation for monitoring | VERIFIED | Exists. Documents HealthMonitor, SandboxMonitorService, AnomalyDetector, GoNoGoReporter. |
| `src/finalayze/backtest/CLAUDE.md` | Updated layer documentation for backtest | VERIFIED | Declares "Cross-cutting test infrastructure (Layer 4-5)" with explicit no-L6 import rule. |
| `src/finalayze/core/events.py` | EventBus with only STREAM_COUPONS | VERIFIED | Line 22: `STREAM_COUPONS = "coupons"`. No other STREAM_ constants present. |
| `src/finalayze/api/v1/signals.py` | 501 responses | VERIFIED | Both endpoints raise HTTPException(501) |
| `src/finalayze/api/v1/trades.py` | 501 responses | VERIFIED | All three endpoints (list, analytics, get_trade) raise HTTPException(501) |
| `src/finalayze/api/v1/news.py` | 501 response | VERIFIED | list_news raises HTTPException(501) |
| `src/finalayze/api/v1/ml.py` | 501 response | VERIFIED | ml_status raises HTTPException(501) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/finalayze/core/trading_loop.py` | `src/finalayze/orchestration/trading_loop.py` | sys.modules alias | WIRED | `sys.modules[__name__] = _canonical` — fully transparent alias |
| `src/finalayze/core/alerts.py` | `src/finalayze/api/alerts.py` | sys.modules alias | WIRED | `sys.modules[__name__] = _canonical` |
| `src/finalayze/main.py` | `src/finalayze/orchestration/trading_loop.py` | deferred import + MetricsCollector injection | WIRED | Line 221: `from finalayze.orchestration.trading_loop import TradingLoop`. Line 463: `metrics_collector=MetricsCollector` passed to constructor. |
| `src/finalayze/main.py` | `src/finalayze/api/alerts.py` | deferred import | WIRED | Line 220: `from finalayze.api.alerts import TelegramAlerter` |
| `src/finalayze/data/bond_discovery.py` | `src/finalayze/core/events.py` | STREAM_COUPONS | WIRED | Line 285: `await self._event_bus.publish(EventBus.STREAM_COUPONS, event)` |
| `src/finalayze/core/kill_switch.py` | `src/finalayze/api/alerts.py` | TYPE_CHECKING import | WIRED | Lines 27-29 under `if TYPE_CHECKING:` guard |
| `src/finalayze/core/layer_ledger.py` | `src/finalayze/api/alerts.py` | TYPE_CHECKING import | WIRED | Lines 21-22 under `if TYPE_CHECKING:` guard |

### Data-Flow Trace (Level 4)

Step 7b does not apply here — this phase is a structural refactor (file moves, dependency injection, dead code removal). No new data-rendering components were introduced.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| orchestration imports resolve | `python -c "from finalayze.orchestration.trading_loop import TradingLoop; from finalayze.orchestration.bond_cycle import BondCycleProcessor; print('OK')"` | All imports OK, module identity verified | PASS |
| backward-compat shims work | `python -c "from finalayze.core.trading_loop import TradingLoop; TradingLoop is TL2 -> True"` | Module identity confirmed | PASS |
| MetricsCollector injected (not deferred) | `grep "from finalayze.api.metrics import MetricsCollector" orchestration/trading_loop.py` — only line 45 (TYPE_CHECKING) | Zero method-body imports found | PASS |
| TradingLoop constructor has metrics_collector | `inspect.signature(TradingLoop.__init__).parameters` | `metrics_collector` present as last param | PASS |
| EventBus has only STREAM_COUPONS | `python -c "from finalayze.core.events import EventBus; assert not hasattr(EventBus, 'STREAM_MARKET_DATA')"` | Assertion passes | PASS |
| Stub endpoints return 501 | FastAPI TestClient on all 6 paths | All 6 paths return 501 with `{"detail": "Not yet implemented"}` | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| LAYER-01 | 22-01-PLAN.md | trading_loop.py and bond_cycle.py moved from core/ to orchestration/ | SATISFIED | Files at `src/finalayze/orchestration/`. Core shims transparent. REQUIREMENTS.md: checked. |
| LAYER-02 | 22-01-PLAN.md | telegram_bot.py and alerts.py moved from core/ to api/ | SATISFIED | Files at `src/finalayze/api/`. Core shims transparent. REQUIREMENTS.md: checked. |
| LAYER-03 | 22-02-PLAN.md | MetricsCollector injected via constructor — no direct L6 import in trading_loop | SATISFIED | Constructor param `metrics_collector` verified. Only TYPE_CHECKING guard import at line 45. main.py wires it. |
| LAYER-04 | 22-02-PLAN.md | backtest/ and monitoring/ have documented layer assignments | SATISFIED | monitoring/CLAUDE.md: Layer 6. backtest/CLAUDE.md: Layer 4-5 cross-cutting, explicit no-L6 rule. |
| DEAD-01 | 22-03-PLAN.md | Dead event bus streams removed or wired to real consumers | SATISFIED | EventBus has only STREAM_COUPONS. Zero references to STREAM_MARKET_DATA/SIGNALS/EXECUTION in src/. test_events.py asserts they do NOT exist. |
| DEAD-02 | 22-03-PLAN.md | Stub API endpoints return 501 Not Implemented with clear message | SATISFIED | All 6 stub endpoints return HTTP 501 with `{"detail": "Not yet implemented"}`. Pydantic response models preserved for OpenAPI docs. |

No orphaned requirements: all 6 IDs appear in PLAN frontmatter and REQUIREMENTS.md maps them to Phase 22.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/finalayze/orchestration/trading_loop.py` | 1884 | `pct = float(qty) * 0.01  # placeholder` | Info | Internal comment describing an approximation in `_compute_top_movers`. Not a stub — the method computes and returns real data. No user-visible impact. Pre-existing. |

No blockers or warnings found. The single "placeholder" comment is an internal note about a calculation approximation, not a stub implementation.

### Layer Violation Check (core/ files)

All non-shim core/ files were scanned for upper-layer imports outside TYPE_CHECKING guards:

- `kill_switch.py` — imports from `api.alerts`, `orchestration.trading_loop`, `execution.broker_router`, `risk.circuit_breaker` — all under `if TYPE_CHECKING:` guard. No runtime violations.
- `layer_ledger.py` — imports from `api.alerts`, `markets.instruments` — all under `if TYPE_CHECKING:` guard. No runtime violations.
- All other core/ files (bond_math.py, bond_math_quantlib.py, clock.py, db.py, events.py, exceptions.py, models.py, modes.py, schemas.py, validation_logger.py) — clean, no upper-layer imports.

### Human Verification Required

None. All success criteria are verifiable programmatically and have been confirmed.

### Gaps Summary

No gaps found. All six success criteria from ROADMAP.md are fully satisfied:

1. trading_loop.py and bond_cycle.py are importable from `finalayze.orchestration` — core/ shims delegate transparently.
2. telegram_bot.py and alerts.py reside under `finalayze.api` — core/ shims delegate transparently.
3. MetricsCollector is injected into TradingLoop via constructor — zero deferred imports in method bodies.
4. backtest/ and monitoring/ have CLAUDE.md files with definitive layer assignments.
5. STREAM_MARKET_DATA, STREAM_SIGNALS, STREAM_EXECUTION are removed from EventBus — only STREAM_COUPONS remains, actively used by bond_discovery.py.
6. All six stub endpoints return HTTP 501 Not Implemented with `{"detail": "Not yet implemented"}`.

---

_Verified: 2026-03-23_
_Verifier: Claude (gsd-verifier)_
