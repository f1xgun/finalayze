---
phase: 29-core-stability
verified: 2026-03-30T21:15:00Z
status: human_needed
score: 5/5 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "Strategy cycles complete without 60-min drift — close() now logs 'event_loop_stop_failed' at warning level with resource/error_type/error kwargs; test_close_logs_warning_on_loop_stop_failure passes (5 passed)"
    - "Promtail ships Docker container logs from all 7 containers to Loki — docker_sd_configs filter now includes all 7 containers (app, db, redis, prometheus, loki, promtail, grafana)"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Deploy sandbox and verify Grafana/Loki receives logs from all 7 containers"
    expected: "LogQL query {container=~\"finalayze-sandbox-.*\"} returns results from all 7 containers within seconds of emission"
    why_human: "Requires running Docker stack — cannot verify log ingestion pipeline without live containers"
  - test: "Verify strategy cycle fires within 5 minutes of scheduled time"
    expected: "APScheduler fires strategy_cycle job at scheduled interval without drift; sandbox logs show cycle_start timestamps within 5 minutes of expected schedule"
    why_human: "Requires running sandbox for multiple cycles over real time — cannot test programmatically"
---

# Phase 29: Core Stability Verification Report

**Phase Goal:** Strategy cycles fire reliably within 5 minutes of scheduled time and all container logs are queryable in Grafana/Loki
**Verified:** 2026-03-30T21:15:00Z
**Status:** human_needed
**Re-verification:** Yes — after gap closure (commit 9c86a10)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TinkoffBroker and TinkoffFetcher do not create their own background event loops when grpc_loop is injected | VERIFIED | `tinkoff_broker.py:175-190` — `_run_async` uses `self._grpc_loop` if set; `close()` skips loop stop when `_grpc_loop` is injected (line 155) |
| 2 | A single shared gRPC event loop is created and injected into broker, bond broker, fetcher, and TradingLoop | VERIFIED | `run_sandbox.py` creates `_grpc_loop` once; injected into TinkoffFetcher, TinkoffBroker (4 times: lines checked with grep returning 4 matches), and bond broker reuses via `equity_broker._grpc_loop` at line 567 |
| 3 | BlockingIOError from PollerCompletionQueue is suppressed on the gRPC loop exception handler | VERIFIED | `trading_loop.py:427-433` — `_grpc_exception_handler` checks `isinstance(exc, BlockingIOError)` and returns silently; also in `run_sandbox.py:247-253` |
| 4 | Strategy cycles complete without 60-min drift caused by event loop contention | VERIFIED | Architecture correct; `close()` logs `"event_loop_stop_failed"` at warning level with `resource`, `error_type`, `error` kwargs (line 159-164); `test_close_logs_warning_on_loop_stop_failure` passes (5/5 tests) |
| 5 | Promtail ships Docker container logs from all 7 containers to Loki | VERIFIED | `monitoring/promtail/promtail-config.yml:18-25` — docker_sd_configs filter includes all 7 containers: finalayze-sandbox-app, -db, -redis, -prometheus, -loki, -promtail, -grafana |

**Score:** 5/5 truths verified

### Required Artifacts

**Plan 01 (GRPC-01):**

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/orchestration/trading_loop.py` | `_grpc_loop, _grpc_thread, _run_grpc() method, exception handler` | VERIFIED | Lines 232-234 (fields), 419-441 (`_init_grpc_loop`), 443-452 (`_run_grpc`), 427-433 (BlockingIOError handler), 668-673 (stop cleanup) |
| `src/finalayze/execution/tinkoff_broker.py` | Accepts external grpc_loop parameter, no self-managed loop | VERIFIED | Line 83: `grpc_loop` param; line 96: `self._grpc_loop = grpc_loop`; `_run_async` prefers injected loop; `close()` logs `"event_loop_stop_failed"` at warning with structured kwargs |
| `src/finalayze/data/fetchers/tinkoff_data.py` | Accepts external grpc_loop parameter, no self-managed loop | VERIFIED | `grpc_loop` param present; `self._grpc_loop = grpc_loop`; `_run_async` prefers injected loop |

**Plan 02 (OBS-01, OBS-02):**

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `docker/docker-compose.sandbox.yml` | Promtail volume mounts for container logs and positions persistence | VERIFIED | `/var/lib/docker/containers:/var/lib/docker/containers:ro` present; `promtail_positions:/positions` present; named volume `promtail_positions:` declared |
| `monitoring/promtail/promtail-config.yml` | Correct __path__ relabeling, all 7 containers, low-cardinality labels, JSON-aware drop stage | VERIFIED | `__meta_docker_container_log_path` -> `__path__` relabeling at lines 30-31; all 7 containers in filter values (lines 18-25); `event` only in `json.expressions`, NOT in labels; positions file at `/positions/positions.yaml` |
| `monitoring/loki/loki-config.yml` | 30-day retention config and ingestion rate limits | VERIFIED | `retention_period: 720h`; ingestion limits present; compactor with `retention_enabled: true` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `trading_loop.py` | `tinkoff_broker.py` | grpc_loop constructor parameter | WIRED | `run_sandbox.py` passes `grpc_loop=_grpc_loop` to TinkoffBroker; 4 injection sites confirmed |
| `trading_loop.py` | `tinkoff_data.py` | grpc_loop constructor parameter | WIRED | `run_sandbox.py` passes `grpc_loop=_grpc_loop` to TinkoffFetcher |
| `docker-compose.sandbox.yml` | `monitoring/promtail/promtail-config.yml` | volume mount config file | WIRED | `../monitoring/promtail/promtail-config.yml:/etc/promtail/config.yml:ro` |
| `monitoring/promtail/promtail-config.yml` | `monitoring/loki/loki-config.yml` | push URL http://loki:3100 | WIRED | Line 9: `url: http://loki:3100/loki/api/v1/push` |

### Data-Flow Trace (Level 4)

Not applicable — phase produces infrastructure config and event loop wiring, not data-rendering components.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Trading loop tests pass (gRPC loop isolation) | `uv run pytest tests/unit/core/test_trading_loop.py --no-cov -q` | 18 passed | PASS |
| TinkoffBroker close() test — warning level log with structured kwargs | `uv run pytest tests/unit/test_tinkoff_broker_close.py --no-cov -q` | 5 passed | PASS |
| Broker unit tests — no regression | `uv run pytest tests/unit/test_broker.py --no-cov -q` | All passed (45 total with above) | PASS |
| Promtail container coverage — all 7 containers in filter | `grep -c "finalayze-sandbox-" monitoring/promtail/promtail-config.yml` | 8 matches (7 filter + 1 drop selector) | PASS |
| Loki retention_period set to 720h | `grep "retention_period" monitoring/loki/loki-config.yml` | `retention_period: 720h` | PASS |
| promtail_positions named volume declared | `grep -c "promtail_positions" docker/docker-compose.sandbox.yml` | 2 matches (mount + definition) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GRPC-01 | 29-01-PLAN.md | gRPC PollerCompletionQueue runs on a dedicated event loop isolated from APScheduler — no BlockingIOError flooding the main asyncio loop, strategy cycles fire within 5 min | SATISFIED | `_grpc_loop` dedicated loop in TradingLoop; `BlockingIOError` suppressed; injected into all gRPC consumers; close() warning logging verified by passing test suite |
| OBS-01 | 29-02-PLAN.md | Promtail ships Docker container logs to Loki — `/var/lib/docker/containers` mounted, JSON log format parsed correctly, all 7 containers covered | SATISFIED | Volume mount present; `__path__` resolved from `__meta_docker_container_log_path`; all 7 container names in docker_sd_configs filter; JSON parsing configured |
| OBS-02 | 29-02-PLAN.md | Loki retains queryable logs for 30 days — dashboard queries return results for all 7 containers | SATISFIED (programmatic) / NEEDS HUMAN (live) | 30-day retention (720h) configured; compactor with `retention_enabled: true` enforces it; ingestion limits set. Live query verification requires running Docker stack. |

### Anti-Patterns Found

No blockers found. All previously identified blockers are resolved:

- `event_loop_stop_failed_on_close` at debug level: RESOLVED — now `"event_loop_stop_failed"` at warning level with structured kwargs matching the test
- Promtail single-container filter: RESOLVED — all 7 containers added to docker_sd_configs filter

### Human Verification Required

#### 1. All 7 Container Logs in Grafana

**Test:** Restart sandbox with `docker compose up`, open Grafana at `localhost:3000`, query LogQL `{container=~"finalayze-sandbox-.*"}`.
**Expected:** Results appear from all 7 containers (db, redis, app, prometheus, loki, promtail, grafana) within seconds.
**Why human:** Requires running Docker stack — log ingestion pipeline cannot be verified without live containers.

#### 2. Strategy Cycle Timing (Within 5 Minutes)

**Test:** Run sandbox for 30 minutes, check `strategy_cycle_summary` log events in Grafana.
**Expected:** Cycle timestamps show each fire within 5 minutes of the scheduled APScheduler interval — no 60-minute drift observed.
**Why human:** Requires multi-cycle real-time sandbox run to confirm drift elimination.

### Gaps Summary

No gaps remain. Both previously identified gaps are closed:

**Gap 1 (CLOSED) — close() log regression:** `TinkoffBroker.close()` now logs `"event_loop_stop_failed"` at warning level with structured kwargs `(resource="event_loop", error_type=..., error=...)`. The test `test_close_logs_warning_on_loop_stop_failure` passes (5 tests, 0 failures).

**Gap 2 (CLOSED) — Promtail single-container coverage:** `monitoring/promtail/promtail-config.yml` docker_sd_configs filter now includes all 7 containers: finalayze-sandbox-app, finalayze-sandbox-db, finalayze-sandbox-redis, finalayze-sandbox-prometheus, finalayze-sandbox-loki, finalayze-sandbox-promtail, finalayze-sandbox-grafana. The success criterion OBS-01 is fully satisfied at the config level.

All automated checks pass. Two items require live sandbox validation before the phase can be fully signed off.

---

_Verified: 2026-03-30T21:15:00Z_
_Verifier: Claude (gsd-verifier)_
