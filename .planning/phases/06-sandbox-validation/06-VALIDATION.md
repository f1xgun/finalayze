---
phase: 6
slug: sandbox-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured) |
| **Config file** | pyproject.toml [tool.pytest] |
| **Quick run command** | `uv run pytest tests/unit/ -x -q` |
| **Full suite command** | `uv run pytest --cov` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q`
- **After every plan wave:** Run `uv run pytest --cov`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 1 | AUT-04 | unit | `uv run pytest tests/unit/test_tinkoff_reconnect.py -x` | Wave 0 | pending |
| 06-01-02 | 01 | 1 | AUT-04 | unit | `uv run pytest tests/unit/test_api_health.py -x` | Exists (needs update) | pending |
| 06-01-03 | 01 | 1 | AUT-04 | unit | `uv run pytest tests/unit/test_trading_loop_jobstore.py -x` | Wave 0 | pending |
| 06-01-04 | 01 | 1 | AUT-04 | unit | `uv run pytest tests/unit/test_candle_staleness.py -x` | Wave 0 | pending |
| 06-01-05 | 01 | 1 | AUT-06 | unit | `uv run pytest tests/unit/test_order_reconciliation.py -x` | Wave 0 | pending |
| 06-01-06 | 01 | 1 | AUT-06 | unit | `uv run pytest tests/unit/test_tinkoff_broker.py -x` | Exists (needs new test) | pending |
| 06-02-01 | 02 | 2 | AUT-04 | unit | `uv run pytest tests/unit/test_validation_logger.py -x` | Wave 0 | pending |
| 06-02-02 | 02 | 2 | AUT-04 | unit | `uv run pytest tests/unit/test_validation_report.py -x` | Wave 0 | pending |
| 06-02-03 | 02 | 2 | AUT-04 | manual | `docker compose -f docker/docker-compose.sandbox.yml up -d` | N/A | pending |
| 06-02-04 | 02 | 2 | AUT-04 | manual | Run validation, check report | N/A | pending |

*Status: pending · green · red · flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_tinkoff_reconnect.py` — stubs for AUT-04 (gRPC reconnection logic)
- [ ] `tests/unit/test_trading_loop_jobstore.py` — stubs for AUT-04 (APScheduler job store)
- [ ] `tests/unit/test_candle_staleness.py` — stubs for AUT-04 (candle staleness check)
- [ ] `tests/unit/test_validation_logger.py` — stubs for AUT-04 (structured cycle logger)
- [ ] `tests/unit/test_validation_report.py` — stubs for AUT-04 (validation report generation)
- [ ] `tests/unit/test_order_reconciliation.py` — stubs for AUT-06 (in-flight order reconciliation)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Docker Compose stack starts cleanly | AUT-04 | Requires Docker daemon and real containers | `docker compose -f docker/docker-compose.sandbox.yml up -d`, verify all services healthy |
| 5-day sandbox run passes criteria | AUT-04 | Requires real T-Invest sandbox API over 5 trading days | Run system, check final validation report for pass/fail |
| Deliberate kill test recovery | AUT-06 | Requires killing running container mid-cycle | Kill container during market hours, verify restart + reconcile |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
