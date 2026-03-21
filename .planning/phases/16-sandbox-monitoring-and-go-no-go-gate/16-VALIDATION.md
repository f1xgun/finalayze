---
phase: 16
slug: sandbox-monitoring-and-go-no-go-gate
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_sandbox_monitor.py tests/unit/test_go_no_go.py tests/unit/test_anomaly_detector.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -x -q` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run `uv run pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | MON-01 | unit | `uv run pytest tests/unit/test_sandbox_monitor.py -x -q` | ❌ W0 | ⬜ pending |
| 16-01-02 | 01 | 1 | MON-02 | unit | `uv run pytest tests/unit/test_sandbox_monitor.py -x -q` | ❌ W0 | ⬜ pending |
| 16-02-01 | 02 | 1 | GATE-01, GATE-02 | unit | `uv run pytest tests/unit/test_go_no_go.py -x -q` | ❌ W0 | ⬜ pending |
| 16-03-01 | 03 | 2 | MON-04 | unit | `uv run pytest tests/unit/test_anomaly_detector.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_sandbox_monitor.py` — stubs for MON-01, MON-02
- [ ] `tests/unit/test_go_no_go.py` — stubs for GATE-01, GATE-02
- [ ] `tests/unit/test_anomaly_detector.py` — stubs for MON-04

*Existing test infrastructure covers TelegramAlerter and MetricsCollector.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Telegram anomaly alert delivery | MON-04 | Requires live Telegram bot | Run sandbox, trigger drawdown spike, verify alert received in Telegram |
| TimescaleDB hypertable creation | MON-01 | Requires running database | Run `alembic upgrade head`, verify `sandbox_metrics` is a hypertable |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
