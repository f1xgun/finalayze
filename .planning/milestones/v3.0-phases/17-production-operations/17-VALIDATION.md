---
phase: 17
slug: production-operations
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 17 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_kill_switch.py tests/unit/test_health_monitor.py tests/unit/test_telegram_bot.py -x -q` |
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
| 17-01-01 | 01 | 1 | OPS-01 | unit | `uv run pytest tests/unit/test_kill_switch.py -x -q` | ❌ W0 | ⬜ pending |
| 17-01-02 | 01 | 1 | OPS-02 | unit | `uv run pytest tests/unit/test_health_monitor.py -x -q` | ❌ W0 | ⬜ pending |
| 17-02-01 | 02 | 2 | OPS-03, OPS-04 | unit | `uv run pytest tests/unit/test_telegram_bot.py -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_kill_switch.py` — stubs for OPS-01
- [ ] `tests/unit/test_health_monitor.py` — stubs for OPS-02
- [ ] `tests/unit/test_telegram_bot.py` — stubs for OPS-03, OPS-04

*Existing test infrastructure covers TelegramAlerter priority queue (OPS-03 partial).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Kill switch 30s SLA on live broker | OPS-01 | Requires live Tinkoff sandbox | Activate kill switch, measure time to order cancellation + alert delivery |
| Telegram /kill confirmation flow | OPS-04 | Requires live Telegram bot | Send /kill to bot, verify confirmation prompt, confirm, verify execution |
| Health monitor alert on broker disconnect | OPS-02 | Requires network failure simulation | Stop broker API, wait 10min, verify 2 missed heartbeat alerts |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
