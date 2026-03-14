---
phase: 5
slug: integration-and-telegram
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-asyncio |
| **Config file** | pyproject.toml (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest tests/unit/ -x -q --timeout=30` |
| **Full suite command** | `uv run pytest --cov -q` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q --timeout=30`
- **After every plan wave:** Run `uv run pytest --cov -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | MON-03 | unit | `uv run pytest tests/unit/test_telegram_queue.py -x` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | MON-01 | unit | `uv run pytest tests/unit/test_telegram_alerter.py -x` | ✅ | ⬜ pending |
| 05-01-03 | 01 | 1 | MON-02 | unit | `uv run pytest tests/unit/test_daily_pnl.py -x` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 1 | AUT-01 | unit | `uv run pytest tests/unit/test_trading_loop_bonds.py -x` | ✅ (minimal) | ⬜ pending |
| 05-02-02 | 02 | 1 | AUT-02 | unit | `uv run pytest tests/unit/test_trading_loop_holidays.py -x` | ✅ (equity only) | ⬜ pending |
| 05-02-03 | 02 | 1 | AUT-03 | integration | `uv run pytest tests/integration/test_circuit_breaker_integration.py -x` | ✅ (equity only) | ⬜ pending |
| 05-02-04 | 02 | 1 | AUT-01 | unit | `uv run pytest tests/unit/test_preflight.py -x` | ❌ W0 | ⬜ pending |
| 05-03-01 | 03 | 2 | MON-04 | unit | `uv run pytest tests/unit/test_bond_cycle.py::test_coupon_alert -x` | ❌ W0 | ⬜ pending |
| 05-03-02 | 03 | 2 | MON-05 | unit | `uv run pytest tests/unit/test_trading_loop_bonds.py::test_cbr_alert -x` | ❌ W0 | ⬜ pending |
| 05-03-03 | 03 | 2 | MON-01 | unit | `uv run pytest tests/unit/test_telegram_webhook.py -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_telegram_queue.py` — stubs for MON-03 (priority queue, rate limiting, batching)
- [ ] `tests/unit/test_daily_pnl.py` — stubs for MON-02 (P&L computation, currency conversion, bond separation)
- [ ] `tests/unit/test_telegram_webhook.py` — stubs for webhook endpoint, command dispatch, auth
- [ ] `tests/unit/test_preflight.py` — stubs for AUT-01 preflight checks, independent degradation
- [ ] Extend `tests/unit/test_trading_loop_bonds.py` — stubs for AUT-01 bond cycle integration, MON-04, MON-05
- [ ] Extend `tests/unit/test_trading_loop_holidays.py` — stubs for AUT-02 bond cycle holiday gating
- [ ] Extend `tests/integration/test_circuit_breaker_integration.py` — stubs for AUT-03 bond layer breakers

*Existing infrastructure covers pytest + pytest-asyncio. No framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Telegram message delivery latency <60s | MON-01 | Requires live Telegram Bot API | Send test alert, measure delivery time in Telegram app |
| Telegram rate limiting under burst | MON-03 | Requires sustained API calls to Telegram | Trigger 20+ alerts rapidly, verify batching in Telegram chat |
| Webhook registration with Telegram | MON-01 | Requires public URL and Telegram API | Run `scripts/register_webhook.py`, verify via getWebhookInfo |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
