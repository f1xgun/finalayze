---
phase: 12
slug: portfolio-assembly
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 12 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/unit/test_portfolio_orchestrator.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x --timeout=30` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run full suite
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 12-01-01 | 01 | 1 | PORT-01 | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py -x` | No | ⬜ pending |
| 12-01-02 | 01 | 1 | PORT-02 | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py -x` | No | ⬜ pending |
| 12-02-01 | 02 | 2 | PORT-03 | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py -x` | No | ⬜ pending |

---

## Wave 0 Requirements

- [ ] `tests/unit/test_portfolio_orchestrator.py` — new test file for portfolio orchestrator

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Blended WF Sharpe >= +0.10 | PORT-03 | Requires real market data + full iteration | Run portfolio backtest iteration, check blended Sharpe |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity
- [ ] Wave 0 covers all MISSING references
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
