---
phase: 18
slug: dashboard-and-api-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 18 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_api_sandbox.py tests/unit/test_dashboard_pages.py -x -q` |
| **Full suite command** | `uv run pytest tests/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run `uv run pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 18-01-01 | 01 | 1 | GATE-03 | unit | `uv run pytest tests/unit/test_api_sandbox.py -x -q` | ❌ W0 | ⬜ pending |
| 18-02-01 | 02 | 1 | MON-03 | unit | `uv run pytest tests/unit/test_dashboard_pages.py -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_api_sandbox.py` — stubs for GATE-03

*`tests/unit/test_dashboard_pages.py` already exists — add sandbox test cases.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Streamlit sandbox page renders correctly | MON-03 | Visual UI verification | Run `streamlit run src/finalayze/dashboard/app.py`, navigate to Sandbox page, verify 5 sections render |
| Equity curve chart displays correctly | MON-03 | Visual chart rendering | Check Plotly chart renders with drawdown overlay |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
