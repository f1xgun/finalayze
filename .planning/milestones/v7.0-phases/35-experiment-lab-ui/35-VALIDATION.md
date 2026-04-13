---
phase: 35
slug: experiment-lab-ui
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-08
---

# Phase 35 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/unit/test_dashboard_pages.py -x` |
| **Full suite command** | `uv run pytest tests/unit/ -x` |
| **Estimated runtime** | ~3 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_dashboard_pages.py -x`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 3 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 35-01-01 | 01 | 1 | UI-EXP-01 | smoke | `uv run pytest tests/unit/test_dashboard_pages.py::test_experiments_list_render_importable -x` | ❌ W0 | ⬜ pending |
| 35-01-02 | 01 | 1 | UI-EXP-02 | smoke | `uv run pytest tests/unit/test_dashboard_pages.py::test_experiment_detail_render_importable -x` | ❌ W0 | ⬜ pending |
| 35-01-03 | 01 | 1 | UI-EXP-03 | smoke | `uv run pytest tests/unit/test_dashboard_pages.py::test_decision_history_render_importable -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_dashboard_pages.py` — add 3 new smoke test functions (file exists, append new tests)

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| List page shows experiments with status badges and filtering | UI-EXP-01 | Requires Streamlit runtime + visual inspection | Run `streamlit run src/finalayze/dashboard/app.py`, navigate to Experiments |
| Detail page shows debate context, charts, A/B/AB comparison | UI-EXP-02 | Requires Streamlit runtime + Plotly rendering | Click an experiment from list page |
| Decision history shows accepted/rejected with reasoning | UI-EXP-03 | Requires Streamlit runtime + visual inspection | Navigate to Decision History page |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 3s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
