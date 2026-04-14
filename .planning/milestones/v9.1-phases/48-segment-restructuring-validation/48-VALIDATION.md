---
phase: 48
slug: segment-restructuring-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-14
---

# Phase 48 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_auto_ml_research_moex.py tests/unit/test_segments.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_auto_ml_research_moex.py tests/unit/test_segments.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 48-01-01 | 01 | 1 | SEGM-01 | unit | `uv run pytest tests/unit/test_segments.py -x -q` | ✅ | ⬜ pending |
| 48-01-02 | 01 | 1 | SEGM-02 | unit | `uv run pytest tests/unit/test_auto_ml_research_moex.py -k "history" -x -q` | ✅ | ⬜ pending |
| 48-01-03 | 01 | 1 | SEGM-03 | manual | N/A (requires TINKOFF_TOKEN) | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ru_energy/ru_finance/ru_tech produce ACCEPT/INCONCLUSIVE verdict | SEGM-03 | Requires FINALAYZE_TINKOFF_TOKEN and live T-Bank API | Run `python scripts/auto_ml_research.py --segment ru_energy --max-experiments 1` for each segment, check verdict |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
