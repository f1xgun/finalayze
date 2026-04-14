---
phase: 47
slug: cross-asset-features-asymmetric-barriers
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-14
---

# Phase 47 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/test_features_moex.py tests/unit/test_auto_ml_research_moex.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_features_moex.py tests/unit/test_auto_ml_research_moex.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 47-01-01 | 01 | 1 | FEAT-01, FEAT-02 | unit | `uv run pytest tests/unit/test_features_moex.py -x -q` | ✅ | ⬜ pending |
| 47-01-02 | 01 | 1 | BARR-01, BARR-02 | unit | `uv run pytest tests/unit/test_auto_ml_research_moex.py -x -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Brent features appear in live experiment run | FEAT-01 | Requires FINALAYZE_TINKOFF_TOKEN and network | Run `python scripts/auto_ml_research.py --segment ru_energy --max-experiments 1` and check feature matrix columns |
| Barrier asymmetry logged at run start | BARR-01 | Requires full experiment run | Check printed barrier params show lower > upper for ru_energy |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
