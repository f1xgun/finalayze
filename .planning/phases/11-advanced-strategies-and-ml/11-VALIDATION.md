---
phase: 11
slug: advanced-strategies-and-ml
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | pyproject.toml [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/unit/test_pairs_strategy.py tests/unit/test_ml_features.py -x -q` |
| **Full suite command** | `uv run pytest tests/unit/ -x --timeout=30` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q --timeout=30`
- **After every plan wave:** Run `uv run pytest tests/unit/ -x --timeout=30`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 11-01-01 | 01 | 1 | ADV-01 | unit | `uv run pytest tests/unit/test_pairs_strategy.py -x` | Partial | ⬜ pending |
| 11-02-01 | 02 | 1 | ADV-02 | unit | `uv run pytest tests/unit/test_ml_features.py -x` | Partial | ⬜ pending |
| 11-03-01 | 03 | 2 | ADV-03 | unit | `uv run pytest tests/unit/test_ml_features.py -x` | Partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_pairs_strategy.py` — add MOEX pref share pair tests (allow_short=False)
- [ ] `tests/unit/test_ml_features.py` — add CBR macro feature computation tests

*Existing infrastructure covers framework needs. Only new test cases required.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ML ensemble passes quality gates on ru_blue_chips | ADV-03 | Requires full training pipeline run | Run `scripts/train_models.py --segment ru_blue_chips --walk-forward` and verify gates pass |
| Cointegration holds on post-2022 data | ADV-01 | Requires real market data | Run backtest with pref share pairs, verify trades generated |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
