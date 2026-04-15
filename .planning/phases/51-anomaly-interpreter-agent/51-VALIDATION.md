---
phase: 51
slug: anomaly-interpreter-agent
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-15
---

# Phase 51 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + pytest-asyncio |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/unit/test_anomaly_detector.py -x` |
| **Full suite command** | `uv run pytest tests/ -x --timeout=30` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_anomaly_detector.py -x`
- **After every plan wave:** Run `uv run pytest tests/unit/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 51-01-01 | 01 | 1 | ANMI-01 | unit | `uv run pytest tests/unit/test_anomaly_detector.py::TestOrderingGuarantee -x` | Wave 0 | pending |
| 51-01-02 | 01 | 1 | ANMI-02 | unit | `uv run pytest tests/unit/test_anomaly_detector.py::TestLLMEnrichment -x` | Wave 0 | pending |
| 51-01-03 | 01 | 1 | ANMI-03 | unit | `uv run pytest tests/unit/test_anomaly_detector.py::TestGracefulDegradation -x` | Wave 0 | pending |

---

## Wave 0 Requirements

- [ ] `tests/unit/test_anomaly_detector.py` — new file covering ANMI-01, ANMI-02, ANMI-03
- [ ] `src/finalayze/analysis/anomaly_detector.py` — AnomalyDetector class + AnomalyResult schema

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Follow-up Telegram message arrives within 30s | ANMI-02 | Requires live Telegram bot + LLM API | Run sandbox, trigger anomaly, check Telegram chat for two messages |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
