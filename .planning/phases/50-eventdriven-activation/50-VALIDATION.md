---
phase: 50
slug: eventdriven-activation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-15
---

# Phase 50 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + pytest-asyncio |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/unit/test_strategy_combiner.py tests/unit/test_event_driven_strategy.py -x` |
| **Full suite command** | `uv run pytest tests/unit/ -q` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_strategy_combiner.py tests/unit/test_event_driven_strategy.py tests/unit/test_news_pipeline.py -x`
- **After every plan wave:** Run `uv run pytest tests/unit/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 50-01-01 | 01 | 1 | EVNT-01 | unit | `uv run pytest tests/unit/test_event_driven_strategy.py -x` | Wave 0 | pending |
| 50-01-02 | 01 | 1 | EVNT-02 | unit | `uv run pytest tests/unit/test_strategy_combiner.py -k "dedup" -x` | Wave 0 | pending |
| 50-01-03 | 01 | 1 | EVNT-03 | unit | `uv run pytest tests/unit/test_news_pipeline.py -k "ttl" -x` | Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] Update `tests/unit/test_event_driven_strategy.py` — add credibility and preset tests
- [ ] Update `tests/unit/test_strategy_combiner.py` — add dedup tests
- [ ] Update `tests/unit/test_news_pipeline.py` — add TTL freeze tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Signal appears in signals DB table after sandbox news cycle | EVNT-01 | Requires running sandbox with news feed | Run sandbox mode, inject test article, verify signals table has EventDriven entry |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 20s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
