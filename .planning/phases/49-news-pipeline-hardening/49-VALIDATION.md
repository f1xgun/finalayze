---
phase: 49
slug: news-pipeline-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-14
---

# Phase 49 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest + pytest-asyncio |
| **Config file** | pyproject.toml `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest tests/unit/test_news_analyzer.py tests/unit/test_news_pipeline.py -x` |
| **Full suite command** | `uv run pytest tests/ -x --timeout=30` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_news_analyzer.py tests/unit/test_news_pipeline.py -x`
- **After every plan wave:** Run `uv run pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 49-01-01 | 01 | 1 | NEWS-02 | unit | `uv run pytest tests/unit/test_news_analyzer.py::test_parse_structured -x` | Wave 0 | pending |
| 49-01-02 | 01 | 1 | NEWS-01 | unit | `uv run pytest tests/unit/test_news_analyzer.py::test_llm_timeout -x` | Wave 0 | pending |
| 49-02-01 | 02 | 1 | NEWS-03 | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_credibility_map -x` | Wave 0 | pending |
| 49-02-02 | 02 | 1 | NEWS-04 | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_ticker_validation -x` | Wave 0 | pending |
| 49-02-03 | 02 | 1 | NEWS-05 | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_llm_liveness -x` | Wave 0 | pending |
| 49-02-04 | 02 | 1 | NEWS-06 | unit | `uv run pytest tests/unit/test_news_pipeline.py::test_budget_cap -x` | Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_news_pipeline.py` — new file covering NEWS-03, NEWS-04, NEWS-05, NEWS-06
- [ ] Update `tests/unit/test_news_analyzer.py` — covers NEWS-01, NEWS-02

*Existing test_news_analyzer.py exists; test_news_pipeline.py is new.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Telegram alert fires on 3 LLM failures | NEWS-05 | Requires Telegram bot connection | Run sandbox cycle with LLM API disabled, verify message in Telegram chat |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
