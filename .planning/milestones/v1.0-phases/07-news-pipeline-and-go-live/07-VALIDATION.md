---
phase: 7
slug: news-pipeline-and-go-live
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-15
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit/ -x -q --timeout=30` |
| **Full suite command** | `uv run pytest tests/ --timeout=60` |
| **Estimated runtime** | ~45 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/ -x -q --timeout=30`
- **After every plan wave:** Run `uv run pytest tests/ --timeout=60`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 45 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | NWS-01 | unit (TDD) | `uv run pytest tests/unit/test_rss_fetcher.py -v` | TDD creates | pending |
| 07-01-02 | 01 | 1 | NWS-02 | unit (TDD) | `uv run pytest tests/unit/test_entity_extractor.py -v` | TDD creates | pending |
| 07-02-01 | 02 | 1 | NWS-03 | unit (TDD) | `uv run pytest tests/unit/test_telegram_reader.py -v` | TDD creates | pending |
| 07-03-01 | 03 | 2 | NWS-04 | unit (TDD) | `uv run pytest tests/unit/test_news_cycle_integration.py -v` | TDD creates | pending |
| 07-03-02 | 03 | 2 | NWS-05, AUT-05 | unit | `uv run pytest tests/unit/test_event_driven_presets.py tests/unit/test_telegram_stop_command.py tests/unit/test_real_mode_guard.py -v` | TDD creates | pending |
| 07-03-03 | 03 | 2 | NWS-05 | backtest | `ls results/iterations/event-driven-enabled/` | N/A | pending |
| 07-03-04 | 03 | 2 | AUT-05 | checkpoint | Human verifies backtest results + go-live readiness | N/A | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

All plans use TDD (`tdd="true"`) which creates test files inline during RED phase.
No separate Wave 0 test stubs needed -- tests are created before implementation in each task.

*Existing test infrastructure covers analysis, event_driven strategy, and trading loop.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real RSS feed fetching | NWS-01 | Requires live network access to RBC/Interfax/TASS | Run `uv run python -c "import feedparser; print(feedparser.parse('https://rssexport.rbc.ru/rbcnews/news/30/full.rss').feed.title)"` |
| Telegram channel reading | NWS-03 | Requires authenticated Telethon session | Configure `.session` file, run reader against test channel |
| Real MOEX trade execution | AUT-05 | Requires funded T-Invest account | Follow `docs/operations/GO_LIVE_CHECKLIST.md` |
| LLM entity extraction accuracy | NWS-02 | Requires LLM API call with Russian text | Feed sample articles, verify ticker extraction quality |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or TDD-inline tests
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covered by TDD (tests created in RED phase before implementation)
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
