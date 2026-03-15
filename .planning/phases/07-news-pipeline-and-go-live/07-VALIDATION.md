---
phase: 7
slug: news-pipeline-and-go-live
status: draft
nyquist_compliant: false
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
| 07-01-01 | 01 | 1 | NWS-01 | unit | `uv run pytest tests/unit/test_rss_fetcher.py -v` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | NWS-01 | unit | `uv run pytest tests/unit/test_rss_fetcher.py -v` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 1 | NWS-02 | unit | `uv run pytest tests/unit/test_news_analyzer.py -v` | ✅ | ⬜ pending |
| 07-02-02 | 02 | 1 | NWS-03 | unit | `uv run pytest tests/unit/test_telegram_reader.py -v` | ❌ W0 | ⬜ pending |
| 07-02-03 | 02 | 1 | NWS-04 | unit | `uv run pytest tests/unit/test_event_driven.py -v` | ✅ | ⬜ pending |
| 07-03-01 | 03 | 2 | NWS-05 | integration | `uv run pytest tests/unit/test_event_driven.py -v` | ✅ | ⬜ pending |
| 07-03-02 | 03 | 2 | AUT-05 | integration | `uv run pytest tests/unit/test_trading_loop.py -v` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_rss_fetcher.py` — stubs for NWS-01 (RSS fetcher tests)
- [ ] `tests/unit/test_telegram_reader.py` — stubs for NWS-03 (Telegram reader tests)

*Existing test infrastructure covers analysis, event_driven strategy, and trading loop.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real RSS feed fetching | NWS-01 | Requires live network access to RBC/Interfax/TASS | Run `scripts/test_rss_feeds.py` and verify articles parsed |
| Telegram channel reading | NWS-03 | Requires authenticated Telethon session | Configure `.session` file, run reader against test channel |
| Real MOEX trade execution | AUT-05 | Requires funded T-Invest account | Execute first trade via sandbox-validated system on real account |
| LLM entity extraction accuracy | NWS-02 | Requires LLM API call with Russian text | Feed sample articles, verify ticker extraction quality |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 45s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
