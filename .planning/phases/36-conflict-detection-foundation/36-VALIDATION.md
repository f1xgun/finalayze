---
phase: 36
slug: conflict-detection-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-12
---

# Phase 36 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `uv run pytest tests/unit/core/test_conflict_detector.py tests/unit/test_llm_client.py -v --tb=short` |
| **Full suite command** | `uv run pytest tests/unit/ -v --tb=short` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/core/test_conflict_detector.py tests/unit/test_llm_client.py -v --tb=short`
- **After every plan wave:** Run `uv run pytest tests/unit/ -v --tb=short`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 36-01-01 | 01 | 1 | CONF-02 | unit | `uv run pytest tests/unit/core/test_conflict_detector.py -v` | ❌ W0 | ⬜ pending |
| 36-01-02 | 01 | 1 | CONF-01, CONF-03, CONF-04 | unit | `uv run pytest tests/unit/core/test_conflict_detector.py -v` | ❌ W0 | ⬜ pending |
| 36-02-01 | 02 | 1 | AGOUT-02 | unit | `uv run pytest tests/unit/test_llm_client.py -v` | ✅ | ⬜ pending |
| 36-02-02 | 02 | 1 | AGOUT-01 | unit | `uv run pytest tests/unit/test_llm_client.py -v` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/core/test_conflict_detector.py` — stubs for ConflictReport schema + ConflictDetector tests
- [ ] Fixtures for synthetic AgentOutput objects (shared across test files)

*Existing test infrastructure covers LLM client tests (`tests/unit/test_llm_client.py`).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Agent .md files have Output Format section | AGOUT-01 | File content in .claude/agents/ | Grep all 6 agent .md files for "## Output Format" |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
