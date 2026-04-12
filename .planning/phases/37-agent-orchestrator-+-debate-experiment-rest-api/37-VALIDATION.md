---
phase: 37
slug: agent-orchestrator-debate-experiment-rest-api
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-12
---

# Phase 37 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `pyproject.toml` ([tool.pytest.ini_options]) |
| **Quick run command** | `uv run pytest tests/unit/core/test_agent_orchestrator.py tests/unit/api/test_debates_api.py tests/unit/api/test_experiments_api.py -v --tb=short` |
| **Full suite command** | `uv run pytest tests/unit/ -v --tb=short` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command
- **After every plan wave:** Run full suite
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 37-01-01 | 01 | 1 | ORCH-03 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py -k snapshot` | ✅ | ⬜ pending |
| 37-01-02 | 01 | 1 | ORCH-01 | unit | `uv run pytest tests/unit/core/test_agent_orchestrator.py -v` | ❌ W0 | ⬜ pending |
| 37-02-01 | 02 | 2 | ORCH-02 | unit | `uv run pytest tests/unit/api/test_debates_api.py tests/unit/api/test_experiments_api.py -v` | ❌ W0 | ⬜ pending |
| 37-02-02 | 02 | 2 | ORCH-04 | manual | `grep "## Output Format" .claude/agents/agent-orchestrator.md` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/core/test_agent_orchestrator.py` — stubs for orchestrator pipeline tests
- [ ] `tests/unit/api/test_debates_api.py` — stubs for debates REST API tests
- [ ] `tests/unit/api/test_experiments_api.py` — stubs for experiments REST API tests

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| agent-orchestrator.md invocable | ORCH-04 | Claude Code agent invocation | Verify file exists and has correct structure |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
