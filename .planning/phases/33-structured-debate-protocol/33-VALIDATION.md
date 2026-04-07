---
phase: 33
slug: structured-debate-protocol
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 33 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (already configured) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/unit/core/test_debate_schemas.py -x` |
| **Full suite command** | `uv run pytest tests/unit/core/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/core/test_debate_schemas.py -x`
- **After every plan wave:** Run `uv run pytest tests/unit/core/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 33-01-01 | 01 | 1 | DEBATE-01 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_claim_requires_source -x` | ❌ W0 | ⬜ pending |
| 33-01-02 | 01 | 1 | DEBATE-01 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_file_line_source_fields -x` | ❌ W0 | ⬜ pending |
| 33-01-03 | 01 | 1 | DEBATE-01 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_metric_source_fields -x` | ❌ W0 | ⬜ pending |
| 33-01-04 | 01 | 1 | DEBATE-01 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_agent_output_requires_claims -x` | ❌ W0 | ⬜ pending |
| 33-01-05 | 01 | 1 | DEBATE-01 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_claim_confidence_bounds -x` | ❌ W0 | ⬜ pending |
| 33-02-01 | 02 | 1 | DEBATE-02 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_fact_check_report_has_contradictions -x` | ❌ W0 | ⬜ pending |
| 33-02-02 | 02 | 1 | DEBATE-02 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_fact_check_report_markdown -x` | ❌ W0 | ⬜ pending |
| 33-03-01 | 03 | 1 | DEBATE-03 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_debate_file_roundtrip -x` | ❌ W0 | ⬜ pending |
| 33-03-02 | 03 | 1 | DEBATE-03 | unit | `uv run pytest tests/unit/core/test_debate_schemas.py::test_debate_escalation -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/core/test_debate_schemas.py` — stubs for DEBATE-01, DEBATE-02, DEBATE-03

*Existing pytest infrastructure covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Arbiter agent produces fact-check from conflicting outputs | DEBATE-02 | Requires Claude Code agent invocation | Run arbiter-agent with two sample conflicting inputs, verify markdown report |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
