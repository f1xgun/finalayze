---
phase: 52-portfolio-review-agent
plan: 01
subsystem: analysis
tags: [portfolio-review, advisory-agent, pydantic-schema, telegram, safety-by-design]
dependency_graph:
  requires: []
  provides: [PortfolioReviewResult, PositionSummary, ConcentrationWarning, CatalystEvent, build_review_prompt, format_review_telegram]
  affects: []
tech_stack:
  added: []
  patterns: [frozen-pydantic-schema, module-level-assertion, code-grep-safety-test]
key_files:
  created:
    - src/finalayze/analysis/portfolio_review_agent.py
    - tests/unit/test_portfolio_review_agent.py
  modified: []
key_decisions:
  - "Used ticker/market field names instead of symbol/market_id for structural disjointness from Signal"
  - "Module-level _FORBIDDEN_FIELDS assertion blocks direction/confidence/side at import time"
  - "Replaced subprocess code-grep with pure-Python file read for ruff S603/S607 compliance"
  - "Plain text Telegram format (not MarkdownV2) for robustness"
metrics:
  duration_seconds: 356
  completed: "2026-04-15T08:58:33Z"
  tasks_completed: 2
  tasks_total: 2
  tests_added: 29
  files_created: 2
  files_modified: 0
---

# Phase 52 Plan 01: Portfolio Review Agent Schema & Helpers Summary

Advisory-only PortfolioReviewResult Pydantic v2 schema with frozen sub-schemas, module-level forbidden-field assertion, prompt builder for LLM portfolio analysis, and structured Telegram formatter -- all enforcing safety-by-design contract (no trade-directive fields).

## What Was Built

### Schemas (Layer 3: analysis/portfolio_review_agent.py)
- **PositionSummary**: ticker, market, quantity, unrealized_pnl, pct_of_portfolio (frozen)
- **ConcentrationWarning**: ticker, market, concentration_pct, warning_level (frozen)
- **CatalystEvent**: ticker, event_type, expected_date (frozen)
- **PortfolioReviewResult**: reviewed_at, positions, concentration_warnings, catalyst_events, overall_assessment, risk_score (frozen, default_factory=list for optional lists)

### Safety Contract (PFRA-03)
- `_FORBIDDEN_FIELDS = {"direction", "confidence", "side"}` -- module-level assertion at import time
- Field naming: `ticker`/`market` (NOT `symbol`/`market_id`) -- structurally distinct from Signal
- Code-grep tests verify zero BrokerRouter/place_order/generate_signal references

### Helpers
- **build_review_prompt()**: Constructs LLM prompt from portfolio data dict (multi-market support, empty portfolio graceful handling)
- **format_review_telegram()**: Produces structured multi-section Telegram message (positions, concentration risk, catalysts, assessment, risk score)
- **PORTFOLIO_REVIEW_SYSTEM_PROMPT**: System prompt with JSON schema guidance for LLM
- **REVIEW_LLM_TIMEOUT = 60.0**: Timeout constant for portfolio review LLM calls

## Task Commits

| Task | Name | Commit | Key Changes |
|------|------|--------|-------------|
| 1 | Schema + safety tests (TDD) | 5b4fea9 | 4 frozen schemas, _FORBIDDEN_FIELDS assertion, 19 tests |
| 2 | Prompt builder + Telegram formatter (TDD) | a3e4d72 | build_review_prompt, format_review_telegram, 10 tests |

## Verification Results

- 29 tests pass: `uv run pytest tests/unit/test_portfolio_review_agent.py -x -v --no-cov`
- Ruff lint clean: `uv run ruff check`
- Ruff format clean: `uv run ruff format --check`
- Mypy strict clean: `uv run mypy --strict`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Docstring contained code-grep target strings**
- **Found during:** Task 1
- **Issue:** Module docstring mentioned "BrokerRouter/place_order/generate_signal" literally, causing the code-grep safety test to fail
- **Fix:** Rewording docstring to say "order-pipeline references" instead of listing the specific function names
- **Files modified:** src/finalayze/analysis/portfolio_review_agent.py
- **Commit:** 5b4fea9

**2. [Rule 1 - Bug] Replaced subprocess code-grep with pure-Python file read**
- **Found during:** Task 1
- **Issue:** Using subprocess.run(["grep", ...]) triggered ruff S603/S607 (untrusted subprocess input) warnings
- **Fix:** Switched to Path.read_text() + `in` operator for the code-grep tests -- simpler and lint-clean
- **Files modified:** tests/unit/test_portfolio_review_agent.py
- **Commit:** 5b4fea9

**3. [Rule 1 - Bug] TC003 conflict with Pydantic runtime type resolution**
- **Found during:** Task 1
- **Issue:** Moving datetime/Decimal to TYPE_CHECKING block (per TC003) caused PydanticUserError at runtime because `from __future__ import annotations` defers type resolution
- **Fix:** Kept imports as runtime imports with `# noqa: TC003` suppression
- **Files modified:** src/finalayze/analysis/portfolio_review_agent.py
- **Commit:** 5b4fea9

## Self-Check: PASSED

- [x] src/finalayze/analysis/portfolio_review_agent.py EXISTS (226 lines)
- [x] tests/unit/test_portfolio_review_agent.py EXISTS (408 lines, > 80 minimum)
- [x] Commit 5b4fea9 EXISTS in git log
- [x] Commit a3e4d72 EXISTS in git log
- [x] 29 tests pass
- [x] ruff check clean
- [x] ruff format clean
- [x] mypy strict clean
