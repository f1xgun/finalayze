# Workflow & Development Process Conventions

This document defines the development workflow for the Finalayze project.
All contributors (human and AI agents) must follow these conventions.

## Superpowers Development Lifecycle

Every feature, bugfix, or refactor follows this mandatory sequence:

### 1. Brainstorm (design before code)
- **Skill:** `brainstorming`
- Explore requirements, constraints, and edge cases through dialogue
- Output: validated design document
- **Gate:** Design must be approved before proceeding

### 2. Isolate (git worktree)
- **Skill:** `using-git-worktrees`
- Create isolated worktree for the feature
- Verify clean test baseline before touching code

### 3. Plan (implementation plan)
- **Skill:** `writing-plans`
- Break work into bite-sized tasks (2-5 minutes each)
- Document exact file paths, code, test steps, verification commands
- Save to `docs/plans/YYYY-MM-DD-<feature-name>.md`

### 4. Execute (TDD + subagents)
- **Skill:** `subagent-driven-development` (same session) or `executing-plans` (batch mode)
- Each task uses `test-driven-development`: RED-GREEN-REFACTOR
  - Write failing test first
  - Verify it fails correctly
  - Write minimal code to pass
  - Verify it passes
  - Refactor if needed
- If bugs arise, use `systematic-debugging` (root cause first, never guess)
- Use `dispatching-parallel-agents` for independent tasks
- `requesting-code-review` after each task (spec compliance + code quality)

### 5. Verify (evidence before claims)
- **Skill:** `verification-before-completion`
- Run full test suite fresh, read output, check exit code
- Never claim "done" without evidence

### 6. Finish (merge or PR)
- **Skill:** `finishing-a-development-branch`
- Options: merge locally, create PR, keep branch, or discard

### 7. PR Review-Fix Cycle (mandatory for PRs)

After CI passes on a PR, run an automated review-fix loop until no issues remain:

1. **Dispatch review subagent** — reads all changed files, creates GitHub issues for every
   problem found (bugs, convention violations, missing tests, type safety, etc.)
2. **Dispatch fix subagent** — reads all open issues on the PR, fixes them on the branch,
   pushes, and verifies CI still passes
3. **Repeat** steps 1-2 until the review subagent finds zero new issues to create
4. **Merge** once CI is green and the review cycle is clean

Each review issue must be:
- Specific (file:line reference)
- Actionable (exact description of the fix)
- Labeled correctly (`bug`, `enhancement`, `test`, etc.)

```bash
# Create a review issue
gh issue create --repo owner/repo --title "..." --body "file:line — ..." --label "bug"

# Close fixed issues from a commit message
gh issue close <number> --comment "Fixed in <commit-sha>"
```

## Branch Strategy

- `main` -- production-ready code, protected
- `dev` -- integration branch, all feature branches merge here first
- `feature/<name>` -- new functionality
- `fix/<name>` -- bug fixes
- `refactor/<name>` -- structural improvements with no behavior change

## Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`.

**Scopes:** `core`, `config`, `data`, `analysis`, `strategies`, `markets`, `ml`,
`risk`, `execution`, `backtest`, `dashboard`, `api`, `infra`.

Examples:
```
feat(markets): add Alpaca REST fetcher with rate limiting
fix(risk): correct max-drawdown calculation for partial fills
test(strategies): add momentum strategy unit tests
docs(architecture): update data flow diagram
```

## Pull Request Process

1. Create a feature branch from `dev`.
2. Implement changes with tests. Minimum coverage: 80% for new code.
3. Run the full quality check locally before pushing:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src/
   uv run pytest --cov
   ```
4. Open a PR against `dev`. The PR description must include:
   - **What** changed and **why**.
   - Link to the relevant phase/task in `docs/plans/`.
   - Test plan or evidence of testing.
5. Wait for all CI jobs to pass (lint, typecheck, test).
6. Run the **PR Review-Fix Cycle** (see Step 7 above) — dispatch review subagent,
   fix all issues, repeat until clean.
7. Squash-merge to keep history clean once CI is green and no open review issues remain.

> **First PR for this project:** https://github.com/f1xgun/finalayze/pull/1 (Phase 1 backtest slice)

## Code Review Checklist

- [ ] Layer violations: no upward imports across dependency layers
- [ ] Type safety: no `Any` without explicit justification
- [ ] Error handling: domain exceptions from `core/exceptions.py`, not bare `Exception`
- [ ] Async correctness: no blocking calls in async functions
- [ ] Tests: unit tests for logic, integration tests for DB/API
- [ ] Docstrings: Google style on all public functions and classes
- [ ] Configuration: no hardcoded values; use `config/settings.py`
- [ ] Secrets: no credentials in code; use environment variables
- [ ] TDD evidence: tests written before implementation code

## Quality Gates

Every PR must pass these automated checks:

| Check | Tool | Threshold |
|---|---|---|
| Linting | ruff | zero warnings |
| Formatting | ruff format | zero diffs |
| Type checking | mypy (strict) | zero errors |
| Unit tests | pytest | all pass |
| Coverage | pytest-cov | >= 80% new code |

## Documentation Updates

When making changes, update the relevant docs:

- New module or feature --> update `docs/architecture/OVERVIEW.md`
- API change --> update `docs/api/ENDPOINTS.md`
- New dependency --> update `docs/architecture/DEPENDENCY_LAYERS.md`
- Architecture decision --> add ADR in `docs/architecture/DECISIONS.md`
- Completed task --> update `docs/plans/PHASE_*.md` and `docs/plans/ROADMAP.md`
- Quality improvement --> update `docs/quality/GRADES.md`
- Resolved tech debt --> update `docs/quality/GAPS.md`

## Work Modes

The system operates in four modes. Always develop and test in the appropriate mode:

| Mode | Purpose | Broker | Data |
|---|---|---|---|
| `debug` | Local development, verbose logging | Mock | Fixtures |
| `sandbox` | Paper trading with real market data | Alpaca Paper / Tinkoff Sandbox | Live |
| `test` | Automated integration testing | Simulated | Historical |
| `real` | Live trading with real money | Alpaca Live / Tinkoff Live | Live |

**Rule:** Never deploy code to `real` mode without passing all quality gates
in `sandbox` and `test` modes first.

## Task Tracking

- Phases and tasks are tracked in `docs/plans/`.
- Update task status as work progresses: `NOT STARTED` -> `IN PROGRESS` -> `DONE`.
- Log blockers and decisions in the relevant phase document.

## Changelog

All user-facing and system-affecting changes must be recorded in `CHANGELOG.md`
following [Keep a Changelog](https://keepachangelog.com/) format.

## Environment Setup

```bash
# Clone and install
git clone <repo-url>
cd finalayze
uv sync

# Copy environment template
cp .env.example .env
# Edit .env with your credentials

# Start infrastructure
docker compose -f docker/docker-compose.dev.yml up -d

# Run migrations
uv run alembic upgrade head

# Verify setup
uv run pytest
uv run ruff check .
uv run mypy src/
```

## §8 Agent Dispatch Rules

The project uses 16 Claude Code sub-agents defined in `.claude/agents/`. Two tiers:

### Tier 1: Domain Expert Agents

Invoke these for high-level analysis, audits, and design review.
They cross-cut the entire codebase and produce structured reports + GitHub issues.

| Invoke when... | Agent |
|---|---|
| Reviewing strategy math, signal quality, or backtest methodology | `quant-analyst` |
| Auditing risk thresholds, circuit breakers, or pre-trade checks | `risk-officer` |
| Reviewing ML pipeline, feature engineering, or model calibration | `ml-engineer` |
| Checking layer violations, async correctness, or data flow | `systems-architect` |

**Brainstorm gate:** Before finalising any design that touches strategies, risk, ML, or architecture, invoke the relevant domain expert(s):

```
Task("quant-analyst: review the proposed momentum strategy changes")
Task("risk-officer: audit new position sizing formula")
```

**Quarterly audit:** Dispatch all 4 experts in parallel to audit the full system:

```
Task("quant-analyst: full strategy and backtest audit — create GitHub issues for every gap")
Task("risk-officer: full risk management audit — create GitHub issues for every gap")
Task("ml-engineer: full ML pipeline audit — create GitHub issues for every gap")
Task("systems-architect: full architecture audit — create GitHub issues for every gap")
```

### Tier 2: Module Agents

Use as the **implementer** in `subagent-driven-development`. The controller identifies which module a task touches and dispatches the appropriate agent.

| Module path | Agent to dispatch |
|---|---|
| `src/finalayze/core/` | `core-agent` |
| `config/` | `config-agent` |
| `src/finalayze/data/` | `data-agent` |
| `src/finalayze/markets/` | `markets-agent` |
| `src/finalayze/analysis/` | `analysis-agent` |
| `src/finalayze/ml/` | `ml-agent` |
| `src/finalayze/strategies/` | `strategies-agent` |
| `src/finalayze/risk/` | `risk-agent` |
| `src/finalayze/execution/` | `execution-agent` |
| `src/finalayze/backtest/` | `backtest-agent` |
| `src/finalayze/api/`, `src/finalayze/dashboard/` | `api-agent` |
| `docker/`, `alembic/`, `pyproject.toml`, CI | `infra-agent` |

**Task touches multiple modules?** Dispatch one agent per module sequentially (not parallel — they may edit overlapping files).
