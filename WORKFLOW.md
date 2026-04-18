# Workflow & Development Process Conventions

Process conventions for humans and AI agents working in this repository.
This file is **not** the agent entry point — that is [`AGENTS.md`](AGENTS.md).

> **What used to live here and where it went:**
> - "Superpowers Development Lifecycle" — **removed**. Replaced by GSD (see `AGENTS.md`).
> - §8 Agent Dispatch Rules — **moved** to [`docs/AGENTS.md`](docs/AGENTS.md).
> - §9 Trading-Specific Skills — **moved** to [`docs/AGENTS.md`](docs/AGENTS.md).

## Development lifecycle

See [`AGENTS.md`](AGENTS.md) for the `/gsd:*` commands (planned work, quick tasks, debugging,
session handoff). That file is the single source of truth for workflow commands.

## Branch strategy

- `main` — production-ready, protected
- `dev` — integration, feature branches merge here first
- `feature/<name>` — new functionality
- `fix/<name>` — bug fixes
- `refactor/<name>` — structural improvements, no behaviour change

## Commit conventions

[Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]
[optional footer]
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`.
**Scopes:** `core`, `config`, `data`, `analysis`, `strategies`, `markets`, `ml`, `risk`,
`execution`, `backtest`, `dashboard`, `api`, `infra`.

Examples:
```
feat(markets): add Alpaca REST fetcher with rate limiting
fix(risk): correct max-drawdown calculation for partial fills
test(strategies): add momentum strategy unit tests
docs(architecture): update data flow diagram
```

## Pull request process

1. Create a feature branch from `dev`.
2. Implement with tests. Minimum coverage: 80% for new code.
3. Run the full quality check locally before pushing:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src/
   uv run pytest --cov
   ```
4. Open a PR against `dev`. Description must include **what**, **why**, link to the relevant
   phase plan in `docs/plans/`, and test plan / evidence.
5. Wait for all CI jobs to pass (lint, typecheck, test).
6. Run the PR review-fix cycle (below).
7. Squash-merge once CI is green and no open review issues remain.

## PR review-fix cycle (mandatory)

After CI passes, loop until clean:

1. **Dispatch review sub-agent** — reads changed files, creates GitHub issues for every problem
   (bugs, convention violations, missing tests, type gaps).
2. **Dispatch fix sub-agent** — reads open issues on the PR, fixes them, pushes, verifies CI.
3. **Repeat** until the review sub-agent finds zero new issues.

Each review issue must be:
- Specific (`file:line` reference)
- Actionable (exact description of the fix)
- Labelled (`bug`, `enhancement`, `test`, ...)

```bash
gh issue create --repo owner/repo --title "..." --body "file:line — ..." --label "bug"
gh issue close <number> --comment "Fixed in <commit-sha>"
```

## Code review checklist

- [ ] Layer violations: no upward imports across dependency layers
- [ ] Type safety: no `Any` without explicit justification
- [ ] Error handling: domain exceptions from `core/exceptions.py`, not bare `Exception`
- [ ] Async correctness: no blocking calls in async functions
- [ ] Tests: unit for logic, integration for DB/API
- [ ] Docstrings: Google style on all public functions and classes
- [ ] Configuration: no hardcoded values; use `config/settings.py`
- [ ] Secrets: no credentials in code; environment variables only
- [ ] TDD evidence: tests written before implementation

## Quality gates

| Check | Tool | Threshold |
|---|---|---|
| Linting | ruff | zero warnings |
| Formatting | ruff format | zero diffs |
| Type checking | mypy (strict) | zero errors |
| Unit tests | pytest | all pass |
| Coverage | pytest-cov | >= 80% new code |

## Documentation updates

When making changes, update the relevant docs:

- New module or feature → update `docs/architecture/OVERVIEW.md`
- API change → update `docs/api/ENDPOINTS.md`
- New dependency → update `docs/architecture/DEPENDENCY_LAYERS.md`
- Architecture decision → add ADR in `docs/architecture/DECISIONS.md`
- Completed task → update `docs/plans/*.md` and `docs/plans/ROADMAP.md`
- Quality improvement → update `docs/quality/GRADES.md`
- Resolved tech debt → update `docs/quality/GAPS.md`
- **New module / renamed AGENTS.md node** → update `.agents/manifest.jsonl`

## Work modes

| Mode | Purpose | Broker | Data |
|---|---|---|---|
| `debug` | Local development, verbose logging | Mock | Fixtures |
| `sandbox` | Paper trading with real market data | Alpaca Paper / Tinkoff Sandbox | Live |
| `test` | Automated integration testing | Simulated | Historical |
| `real` | Live trading with real money | Alpaca Live / Tinkoff Live | Live |

**Rule:** never deploy code to `real` without passing all quality gates in `sandbox` and `test` first.

## Task tracking

- Phases and tasks live in `.planning/` (GSD state) and `docs/plans/` (historical plans).
- Status transitions: `NOT STARTED` → `IN PROGRESS` → `DONE`.
- Log blockers and decisions in the relevant phase document.

## Changelog

All user-facing and system-affecting changes go in `CHANGELOG.md` following
[Keep a Changelog](https://keepachangelog.com/).

## Environment setup

```bash
git clone <repo-url>
cd finalayze
uv sync

cp .env.example .env
# Edit .env with your credentials (notably FINALAYZE_TINKOFF_TOKEN for MOEX)

docker compose -f docker/docker-compose.dev.yml up -d
uv run alembic upgrade head

uv run pytest
uv run ruff check .
uv run mypy src/
```
