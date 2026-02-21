# Workflow & Development Process Conventions

This document defines the development workflow for the Finalayze project.
All contributors (human and AI agents) must follow these conventions.

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
5. At least one approval required before merge.
6. Squash-merge to keep history clean.

## Code Review Checklist

- [ ] Layer violations: no upward imports across dependency layers
- [ ] Type safety: no `Any` without explicit justification
- [ ] Error handling: domain exceptions from `core/exceptions.py`, not bare `Exception`
- [ ] Async correctness: no blocking calls in async functions
- [ ] Tests: unit tests for logic, integration tests for DB/API
- [ ] Docstrings: Google style on all public functions and classes
- [ ] Configuration: no hardcoded values; use `config/settings.py`
- [ ] Secrets: no credentials in code; use environment variables

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
docker compose -f docker/docker-compose.yml up -d

# Run migrations
uv run alembic upgrade head

# Verify setup
uv run pytest
uv run ruff check .
uv run mypy src/
```
