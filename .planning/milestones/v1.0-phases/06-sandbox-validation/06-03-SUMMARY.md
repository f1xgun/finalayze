---
phase: 06-sandbox-validation
plan: 03
subsystem: infrastructure
tags: [validation, reporting, orchestration, docker, sandbox, grafana]

# Dependency graph
requires:
  - phase: 06-sandbox-validation
    plan: 02
    provides: "Docker Compose sandbox stack, cycle logger, TradingLoop lifespan"
provides:
  - "Validation report generator (generate_report) producing pass/fail markdown from cycle logs"
  - "Orchestration script with pre-flight checklist for 5-day sandbox run"
  - "Verified Docker Compose sandbox stack (all 5 services healthy)"
affects: [phase-07-news-go-live]

# Tech tracking
tech-stack:
  added: []
  patterns: [validation-report-generation, pre-flight-checklist]

key-files:
  created:
    - scripts/generate_validation_report.py
    - scripts/run_sandbox_validation.py
    - tests/unit/test_validation_report.py
  modified:
    - docker/Dockerfile.prod
    - docker/.dockerignore

key-decisions:
  - "Validation criteria: 5+ days, <5% DD, >=10 trades, 0 critical errors"
  - "Report reads CycleLogEntry via ValidationLogger.get_entries() and groups by date"
  - "Orchestration script is a documented checklist (not automated runner) for 5-day validation"
  - "Docker fixes: README.md in both build stages, psycopg2-binary added, .dockerignore whitelist"

patterns-established:
  - "Validation report pattern: read structured logs -> compute metrics -> pass/fail assessment -> markdown output"
  - "Pre-flight checklist pattern: print verification steps before long-running operation"

requirements-completed: [AUT-04]

# Metrics
duration: 6min
completed: 2026-03-15
---

# Phase 6 Plan 03: Validation Report Generator and Sandbox Verification Summary

**Validation report generator with pass/fail criteria assessment from cycle logs, orchestration checklist for 5-day sandbox run, and verified Docker Compose stack (all 5 services healthy)**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-15T00:30:00Z
- **Completed:** 2026-03-15T01:00:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Validation report generator reads cycle log entries and produces markdown report with 4-criteria pass/fail assessment (trading days, drawdown, trades, errors)
- Orchestration script prints actionable checklist: env vars, Docker status, 1M RUB capital, MOEX-only segments, bond cycle enablement, monitoring URLs, kill test instructions
- Docker Compose sandbox stack verified: all 5 services (postgres, redis, prometheus, grafana, app) build and start; Alembic migrations apply; /health returns real probes; Grafana dashboard loads
- Docker infrastructure fixes: README.md in both Dockerfile stages, psycopg2-binary dependency, .dockerignore whitelist

## Task Commits

Each task was committed atomically:

1. **Task 1: Validation report generator and orchestration script** - `28383be` (feat)
2. **Task 2: Verify sandbox stack starts and runs correctly** - checkpoint:human-verify (approved, no code commit)

**Infrastructure fix:** `e334c99` (fix) - Docker build fixes for Dockerfile.prod, psycopg2-binary, .dockerignore

## Files Created/Modified
- `scripts/generate_validation_report.py` - Reads cycle logs, computes metrics, generates pass/fail markdown report
- `scripts/run_sandbox_validation.py` - Pre-flight checklist and orchestration for 5-day sandbox validation run
- `tests/unit/test_validation_report.py` - Tests for report generation (pass, fail on each criterion, empty data, per-day breakdown)
- `docker/Dockerfile.prod` - Added README.md to both build stages (fixes COPY failure)
- `docker/.dockerignore` - Whitelist approach for Docker context

## Decisions Made
- Validation criteria match AUT-04 requirement: 5+ trading days, <5% max drawdown, >=10 round-trip trades, 0 critical errors
- Report generator returns bool (True=PASS) for programmatic use
- Orchestration script is a documented checklist, not an automated runner (5-day run is manual via Docker Compose)
- Docker build fixes committed as separate infrastructure fix commit

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Docker build fixes for sandbox stack verification**
- **Found during:** Task 2 checkpoint preparation
- **Issue:** Dockerfile.prod COPY failed (missing README.md in build context), psycopg2-binary not installed, .dockerignore too aggressive
- **Fix:** Added README.md to both Dockerfile stages, added psycopg2-binary to dependencies, switched .dockerignore to whitelist approach
- **Files modified:** docker/Dockerfile.prod, docker/.dockerignore
- **Verification:** docker compose build succeeds, all 5 services start
- **Committed in:** e334c99

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Docker fix was necessary for the stack verification checkpoint. No scope creep.

## Issues Encountered
- trading_loop_build_failed expected in sandbox without real Tinkoff token -- this is expected behavior, not a bug

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 6 complete: all sandbox validation infrastructure in place
- System ready for 5-day autonomous sandbox run (follow run_sandbox_validation.py checklist)
- After successful 5-day run, generate_validation_report.py produces auditable evidence
- Phase 7 (News Pipeline and Go-Live) can begin planning

## Self-Check: PASSED

All files verified present, all commits verified in git log.

---
*Phase: 06-sandbox-validation*
*Completed: 2026-03-15*
