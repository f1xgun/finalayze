---
phase: 29-core-stability
plan: 02
subsystem: infra
tags: [promtail, loki, grafana, docker, observability, logging]

requires:
  - phase: 29-core-stability
    provides: "Docker compose sandbox infrastructure"
provides:
  - "Working Promtail -> Loki -> Grafana log pipeline"
  - "30-day log retention with compactor enforcement"
  - "Low-cardinality Promtail labels (container, level only)"
  - "Persistent Promtail positions across container restarts"
affects: [sandbox-monitoring, production-operations]

tech-stack:
  added: []
  patterns:
    - "JSON-aware pipeline_stages for structured log parsing"
    - "Named volumes for stateful sidecar persistence"
    - "Compactor-backed retention enforcement in Loki"

key-files:
  created: []
  modified:
    - docker/docker-compose.sandbox.yml
    - monitoring/promtail/promtail-config.yml
    - monitoring/loki/loki-config.yml

key-decisions:
  - "Removed event label from Promtail labels to prevent stream explosion (50+ unique values)"
  - "Used named volume promtail_positions for positions persistence instead of host bind mount"
  - "Set Loki ingestion limits (10MB/s rate, 20MB burst, 5000 max streams) to prevent burst overload"

patterns-established:
  - "Promtail __path__ must be relabeled from __meta_docker_container_log_path for docker_sd_configs"
  - "Drop stages for health/metrics use regex on raw log line, not parsed JSON fields"

requirements-completed: [OBS-01, OBS-02]

duration: 2min
completed: 2026-03-30
---

# Phase 29 Plan 02: Loki Log Pipeline Summary

**Fixed Promtail->Loki log pipeline with container log volume mounts, __path__ relabeling, low-cardinality labels, and 30-day retention via compactor**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-30T19:34:05Z
- **Completed:** 2026-03-30T19:35:44Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Promtail can now read Docker container logs via /var/lib/docker/containers volume mount
- __path__ correctly resolved from docker_sd_configs metadata so Promtail knows which files to tail
- High-cardinality event label removed from labels (only level and container remain)
- Loki configured with 30-day retention and compactor for enforcement
- Ingestion rate limits protect Loki from burst overload

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix Promtail volume mounts and config** - `abd2cc3` (fix)
2. **Task 2: Configure Loki 30-day retention and ingestion limits** - `9ed608c` (feat)

## Files Created/Modified
- `docker/docker-compose.sandbox.yml` - Added container log volume mount, promtail_positions named volume
- `monitoring/promtail/promtail-config.yml` - Added __path__ relabel, removed event label, fixed drop stage, added timestamp stage, persisted positions
- `monitoring/loki/loki-config.yml` - Added 30-day retention, compactor config, ingestion rate limits

## Decisions Made
- Removed event label from Promtail labels to prevent Loki stream explosion (50+ unique event values would create excessive streams)
- Used named Docker volume for positions persistence rather than host bind mount (portable, managed by Docker)
- Set conservative ingestion limits (10MB/s, 5000 streams) appropriate for single-app sandbox

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None.

## Next Phase Readiness
- Log pipeline configuration complete, ready for deployment verification
- Logs will be queryable in Grafana once sandbox containers are restarted

---
*Phase: 29-core-stability*
*Completed: 2026-03-30*
