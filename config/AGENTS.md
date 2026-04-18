# config/ — Configuration (Area Node)

Parent: [root AGENTS.md](../AGENTS.md) · Layer 1 · Agent: `config-agent`

## Purpose

Pydantic settings, work-mode enum, per-segment configuration, trading universes, logging setup,
and gate thresholds. Sits at Layer 1 — may import from `core/` (Layer 0) only.

## Files

| File | Purpose |
|---|---|
| `settings.py` | `Settings` (Pydantic BaseSettings, env prefix `FINALAYZE_`). Everything downstream reads this. |
| `modes.py` | `WorkMode` enum (DEBUG/SANDBOX/TEST/REAL), `RolloutPhase`, mode-derived flags. |
| `segments.py` | `SegmentConfig` dataclass + 9 segments (us_tech, us_broad, us_finance, us_healthcare, us_losers, ru_blue_chips, ru_energy, ru_finance, ru_tech). |
| `logging.py` | structlog setup — must be called before any `structlog.get_logger()` at module level (cache_logger_on_first_use=True). |
| `pipelines.yaml` | Autonomous pipeline config: daily review, weekly deep dive, go-live scorecard. |
| `gate_thresholds.yaml` | Go/no-go thresholds consumed by `monitoring/go_no_go.py`. |
| `universes/*.json` | Ticker lists per universe: `moex_blue_chips`, `sp500_sample`, `us_mega`, `us_losers`. |

## Public API

- `Settings`, `get_settings()` (lru_cache) — the single source of truth for runtime config
- `WorkMode`, `RolloutPhase`, `ModeManager`
- `SegmentConfig`, `get_segment(segment_id)`, `all_segments()`

## Contracts & invariants

- Env vars are prefixed `FINALAYZE_` (e.g. `FINALAYZE_TINKOFF_TOKEN`, `FINALAYZE_MODE`).
- All secrets come from `.env` (never checked in) — `.env.example` is the template.
- Segment IDs are lowercase, snake_case, prefixed `us_` or `ru_`.
- `gate_thresholds.yaml` schema is validated at monitoring-startup via `GateThresholds.load()`.
- Universes must be valid JSON arrays of ticker strings; MOEX tickers use base symbol (no `.ME`).

## Testing

- `tests/unit/test_settings.py`, `tests/unit/test_segments.py`, `tests/unit/test_modes.py`
- `uv run pytest tests/unit/test_settings.py tests/unit/test_segments.py -v`

## Common edits

| Task | Edit |
|---|---|
| New env variable | `settings.py` (add field + default) + `.env.example` |
| New trading universe | drop JSON into `universes/` + add segment in `segments.py` |
| Tweak rollout phase thresholds | `modes.py` `RolloutPhase` enum |
| Raise/lower go-live thresholds | `gate_thresholds.yaml` (no code change) |
