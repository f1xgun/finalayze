# Quality Grades

Quality is assessed per module domain. Grades are re-evaluated after each phase.

**Grading scale:**
- **A**: Full test coverage, documented, clean interfaces, type-safe
- **B**: Good coverage, documented, minor gaps
- **C**: Partial coverage, some documentation, needs improvement
- **D**: Minimal coverage, undocumented, needs significant work
- **F**: Broken, untested, or missing
- **N/A**: Not yet implemented

## Current Grades (Post Phase 1 Backtest Slice)

| Module | Grade | Coverage | Notes |
|--------|-------|----------|-------|
| `core/schemas.py` | A | 100% | Full Pydantic v2 schemas, fully tested, type-safe |
| `core/models.py` | B | 100% | ORM models defined and schema-tested; not integration-tested against real DB |
| `core/exceptions.py` | A | 100% | All exception types defined and tested |
| `core/db.py` | D | 0% | Async engine/session factory stub; not tested |
| `config/` | C | Partial | Settings, modes, segments defined and tested; settings.py not unit-tested |
| `markets/registry.py` | A | 100% | MarketRegistry fully tested including open/closed logic |
| `data/fetchers/` | B | 100% | Unit-tested with mocks; no live API integration test |
| `analysis/` | N/A | -- | Not implemented |
| `strategies/momentum.py` | B | 80% | Unit-tested; some edge-case branches not covered |
| `ml/` | N/A | -- | Not implemented |
| `risk/` | A | 97% | Fully tested financial safety modules (position sizer, stop-loss, pre-trade checks) |
| `execution/simulated_broker.py` | B | 95% | Unit-tested; simulated only, no live broker integration |
| `backtest/` | B | 89% | Unit-tested engine and performance analyzer; no real-DB integration test |
| `dashboard/` | N/A | -- | Not implemented |
| `api/` | N/A | -- | Not implemented |

## History

| Date | Phase | Changes |
|------|-------|---------|
| 2026-02-21 | Phase 0 | Initial grades assigned |
| 2026-02-22 | Phase 1 Backtest Slice | Updated grades for core, markets, data, strategies, risk, execution, backtest |
