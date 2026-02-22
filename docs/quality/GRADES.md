# Quality Grades

Quality is assessed per module domain. Grades are re-evaluated after each phase.

**Grading scale:**
- **A**: Full test coverage, documented, clean interfaces, type-safe
- **B**: Good coverage, documented, minor gaps
- **C**: Partial coverage, some documentation, needs improvement
- **D**: Minimal coverage, undocumented, needs significant work
- **F**: Broken, untested, or missing
- **N/A**: Not yet implemented

## Current Grades (Post Phase 1)

| Module | Grade | Coverage | Notes |
|--------|-------|----------|-------|
| `core/schemas.py` | A | 100% | Full Pydantic v2 schemas, fully tested, type-safe |
| `core/models.py` | B | 100% | ORM models defined and schema-tested; not integration-tested against real DB |
| `core/exceptions.py` | A | 100% | 12 exception classes defined and tested |
| `core/modes.py` | B | ~90% | WorkMode enum, ModeManager, real-mode guard — tested, strict mypy |
| `core/clock.py` | B | ~90% | RealClock + SimulatedClock — tested |
| `core/events.py` | B | ~85% | Redis Streams EventBus, MarketDataEvent, SignalEvent — tested |
| `core/db.py` | D | 0% | Async engine/session factory stub; not tested |
| `config/` | B | Partial | Settings, modes, segments, structlog logging — all implemented; settings.py not fully unit-tested |
| `markets/registry.py` | A | 100% | MarketRegistry fully tested including open/closed logic |
| `markets/schedule.py` | B | ~90% | US 09:30-16:00 ET + MOEX weekday guards — tested |
| `data/fetchers/` | B | 100% | FinnhubFetcher + YFinanceFetcher unit-tested with mocks; no live API integration test |
| `data/rate_limiter.py` | B | ~95% | Token bucket, async acquire — tested |
| `data/normalizer.py` | B | ~95% | OHLCV validation, batch mode — tested |
| `analysis/` | F | -- | Not started |
| `strategies/momentum.py` | B | 80% | Unit-tested; some edge-case branches not covered |
| `strategies/mean_reversion.py` | B | ~85% | Bollinger Bands, per-segment params — tested |
| `strategies/combiner.py` | B | ~90% | Weighted ensemble, YAML preset loading — tested |
| `ml/` | F | -- | Not started |
| `risk/` | A | 97% | Kelly sizer, ATR stop-loss, pre-trade checks (11 checks) — 95%+ coverage, strict mypy |
| `execution/simulated_broker.py` | B | 95% | Fill at candle open, stop-loss, portfolio tracking — unit-tested; no live broker |
| `backtest/` | B | 89% | BacktestEngine + PerformanceAnalyzer — unit-tested; no real-DB integration test |
| `api/` | C | -- | Health + mode endpoints only, no auth, no tests yet |
| `dashboard/` | F | -- | Not started |

## History

| Date | Phase | Changes |
|------|-------|---------|
| 2026-02-21 | Phase 0 | Initial grades assigned |
| 2026-02-22 | Phase 1 Backtest Slice | Updated grades for core, markets, data, strategies, risk, execution, backtest |
| 2026-02-22 | Phase 1 Full Update | Added grades for core/modes, core/clock, core/events, markets/schedule, data/rate_limiter, data/normalizer, strategies/mean_reversion, strategies/combiner; updated api/ to C |
