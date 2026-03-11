# Quality Grades

Quality is assessed per module domain. Grades are re-evaluated after each phase.

**Grading scale:**
- **A**: Full test coverage, documented, clean interfaces, type-safe
- **B**: Good coverage, documented, minor gaps
- **C**: Partial coverage, some documentation, needs improvement
- **D**: Minimal coverage, undocumented, needs significant work
- **F**: Broken, untested, or missing
- **N/A**: Not yet implemented

## Current Grades (Post Phase 5 Week 5)

| Module | Grade | Coverage | Notes |
|--------|-------|----------|-------|
| `core/schemas.py` | A | 100% | Full Pydantic v2 schemas, type-safe |
| `core/models.py` | B | 100% | ORM models tested; no real DB integration test |
| `core/exceptions.py` | A | 100% | 12 exception classes |
| `core/modes.py` | B | ~90% | WorkMode enum, ModeManager |
| `core/clock.py` | B | ~90% | RealClock + SimulatedClock |
| `core/events.py` | B | ~85% | Redis Streams EventBus |
| `core/db.py` | D | 0% | Async engine/session factory stub |
| `config/` | B | Partial | Settings, modes, segments, logging — all implemented |
| `markets/registry.py` | A | 100% | MarketRegistry fully tested |
| `markets/schedule.py` | B | ~90% | US + MOEX schedule guards |
| `data/fetchers/` | B | 100% | Finnhub + YFinance + Tinkoff fetchers unit-tested |
| `data/rate_limiter.py` | B | ~95% | Token bucket, async acquire |
| `data/normalizer.py` | B | ~95% | OHLCV validation, batch mode |
| `analysis/llm_client.py` | C | ~70% | OpenRouter/OpenAI/Anthropic with cache + retry |
| `analysis/news_analyzer.py` | C | ~70% | EN/RU sentiment analysis via LLM |
| `analysis/event_classifier.py` | C | ~65% | EventType StrEnum classification |
| `analysis/impact_estimator.py` | C | ~65% | Scope routing for impact |
| `strategies/momentum.py` | B | ~85% | RSI+MACD, trend/ADX/volume filters |
| `strategies/mean_reversion.py` | B | ~85% | Bollinger Bands, per-segment params |
| `strategies/dual_momentum.py` | B | ~80% | Cross-asset momentum, 1m/3m/6m lookbacks |
| `strategies/rsi2_connors.py` | B | ~80% | 2-period RSI, SMA trend filter |
| `strategies/ou_mean_reversion.py` | B | ~75% | Ornstein-Uhlenbeck, MLE estimation |
| `strategies/pairs.py` | B | ~75% | Cointegration gate, OLS beta |
| `strategies/combiner.py` | B+ | ~90% | ADX routing, DRY hooks, journaling |
| `strategies/adx.py` | B | ~85% | Regime routing (trend/MR pools) |
| `ml/features.py` | C | ~60% | 28 base + 16 new features (cross-asset, regime, calendar, z-scores) |
| `ml/labels.py` | C | ~60% | Triple-barrier, market-neutral labels |
| `ml/models/` | C- | ~55% | XGBoost + LightGBM + LSTM + Ensemble; accuracy suboptimal (~57%) |
| `ml/registry.py` | C | ~60% | Per-segment model storage |
| `ml/quality_gates.py` | C | ~50% | Brier validation, feature importance budget |
| `risk/` | A | 97% | Kelly sizer, ATR stops, 11-check pre-trade pipeline |
| `execution/simulated_broker.py` | B | 95% | Fill at open, stop-loss, trailing stops |
| `execution/alpaca_broker.py` | B | ~80% | Paper/live via alpaca-py, RetryPolicy |
| `execution/tinkoff_broker.py` | B | ~75% | Sandbox/live via t-tech, lot-size aware |
| `execution/broker_router.py` | B | ~85% | Multi-market dispatch |
| `backtest/engine.py` | B+ | ~90% | Grace bar, strategy-specific stops, walk-forward |
| `backtest/performance.py` | B | ~85% | Sharpe, PF, drawdown, Monte Carlo |
| `api/` | B+ | ~80% | 20+ endpoints, X-API-Key auth, Prometheus metrics |
| `dashboard/` | B | ~60% | Streamlit 5-page dashboard |

## History

| Date | Phase | Changes |
|------|-------|---------|
| 2026-02-21 | Phase 0 | Initial grades assigned |
| 2026-02-22 | Phase 1 | Core, markets, data, strategies, risk, execution, backtest graded |
| 2026-02-23 | Phase 2 | Analysis C, ML scaffold F→C-, Tinkoff fetcher added |
| 2026-02-25 | Phase 3 | LSTM, pairs, circuit breakers, E2E tests. 619 tests |
| 2026-02-28 | Phase 4 | API B+, dashboard B, execution B, Redis caching. 949 tests |
| 2026-03-08 | Phase 5 | ML C- (16 features, pipeline exists, accuracy WIP). Strategies B (8 total). ADX routing. 2325 tests |
