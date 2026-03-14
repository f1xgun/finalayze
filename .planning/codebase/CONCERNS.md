# Codebase Concerns

**Analysis Date:** 2026-03-14

## Tech Debt

**G-008: ML Model Accuracy Suboptimal**
- Issue: 16 new features added (cross-asset correlations, regime indicators, calendar effects, z-scores) but models achieve ~57% accuracy best fold. Models fail quality gates and cannot pass Brier/accuracy thresholds for deployment.
- Files: `src/finalayze/ml/features/technical.py` (1059 LOC), `src/finalayze/ml/models/*.py`, `src/finalayze/ml/training/quality_gates.py`, `src/finalayze/ml/training/__init__.py`
- Impact: ML models cannot improve signal quality. `ml_ensemble` remains disabled across all segments. Walk-forward Sharpe still negative (-0.004) despite feature engineering efforts.
- Fix approach: (1) Hyperparameter retuning with Optuna (reduce overfitting guardrails). (2) Feature importance analysis — identify/remove low-signal features. (3) Investigate label leakage or look-ahead bias in 16 new features. (4) Consider ensemble meta-learner recalibration or conformal prediction approach. (5) Backtest each model in isolation to identify worst performer.

**G-009: Trade Count Low — Signal Filtering Too Aggressive**
- Issue: Only 626 trades vs 1300 target. ADX(14) regime routing (trend pool vs MR pool) + confidence threshold _MIN_EXIT_CONFIDENCE=0.38 + entry gate filter out >50% of signals.
- Files: `src/finalayze/strategies/adx.py`, `src/finalayze/strategies/combiner.py` (496 LOC), `src/finalayze/backtest/engine.py` (1431 LOC)
- Impact: Reduced statistical edge due to small trade sample. Harder to validate improvements. Some edge-case opportunities lost.
- Fix approach: (1) Lower _MIN_EXIT_CONFIDENCE to 0.30 (tune sensitivity). (2) Relax ADX pool gating — use weighted hybrid instead of hard cutoff. (3) Review confidence threshold heuristic across all strategies. (4) Analyze signal rejection rate per strategy; identify which filters are most aggressive.

**G-011: ML Ensemble Disabled — Feature Quality or Data Issue**
- Issue: `ml_ensemble` strategy disabled in all presets (`weight: 0.00`, `enabled: false`). Accuracy/Brier failures prevent enabling across us_broad, us_finance, us_healthcare.
- Files: `src/finalayze/strategies/ml_strategy.py`, `src/finalayze/strategies/presets/*.yaml` (ru_blue_chips, ru_energy, ru_finance, us_broad, us_finance, us_healthcare, us_tech), `src/finalayze/ml/training/quality_gates.py`
- Impact: No ensemble signal voting. Reduced signal diversity. Feature quality gains not realized in live/sandbox trading.
- Fix approach: (1) Audit feature generation for NaN/inf/outlier leakage. (2) Rebalance label distribution (Brier threshold 0.25 may be too tight for imbalanced data). (3) Validate walk-forward fold independence — no data leakage. (4) Retrain all segments together on pooled data to increase n_test. (5) Enable with weight=0.05 on us_tech only; iterate.

**G-013: Walk-Forward Sharpe Still Negative**
- Issue: OOS Sharpe=-0.004 despite structural fixes (grace bar, pipeline floor, strategy-specific stops). Suggests regime-dependent strategy underperformance or inadequate regime detection.
- Files: `src/finalayze/backtest/engine.py`, `src/finalayze/backtest/walk_forward.py` (361 LOC), `src/finalayze/risk/regime.py` (444 LOC), `src/finalayze/strategies/adx.py`
- Impact: Live trading returns may be near-zero or negative OOS. Backtest overfitting masks poor generalization. Walk-forward validation not catching regime shifts.
- Fix approach: (1) Break down Sharpe by regime (trend/MR) to isolate weak arm. (2) Test seasonal/calendar effects separately. (3) Add regime-adaptive stop sizing (widen stops in choppy markets). (4) Increase test fold length (currently 6mo; try 12mo OOS). (5) Audit ADX(14) for look-ahead bias or regime lag.

## Known Bugs

**TODO: Portfolio API Endpoint Incomplete**
- Symptoms: `GET /api/v1/portfolio` returns schema but does not execute broker_router. Returns mock data.
- Files: `src/finalayze/api/v1/portfolio.py:98` (line 98 has `# TODO: wire to real broker_router`)
- Trigger: Call `/api/v1/portfolio` when broker has open positions.
- Workaround: Query `/api/v1/positions` instead; does connect to live broker state.
- Impact: Dashboard portfolio widget shows stale/mock values; not critical for trading since execution uses separate BrokerRouter path.

**TODO: Candle Immutability Not Enforced**
- Symptoms: Design specifies `tuple[Candle, ...]` but implementation uses `list[Candle]`. Code mutates candle lists in-place.
- Files: `src/finalayze/core/schemas.py:line 48` (Candle list field has comment "TODO: Design specifies tuple...")
- Trigger: Any code that sorts/filters candle lists modifies the original list-of-lists.
- Workaround: Treat candle lists as immutable; create fresh lists where filtering needed.
- Impact: Low risk currently since candle mutations are localized. Will cause bugs if multi-threaded candle analysis is added.
- Fix approach: Convert all candle storage to tuples; use `tuple[Candle, ...]` in schemas. Requires audit of data fetchers and backtest engine.

**Event Classifier JSON Parsing Fragile**
- Symptoms: If LLM returns malformed JSON or unexpected field names, fallback to plain-text parsing that may misclassify event type.
- Files: `src/finalayze/analysis/event_classifier.py:66-99` (try/except with generic ValueError catch; silent fail)
- Trigger: Any deviation from expected `{"event_types": [....]}` JSON structure.
- Workaround: Currently silent; returns EventType.OTHER if parsing fails.
- Impact: Medium risk. Event-driven strategy disabled anyway, so impact is low. But if re-enabled, event misclassification could reduce alpha.
- Fix approach: (1) Add structured logging on JSON parse failures. (2) Validate response schema with Pydantic. (3) Implement retry with different prompt format.

## Security Considerations

**Async Event Loop Threading Complexity**
- Risk: TradingLoop runs APScheduler with both sync and async tasks. Sentiment cache protected by threading.Lock; stop-loss state protected by separate lock. Potential deadlock or race condition if async tasks call sync functions that re-acquire locks.
- Files: `src/finalayze/core/trading_loop.py:lines 20-50, 440-490` (threading.Lock + asyncio mixing, _sentiment_lock, _stop_loss_lock)
- Current mitigation: Comments document thread safety; RWLock not used (only reader tasks). Minimal lock contention in practice since sentiment analysis is slow.
- Recommendations: (1) Audit all lock-holding code paths for reentrancy. (2) Use asyncio.Lock instead of threading.Lock for async code paths. (3) Add unit test for concurrent sentiment cache reads + stop-loss updates.

**Tinkoff gRPC Endpoint Hardcoding**
- Risk: Target endpoint `invest-public-api.tbank.ru:443` (and sandbox equivalent) hardcoded in code. If T-Bank changes endpoint domain or DNS resolution, data fetcher breaks without warning.
- Files: `src/finalayze/data/fetchers/tinkoff_data.py` (AsyncClient(target=...) passed explicitly)
- Current mitigation: Already fixed in codebase (target parameter passed, not using old domain). GRPC_DNS_RESOLVER=native set to work around gRPC C-ares resolver issues.
- Recommendations: (1) Make endpoint configurable via `config/settings.py` FINALAYZE_TINKOFF_TARGET env var. (2) Add endpoint health check before fetcher init. (3) Test with sandbox endpoint on every backtest run.

**API Key Not Rotated**
- Risk: X-API-Key header used for auth on all `/api/v1/*` endpoints. No key rotation mechanism; if key leaked, attacker has full access until manual rotation.
- Files: `src/finalayze/api/v1/auth.py` (static key comparison)
- Current mitigation: Key stored in .env (not committed). FastAPI protection: list of allowed keys with hash comparison.
- Recommendations: (1) Implement key versioning / deprecation workflow. (2) Add rate limiting per API key. (3) Log all auth failures to detect brute-force attempts.

**gRPC Certificate Path Hardcoded**
- Risk: `_GRPC_ROOTS = _PROJECT_ROOT / "certs" / "grpc_roots.pem"` hardcoded. If cert file missing or corrupted, Tinkoff fetcher fails silently on SSL verification.
- Files: `src/finalayze/data/fetchers/tinkoff_data.py`
- Current mitigation: certs/ directory committed with pem file. No fallback if cert invalid.
- Recommendations: (1) Add cert validity check at startup. (2) Embed root certs in code or load from system trust store. (3) Add monitoring alert if SSL verification fails.

## Performance Bottlenecks

**Backtest Engine Large (1431 LOC)**
- Problem: Single file handles candle iteration, position tracking, risk management, stop-loss, trailing stop, grace bar, catastrophic drop detection, pre-trade checks, Kelly sizing, position sizing pipeline, and journaling. Hard to test individual logic.
- Files: `src/finalayze/backtest/engine.py` (1431 LOC)
- Cause: Layered responsibilities combined for performance (no intermediate allocations).
- Improvement path: (1) Extract position state tracking into BacktestPortfolio class. (2) Extract stop-loss/trailing logic into StopLossManager. (3) Keep candle iteration loop in main engine but delegate decision-making.

**Technical Features 1059 LOC — Compute Intensive**
- Problem: 45 technical indicators (RSI, MACD, Bollinger, wavelets, Amihud illiquidity, cross-asset correlations) computed per bar. Lookback windows up to 252 bars. Slow for large backtests or real-time computation.
- Files: `src/finalayze/ml/features/technical.py` (1059 LOC)
- Cause: pandas + numpy operations not vectorized; per-bar indicator updates. No caching of rolling windows.
- Improvement path: (1) Cache rolling window objects (pandas.rolling) at symbol level. (2) Vectorize Amihud calculation for multiple symbols. (3) Profile hotspots; consider numba JIT for ROC/OBV slope. (4) Cache wave decomposition results.

**Copula Risk Model Correlation Matrix Computation**
- Problem: Copula construction recomputes 252-bar correlation matrix on every position size update. Matrix inversion O(n^3) for n instruments.
- Files: `src/finalayze/risk/copula.py` (373 LOC)
- Cause: Called in position sizing pipeline per trade; no correlation stability check.
- Improvement path: (1) Cache correlation matrix; only recompute on new day. (2) Use lower-rank approximation (PCA) for >10 instruments. (3) Skip copula if <5 open positions.

**Walk-Forward Fold Splits Recalculate Train/Test Data**
- Problem: Walk-forward splits (12-month train / 6-month test folds) recalculate candle normalization, label generation, feature engineering for each fold. Entire feature matrix rebuilt 5+ times.
- Files: `src/finalayze/backtest/walk_forward.py` (361 LOC), `src/finalayze/ml/training/labeling.py` (357 LOC)
- Cause: Fold independence requires no data leakage; features/labels must be recomputed in isolation per fold.
- Improvement path: (1) Pre-compute all features and labels once; store in parquet. (2) Memmap features to avoid memory duplication. (3) Parallelize fold processing (currently sequential).

## Fragile Areas

**ADX Regime Routing Logic**
- Files: `src/finalayze/strategies/adx.py` (444 LOC)
- Why fragile: Hard cutoff at ADX=30 (trend) and ADX=20 (MR). Between 20-30, signal dominates (winner-take-all). If ADX oscillates near threshold, strategy weights flip rapidly, causing whipsaw.
- Safe modification: (1) Use weighted blend (ADX as weight) instead of hard cutoff. (2) Add hysteresis (20-22 for exit, 28-30 for entry). (3) Validate with strategy-isolation backtest after any ADX parameter change.
- Test coverage: ADX logic in `test_strategy_adx.py`; add tests for threshold oscillation cases.

**Grace Bar Exception Logic**
- Files: `src/finalayze/backtest/engine.py:lines 600-650` (estimated)
- Why fragile: Grace bar skips stop-loss check on entry candle (entry_bars[sym]+1 == i). Also has special case for _CATASTROPHIC_DROP_PCT=15% that forces exit even on grace bar. Two separate stop mechanisms (ATR + catastrophic drop) interact.
- Safe modification: (1) Extract both stop checks into a `should_force_exit()` method. (2) Add unit test for each stop condition independently. (3) Test grace bar + catastrophic drop interaction with isolated candle sequences.
- Test coverage: `test_engine_grace_bar.py`; add edge case for grace bar + 15% drop.

**Position Sizing Pipeline Ordering**
- Files: `src/finalayze/risk/position_sizing_pipeline.py` (multi-step: KellyStep → VolTargetStep → EVTStep → RegimeStep → CopulaStep → MetaLabelStep → HardCapsStep)
- Why fragile: Each step modifies position size; order matters. If pipeline floor (15%) or hard caps change, downstream steps may produce unintuitive results (e.g., VolTargetStep reduces size below pipeline floor, then pipeline floor re-applies).
- Safe modification: (1) Document required ordering. (2) Add assertions that each step respects previous caps. (3) Add unit test for pipeline with all steps in wrong order; should fail.
- Test coverage: `test_position_sizer.py`; add ordering invariant tests.

**ML Model Serialization — Model Weights Mismatch**
- Files: `src/finalayze/ml/registry.py`, `models/<segment>/` (xgb.pkl, lgbm.pkl, catboost.pkl, calibrator.pkl, meta_learner.pkl, model_weights.json, segment_meta.json, selected_features.json)
- Why fragile: Multiple pickle files + JSON metadata. If selected_features.json out of sync with model input dimensions, inference fails at runtime with cryptic error. No validation on load.
- Safe modification: (1) Wrap model load in schema validation. (2) Check feature count matches model input. (3) Version model artifacts (add version field to segment_meta.json). (4) Test load/save cycle with size check.
- Test coverage: Add test_model_serialization.py with mismatch scenarios.

## Scaling Limits

**In-Memory Candle Buffer**
- Current capacity: ~252 bars per instrument × 100 instruments × 3 timeframes = 75,600 candles in memory.
- Limit: If market grows to 500 instruments and 8 timeframes, buffer = 1M candles. Current dict-based storage will consume >1GB RAM. Backtest engine will slow on candle lookups.
- Scaling path: (1) Use DuckDB or Parquet for candle storage (lazy load per symbol). (2) Implement disk-backed LRU cache. (3) Batch candle fetches by timeframe instead of symbol-by-symbol.

**Real-Time News Sentiment Cache**
- Current capacity: 1000s of news articles per day, cached in memory with threading.Lock. Indefinite growth over weeks.
- Limit: After 1M articles, sentiment_cache dict grows to 100s of MB. Lock contention on every sentiment lookup.
- Scaling path: (1) Replace in-memory dict with Redis cache (already integrated). (2) Set TTL on cached sentiment (7 days). (3) Use async-safe queue instead of lock.

**ML Model Training on Full History**
- Current capacity: 3+ years × 252 trading days × 100 symbols × 45 features = 34M feature vectors. Training XGBoost on 34M rows takes hours.
- Limit: Adding more symbols or features pushes training into >24hr range. Walk-forward retraining becomes infeasible on daily basis.
- Scaling path: (1) Implement sliding-window training (drop oldest 1 year, add newest 1 year). (2) Parallel fold training across CPU cores. (3) Feature subsampling per model (train on 30 of 45 features per tree).

**Alpaca + Tinkoff Order Latency**
- Current capacity: Backtest engine handles 100 symbols × 10 positions = 1000 concurrent orders at portfolio level. Broker API (Alpaca/Tinkoff) rate-limited.
- Limit: Alpaca: 200 orders/min. Tinkoff: ~100 orders/min. If trading loop tries to rebalance all 1000 positions daily, will hit rate limits.
- Scaling path: (1) Order batching — submit 10-position chunks sequentially. (2) Weighted portfolio filtering — only rebalance top 20 positions by notional. (3) Time-spread orders across trading hours (avoid market open spike).

## Dependencies at Risk

**T-Tech (Tinkoff API SDK) — Proprietary gRPC Binding**
- Risk: SDK installed from custom index (`t-tech-investments` on tbank custom uv index). May be unmaintained or break on Python/protobuf version upgrades.
- Impact: MOEX data fetcher fails if SDK incompatible. No fallback to REST API (Tinkoff REST only works for cash, not futures/bonds).
- Migration plan: (1) Evaluate tinkoff-invest-api (community alternative). (2) Implement homegrown gRPC client using tinkoff-public-invest-api protobuf specs. (3) Fallback to MOEX ISS REST for candles (lower freq, 1-day only).

**XGBoost + LightGBM — ML Framework Lock-In**
- Risk: Models trained with sklearn-compatible interfaces. If sklearn breaks compatibility or these libraries diverge, retraining models becomes difficult.
- Impact: Cannot easily swap in alternatives (e.g., RF, Neural Networks) without code refactor.
- Migration plan: (1) Wrap all model interfaces in BaseModel abstract class (already done in `src/finalayze/ml/models/base.py`). (2) Test model swaps with unit tests. (3) Keep LSTM model for fallback.

**APScheduler — Background Job Orchestration**
- Risk: Deprecated in favor of async-native alternatives (e.g., schedule library). May not support Python 3.13+ long-term.
- Impact: TradingLoop dependent on APScheduler for news cycle, strategy cycle, daily reset scheduling. Switching requires rewrite of scheduling logic.
- Migration plan: (1) Evaluate Python 3.13 compatibility of APScheduler. (2) Add compatibility test in CI. (3) Design TradingLoop to allow scheduler swaps (dependency injection).

## Missing Critical Features

**Real-Time News Feed Integration**
- Problem: `event_driven` strategy disabled. NewsAnalyzer + EventClassifier exist but no live news feed integrated.
- Blocks: Cannot implement earnings surprise strategies, geopolitical event trading, or macro news alpha.
- Solution: Integrate NewsAPI (already fetcher exists). Add scheduled news cycle to TradingLoop. Rate-limit LLM sentiment analysis.

**Dividend Adjustment Pipeline (Incomplete)**
- Problem: Dividend-adjusted prices missing for some MOEX instruments. Static YAML fallback in place but not comprehensive.
- Blocks: Accurate backtest metrics for dividend-heavy portfolios (Gazprom, Sber). Live dividend calendar not fetched.
- Solution: Implement T-Bank corporate actions API integration. Fetch forward dividend calendar from Tinkoff. Validate against static YAML.

**Circuit Breaker Optimization**
- Problem: Circuit breakers exist (stop-loss limits, daily loss limits) but not dynamically calibrated to market volatility.
- Blocks: In high-vol markets, circuit breaker thresholds too tight; in low-vol, too loose.
- Solution: Implement Vol-adjusted circuit breaker thresholds (scale limits by realized volatility vs baseline).

## Test Coverage Gaps

**Configuration Module (config/settings.py)**
- What's not tested: Settings validation; env var parsing; missing required vars behavior.
- Files: `config/settings.py`, `config/modes.py`, `config/segments.py`
- Risk: Settings misconfiguration could cause silent failures (e.g., wrong database URL, unset API keys).
- Priority: **High** — Config is layer 1 dependency; all failures propagate downstream.
- Test plan: Add `test_config.py` with schema validation tests, env var override tests, missing-var error tests.

**Database Layer (core/db.py)**
- What's not tested: Engine pooling, session cleanup, transaction rollback on exception.
- Files: `src/finalayze/core/db.py` (Grade D, 0% coverage)
- Risk: Database leaks (unclosed sessions); deadlocks on concurrent access; stale connections.
- Priority: **High** — Any DB issue breaks backtest and live trading.
- Test plan: Add integration tests with real PostgreSQL + TimescaleDB. Test session cleanup, pool exhaustion, rollback scenarios.

**Async Event Loop + Threading Interaction**
- What's not tested: TradingLoop mixing APScheduler sync tasks + asyncio coroutines + threading locks. Potential race condition in sentiment cache + stop-loss updates.
- Files: `src/finalayze/core/trading_loop.py` (1108 LOC, complex async/sync boundary)
- Risk: Deadlock or data race if multiple threads call _update_stop_loss while sentiment analysis running.
- Priority: **High** — Concurrency bugs hard to reproduce and catastrophic in production.
- Test plan: Add stress test with concurrent sentiment updates + stop-loss modifications. Use thread-safety checker (e.g., ThreadSanitizer).

**Backtest Engine Grace Bar Edge Cases**
- What's not tested: Grace bar interaction with catastrophic drop; grace bar on multiple consecutive fills; grace bar with multi-leg orders.
- Files: `src/finalayze/backtest/engine.py`
- Risk: Stop-loss logic on grace bar may not trigger correctly under edge conditions, causing unintended positions.
- Priority: **Medium** — Edge case risk, but directly impacts risk management.
- Test plan: Add parametrized tests for: (1) entry + catastrophic drop same bar, (2) grace bar + second fill same day, (3) grace bar + volatility spike.

**Position Sizing Pipeline Interactions**
- What's not tested: Full pipeline ordering; interactions between Kelly sizing + vol targeting + regime scaling + copula reduction + hard caps.
- Files: `src/finalayze/risk/position_sizing_pipeline.py`, `src/finalayze/risk/kelly.py`
- Risk: Pipeline step order change could silently alter position sizes (e.g., Kelly oversizing then hard cap truncates, vs hard cap first then Kelly underfits).
- Priority: **Medium** — Affects position sizing accuracy, impacts risk/reward.
- Test plan: Add unit tests for all (step1, step2) pairs. Test that final size respects all constraints.

---

*Concerns audit: 2026-03-14*
