# Finalayze Improvement Plan

**Date:** 2026-03-01
**Sources:** Quant Analyst, Risk Officer, ML Engineer, Systems Architect audits
**Total findings:** 112 across 4 domains

---

## Executive Summary

Four domain expert audits identified **3 critical (P0)** issues that must be fixed before any live trading, **~30 high-priority (P1)** issues affecting safety/correctness, and **~50 medium/low-priority** improvements. The system has solid foundations but has significant gaps in: live trading safety (no stop-losses set), async architecture (repeated `asyncio.run()`), data integrity (Tinkoff cash=equity bug), ML pipeline (LSTM buffer contamination, temporal ordering), and observability (metrics defined but not wired).

---

## Phase 5: Critical Safety Fixes (P0)

**Must be fixed before ANY live/sandbox trading.**

| # | Finding | Source | File(s) | Complexity |
|---|---------|--------|---------|------------|
| 5.1 | Stop-losses never set in live trading — `_stop_loss_prices` is never populated after order fill | Risk | `core/trading_loop.py` | M |
| 5.2 | TinkoffBroker `cash=equity` — `PortfolioState.cash` set to total portfolio value, defeating cash sufficiency and exposure checks | Risk | `execution/tinkoff_broker.py` | M |
| 5.3 | Cross-market exposure mixes USD/RUB without currency conversion — 80% limit is meaningless | Risk | `core/trading_loop.py` | M |
| 5.4 | `asyncio.run()` called repeatedly from APScheduler threads — creates new event loops per call, will crash inside existing loop | Arch | `core/trading_loop.py` | M |
| 5.5 | Pairs cointegration uses full history including current bar — look-ahead bias | Quant | `strategies/pairs.py` | L |
| 5.6 | LSTM feature buffer shared across symbols — data contamination | ML | `ml/models/lstm_model.py` | M |
| 5.7 | Multi-symbol training aggregation breaks temporal ordering | ML | `ml/training/__init__.py` | M |
| 5.8 | No temporal gap between train/test in `train_models.py` script | ML | `scripts/train_models.py` | S |

### Implementation plan

**5.1 — Wire stop-losses in TradingLoop:**
- After successful order fill in `_process_instrument()`, call `compute_atr_stop_loss()` with market-specific multiplier (US=2.0, MOEX=2.5)
- Store result in `self._stop_loss_prices[symbol]`

**5.2 — Fix TinkoffBroker portfolio:**
- Parse actual cash (RUB money position) separately from `total_amount_portfolio`
- Map `total_amount_portfolio` → equity, cash positions → cash

**5.3 — Add currency conversion:**
- Create `CurrencyConverter` service with USD/RUB rate (from Tinkoff or CBR API)
- Normalize all cross-market calculations to base currency (USD)
- Add `FXRateProvider` interface for testability

**5.4 — Fix async/sync boundary:**
- Create a single event loop in a dedicated thread at `TradingLoop` startup
- Replace all `asyncio.run()` with `asyncio.run_coroutine_threadsafe(coro, loop)`
- OR use synchronous Redis (`redis.Redis`) + synchronous Tinkoff wrappers in the TradingLoop path

**5.5 — Fix pairs look-ahead:**
- Use rolling window (last 252 bars) for cointegration test, beta, spread stats
- Exclude current bar from statistics computation

**5.6 — Fix LSTM buffer:**
- Option A: per-symbol buffers keyed by `(segment_id, symbol)`
- Option B: require caller to pass full feature sequence, remove stateful buffer

**5.7 — Fix temporal ordering:**
- Attach timestamps to features in `build_dataset`, sort aggregated dataset by time
- OR split per-symbol first, then merge chronologically

**5.8 — Add gap in train_models.py:**
- Add `gap = _WINDOW_SIZE` between train/test splits, matching `trading_loop.py`

---

## Phase 6A: Risk & Compliance (P1)

| # | Finding | Source | Complexity |
|---|---------|--------|------------|
| 6A.1 | Mode gate (check #3) not enforced — DEBUG mode could send real orders | Risk | S |
| 6A.2 | Sector/segment concentration (40% limit) not implemented | Risk | M |
| 6A.3 | Min cash reserve (20%) not enforced in pre-trade check | Risk | S |
| 6A.4 | Cross-market exposure computed per-market only, not aggregated | Risk | S |
| 6A.5 | Circuit breaker L2 auto-resets without requiring 2 profitable days | Risk | S |
| 6A.6 | Circuit breaker de-escalates intraday on equity recovery | Risk | S |
| 6A.7 | PDT tracker not wired in TradingLoop + no day trade detection | Risk | M |
| 6A.8 | gRPC errors not in Tinkoff retry policy | Risk | S |
| 6A.9 | Partial fills may be mis-reported as full fills (Alpaca) | Risk | S |
| 6A.10 | Weekly loss limit reset not wired in live TradingLoop | Risk | S |
| 6A.11 | Kelly sizes against available cash, not portfolio equity | Risk | S |

### Implementation plan

- Add `WorkMode` gate in `_strategy_cycle` — reject order if mode is DEBUG
- Add segment concentration check to `PreTradeChecker` with configurable 40% cap
- Add post-trade cash reserve check: reject if `(cash - order_value) / equity < 0.20`
- Make circuit breaker levels "sticky" per day — only reset via `reset_daily()`
- Add `_profitable_days` counter; require 2 consecutive profitable days for L2→NORMAL
- Wire `PDTTracker` in TradingLoop, add `_is_day_trade()` detection method
- Add `grpc.RpcError` to retryable exceptions in Tinkoff retry config
- Fix `filled_qty` fallback: use 0 not `order.quantity` when fill qty is None
- Wire `reset_week()` in `_daily_reset()` when `weekday == 0` (Monday)
- Change Kelly to size against `portfolio.equity`, not `available_cash`

---

## Phase 6B: Strategy Quality (P1)

| # | Finding | Source | Complexity |
|---|---------|--------|------------|
| 6B.1 | Momentum `_SignalState` shared across segments — state leak | Quant | M |
| 6B.2 | Combiner normalizes by firing-strategy weight, not total configured weight | Quant | M |
| 6B.3 | Mean reversion has no exit-at-mean signal | Quant | M |
| 6B.4 | MOEX commission model is 30x too low (per-share vs per-value) | Quant | M |
| 6B.5 | Walk-forward discards training data, no actual optimization | Quant | M |
| 6B.6 | No portfolio-level backtest simulation | Quant | L |
| 6B.7 | No volatility-adjusted position sizing | Quant | M |
| 6B.8 | Direct `broker._cash` mutation in backtest engine | Quant | S |

### Implementation plan

- Make `_SignalState` per-segment (dict keyed by segment_id)
- Add combiner option: normalize by total enabled weight or only firing weight (configurable)
- Add mean-band-crossover exit signal in mean reversion strategy
- Add `commission_rate` field to `TransactionCosts`, compute MOEX as `price * qty * 0.0003`
- Implement actual walk-forward optimization with parameter grid search on train window
- Extend `BacktestEngine.run()` to accept multiple symbols for portfolio simulation
- Add vol-scaling: `position_size *= target_vol / asset_vol`
- Add `deduct_fees(amount)` method to `SimulatedBroker`

---

## Phase 6C: ML Pipeline (P1)

| # | Finding | Source | Complexity |
|---|---------|--------|------------|
| 6C.1 | Only 6 features — insufficient diversity | ML | M |
| 6C.2 | ATR and MACD histogram not normalized by price level | ML | S |
| 6C.3 | No regularization in XGBoost/LightGBM | ML | S |
| 6C.4 | Ensemble predict_proba does not catch per-model exceptions | ML | S |
| 6C.5 | LSTM: no early stopping, no gradient clipping | ML | M |
| 6C.6 | LSTM: no dropout or weight decay | ML | S |
| 6C.7 | Validation gate uses only accuracy at 52% | ML | M |
| 6C.8 | No corporate action (split/dividend) handling | ML | M |
| 6C.9 | LSTM save is not atomic in loader.py | ML | S |

### Implementation plan

- Add 10+ features: ROC, Williams %R, ADX, MA slope, historical vol, Garman-Klass vol, day-of-week
- Normalize: `atr_pct = atr / close`, `macd_hist_pct = macd_hist / close`
- Add `reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8` to tree models
- Wrap each model's `predict_proba` in try/except within ensemble, log and skip on error
- Add early stopping via validation loss, `torch.nn.utils.clip_grad_norm_(params, 1.0)`
- Add `dropout=0.2` to LSTM, `weight_decay=1e-4` to Adam
- Add Brier score + log-loss to validation gate; raise threshold or use expected-profit metric
- Verify yfinance returns adjusted data; add split-detection validation
- Implement atomic save for LSTM (temp + rename pattern matching XGB/LGBM)

---

## Phase 6D: Architecture & Security (P1)

| # | Finding | Source | Complexity |
|---|---------|--------|------------|
| 6D.1 | Tinkoff broker/fetcher create new event loop + gRPC channel per call | Arch | M |
| 6D.2 | `_stop_loss_prices` dict not thread-safe | Arch | S |
| 6D.3 | `asyncio.run()` inside `_sentiment_lock` can block other threads | Arch | S |
| 6D.4 | No explicit DB connection pool sizing | Arch | S |
| 6D.5 | Redis connection never closed on shutdown | Arch | S |
| 6D.6 | CORS wildcard `*` default in production | Arch | S |
| 6D.7 | API key comparison not timing-safe (`!=` vs `hmac.compare_digest`) | Arch | S |
| 6D.8 | 8/9 logging modules use stdlib instead of structlog | Arch | M |
| 6D.9 | MetricsCollector defined but never called from TradingLoop | Arch | M |
| 6D.10 | `latest` Docker tag for TimescaleDB in production | Arch | S |
| 6D.11 | No TLS in production deployment | Arch | M |
| 6D.12 | Default DB password `secret` in production config | Arch | S |
| 6D.13 | `joblib.load()` without integrity verification for ML models | Arch | M |
| 6D.14 | Multiple redundant `get_portfolio()` broker API calls per cycle | Arch | M |
| 6D.15 | No FX rate tracking service | Risk | M |

### Implementation plan

- Maintain persistent Tinkoff async client with connection reuse
- Add `threading.Lock` to `_stop_loss_prices` access
- Move Redis writes outside `_sentiment_lock`
- Set `pool_size=10, max_overflow=5, pool_timeout=30, pool_recycle=1800` on engine
- Add `close()` method to TradingLoop for cleanup; call on shutdown
- Default CORS to empty list; require explicit `FINALAYZE_CORS_ORIGINS`
- Replace `if key != expected` with `if not hmac.compare_digest(key, expected)`
- Migrate all `logging.getLogger` to `structlog.get_logger`
- Wire `MetricsCollector` calls in `_submit_order`, `_strategy_cycle`, `_daily_reset`
- Pin TimescaleDB to specific version tag
- Add TLS termination via nginx or Traefik
- Remove default passwords; require env vars (fail on startup if missing)
- Add HMAC verification of model files before `joblib.load()`
- Cache `get_portfolio()` per market per strategy cycle

---

## Phase 7: Medium Priority Improvements (P2)

### Strategy & Backtest
- Momentum: fix histogram "rising" check (use `current > prev`, not `current > min`)
- Momentum: improve confidence formula dynamic range
- Event-driven: use declared event_types for filtering/weighting
- Event-driven: wire credibility parameter from YAML
- Pairs: add half-life filter (reject pairs with OU half-life > 30 days)
- Mean reversion: add YAML parameter caching (like momentum)
- Backtest: risk-free rate per market (US ~4%, MOEX ~21%)
- Backtest: fix Sortino to return negative values (not clamp to 0)
- Backtest: fix beta covariance consistency (both pop or both sample)
- Monte Carlo: implement block bootstrap (preserve serial dependence)
- Monte Carlo: fix Sharpe annualization for trade frequency
- Walk-forward: compute max drawdown from equity snapshots, not trade P&L
- Slippage: add volume-dependent market impact model
- No short selling capability in simulated broker
- No survivorship bias protection in backtests
- No regime detection (bull/bear/sideways/high-vol)

### ML Pipeline
- NaN defaults: use domain-appropriate values (RSI=50, bb_pct_b=0.5)
- Differentiate XGB/LGBM hyperparameters for ensemble diversity
- Multi-day label horizon (5-day forward return) or 3-class labels
- Time-series cross-validation (5 folds minimum)
- Increase isotonic calibration minimum samples to 30
- Performance-weighted or stacked ensemble (not equal averaging)
- Log training failures in ensemble (not silent suppress)
- LSTM: mini-batch training (DataLoader with batch_size=32)
- Feature outlier handling: winsorize at 1st/99th percentile
- Remove dead `sentiment` feature (always 0.0) or integrate real sentiment
- Model versioning with timestamped files + `latest` symlink
- LLM sentiment output validation (clamp to [-1.0, 1.0])
- Simplify or fully utilize EventClassifier prompt JSON fields

### Architecture
- Rate limiter `time.sleep()` blocks scheduler threads
- Retry policy `time.sleep()` blocks threads (up to 45s worst case)
- Deprecated `get_event_loop()` in TelegramAlerter
- `_baseline_equities` dict not thread-safe
- `_health_cache` globals unsynchronized in system.py
- LLM client OrderedDict cache not concurrency-safe
- Auto-commit on read-only DB endpoints (wasteful)
- `reset_engine()` doesn't await `engine.dispose()` (leaks pools)
- Missing database indices (signals, orders, news_articles)
- Candle cache is dead code (defined but never called)
- Health probe creates throwaway Redis connections
- `/health/feeds` returns stubs
- No application-level rate limiting (nginx-only)
- TradingLoop errors not reported to API error buffer
- Redis cache failures logged at DEBUG (invisible in production)
- `lru_cache` on `get_settings()` prevents env var updates in tests
- Credentials not typed as `SecretStr`
- Settings resolved at import time in `main.py`
- mypy ignores all dashboard errors
- No DB/Redis services in CI
- Inconsistent Alembic URL replacement strategies
- YAML presets read from disk per signal (no caching in combiner)
- Redis password may be empty in production

### Risk
- Holiday calendar for US (NYSE) and MOEX
- Circuit breaker: add intraday high-water mark tracking
- `override_level()` has no audit trail
- PDT window uses calendar days, not business days
- Daily loss limit fallback inconsistency (0.05 vs 0.02)
- No flash crash / absurd move filter per stock
- No liquidity check before order submission
- Correlation limit (max 3 with r>0.7) not implemented

---

## Phase 8: Nice-to-Have (P3)

- Two Kelly implementations could be unified
- Break-even trades excluded from Kelly win rate
- Combined min_confidence should be per-segment in YAML
- Remove deprecated `config/modes.py` re-export
- `core/alerts.py` TYPE_CHECKING imports from L4/L5
- EventBus is dead infrastructure (built but unused)
- `create_group` suppresses all exceptions (should only suppress BUSYGROUP)
- No request correlation IDs / tracing
- Coverage threshold at 50% vs actual 86%
- Python-based Docker healthcheck is expensive
- `_collect_active_segments()` rebuilds set every call
- Finnhub API key in query string (Finnhub design, limited recourse)
- LSTM sequence_length=20 unvalidated
- Loader accesses private ensemble attributes
- Incremental/online learning for tree models
- No take-profit mechanism
- No intraday data support
- No order book simulation
- Gap risk documentation for market orders
- Alert flood throttling on repeated order rejections
- No cache invalidation beyond TTL

---

## Suggested Execution Order

```
Phase 5  (P0 critical)     →  ~2 weeks   →  8 tasks, blocks all live trading
Phase 6A (Risk/Compliance)  →  ~1 week    →  11 tasks, safety-critical
Phase 6B (Strategy Quality) →  ~2 weeks   →  8 tasks, correctness
Phase 6C (ML Pipeline)      →  ~1 week    →  9 tasks, model quality
Phase 6D (Arch/Security)    →  ~2 weeks   →  15 tasks, production readiness
Phase 7  (P2 improvements)  →  ~4 weeks   →  ~50 tasks, quality & robustness
Phase 8  (P3 nice-to-have)  →  ongoing    →  ~20 tasks, polish
```

Phases 6A–6D can be parallelized across 4 tracks since they touch different files.

---

## Key Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Test count | ~1009 | 1500+ |
| Coverage | 86% | 90%+ |
| P0 issues | 8 | 0 |
| P1 issues | ~30 | 0 |
| Strategies with look-ahead bias | 1 (pairs) | 0 |
| Thread-safety violations | 3 | 0 |
| Async correctness issues | 5 | 0 |
| Metrics wired to Prometheus | 0/18 | 18/18 |
