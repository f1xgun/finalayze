# Codebase Structure

**Analysis Date:** 2026-03-14

## Directory Layout

```
finalayze/
├── config/                     # Layer 1: Configuration (project root, not in src/)
│   ├── __init__.py
│   ├── settings.py             # Pydantic Settings + get_settings() factory
│   ├── modes.py                # WorkMode enum + ModeManager
│   ├── segments.py             # Segment registry (us_tech, us_broad, ru_blue_chips, etc.)
│   └── logging.py              # structlog setup + setup_logging()
│
├── src/finalayze/              # Layer 0–6 main codebase
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point (Layer 6)
│   │
│   ├── core/                   # Layer 0: Types & Schemas
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic models: Candle, Signal, PortfolioState, TradeResult, SignalDirection, PortfolioLayer
│   │   ├── exceptions.py       # Domain exceptions (DataFetchError, BrokerError, ModeError, etc.)
│   │   ├── modes.py            # WorkMode + ModeManager (runtime safety)
│   │   ├── events.py           # EventBus + event models (MarketDataEvent, SignalEvent) — Redis Streams
│   │   ├── clock.py            # ClockBase, RealClock, SimulatedClock
│   │   ├── models.py           # SQLAlchemy 2.0 async ORM models
│   │   ├── db.py               # Database engine/session factory
│   │   ├── trading_loop.py     # TradingLoop orchestrator (Layer 6, lives here for import convenience)
│   │   ├── alerts.py           # TelegramAlerter
│   │   ├── bond_cycle.py       # BondCycleProcessor (4-layer portfolio scheduler)
│   │   ├── bond_math.py        # Fixed-income math (duration, convexity, YTM)
│   │   └── layer_ledger.py     # Portfolio layer tracking
│   │
│   ├── data/                   # Layer 2: Data fetchers, normalization, caching
│   │   ├── __init__.py
│   │   ├── fetchers/           # Data source implementations
│   │   │   ├── base.py         # FetcherBase ABC
│   │   │   ├── yfinance.py     # YFinanceFetcher (US stocks)
│   │   │   ├── finnhub.py      # FinnhubFetcher (fundamental data)
│   │   │   ├── tinkoff_data.py # TinkoffFetcher (MOEX via gRPC, OHLCV + dividends)
│   │   │   ├── newsapi.py      # NewsApiFetcher (news articles)
│   │   │   ├── cbr.py          # CBRFetcher (Russian Central Bank rates)
│   │   │   ├── moex_iss.py     # MOEXISSFetcher (MOEX ISS REST API, deprecated in favor of Tinkoff)
│   │   │   ├── caching.py      # Caching wrapper for fetchers
│   │   │   └── _cache_utils.py # Cache helper utilities
│   │   ├── cache.py            # RedisCache (async Redis wrapper)
│   │   ├── normalizer.py       # DataNormalizer (OHLCV validation, UTC conversion)
│   │   ├── rate_limiter.py     # RateLimiter (token bucket)
│   │   ├── loader.py           # Data loading utilities
│   │   ├── macro_cache.py      # MacroCacheService (CBR rates, commodity indices)
│   │   └── moex_calendar.py    # MOEX holiday calendar
│   │
│   ├── markets/                # Layer 2: Instrument registry, schedules, FX
│   │   ├── __init__.py
│   │   ├── instruments.py      # Instrument, InstrumentRegistry (symbol → FIGI, market ID)
│   │   ├── registry.py         # MarketRegistry (market definitions)
│   │   ├── schedule.py         # MarketSchedule (US/MOEX trading hours)
│   │   ├── currency.py         # CurrencyConverter (cross-asset FX)
│   │   └── fx_service.py       # FXRateService (live FX rate updates)
│   │
│   ├── analysis/               # Layer 3: News & sentiment
│   │   ├── __init__.py
│   │   ├── llm_client.py       # LLMClient (Claude Sonnet wrapper)
│   │   ├── news_analyzer.py    # NewsAnalyzer (article → sentiment + entity extraction)
│   │   ├── event_classifier.py # EventClassifier (M&A, dividend, earnings, etc.)
│   │   ├── impact_estimator.py # ImpactEstimator (market impact from news)
│   │   ├── fact_checker.py     # FactChecker (claim validation)
│   │   └── prompts/            # LLM prompt templates
│   │
│   ├── ml/                     # Layer 3: Machine learning
│   │   ├── __init__.py
│   │   ├── registry.py         # MLModelRegistry (load/verify models per segment)
│   │   ├── loader.py           # Model loading utilities
│   │   ├── integrity.py        # HMAC signature verification for model artifacts
│   │   ├── staleness.py        # Staleness checks (retraining intervals)
│   │   ├── calibration.py      # Conformal calibration (prediction intervals)
│   │   ├── meta_labeler.py     # MetaLabelStep (ML-assisted label smoothing)
│   │   ├── features/           # Feature engineering
│   │   │   ├── __init__.py
│   │   │   ├── technical.py    # TechnicalFeatures (45 indicators via pandas-ta)
│   │   │   ├── multi_timeframe.py # MultiTimeframeFeatures
│   │   │   └── corporate_actions.py # CorporateActionsFeatures (dividends, splits)
│   │   ├── models/             # Model implementations
│   │   │   ├── base.py         # BaseModel ABC
│   │   │   ├── xgboost_model.py  # XGBoostModel wrapper
│   │   │   ├── lightgbm_model.py # LightGBMModel wrapper
│   │   │   ├── catboost_model.py # CatBoostModel wrapper
│   │   │   ├── lstm_model.py   # LSTMModel (PyTorch)
│   │   │   ├── ensemble.py     # EnsembleModel (voting/stacking meta-learner)
│   │   │   └── stacking.py     # Stacking utilities
│   │   └── training/           # Training pipeline
│   │       ├── __init__.py
│   │       ├── labeling.py     # Label generation (trend-scanning, daily returns)
│   │       ├── sample_weights.py # Sequential bootstrap weighting
│   │       ├── splitter.py     # Walk-forward + Combinatorial Purged Cross-Validation
│   │       ├── feature_selection.py # Feature selection (mutual info, permutation)
│   │       ├── quality_gates.py # Quality checks (Brier score, feature budget)
│   │       ├── trend_scanning.py # Trend-scanning labels (high/low breakout detection)
│   │       └── cpcv.py         # Combinatorial Purged CV implementation
│   │
│   ├── strategies/             # Layer 4: Trading signals
│   │   ├── __init__.py
│   │   ├── base.py             # BaseStrategy ABC
│   │   ├── combiner.py         # StrategyCombiner (YAML presets, ADX routing, confidence aggregation)
│   │   ├── adaptive_combiner.py # Adaptive weighting variant
│   │   ├── adx.py              # ADX(14) regime classifier
│   │   ├── momentum.py          # MomentumStrategy (RSI + MACD)
│   │   ├── mean_reversion.py    # MeanReversionStrategy (Bollinger Bands)
│   │   ├── dual_momentum.py     # DualMomentumStrategy (cross-asset momentum)
│   │   ├── rsi2_connors.py      # RSI2ConnorsStrategy (2-period RSI reversal)
│   │   ├── ou_mean_reversion.py # OUMeanReversionStrategy (Ornstein-Uhlenbeck)
│   │   ├── pairs.py            # PairsStrategy (cointegration)
│   │   ├── event_driven.py      # EventDrivenStrategy (news-triggered, disabled)
│   │   ├── ml_strategy.py       # MLStrategy (ML ensemble wrapper)
│   │   ├── dividend_gap.py      # DividendGapStrategy (ex-dividend arbitrage)
│   │   ├── bond_duration_rotation.py # BondDurationRotationStrategy
│   │   ├── bond_carry.py        # BondCarryStrategy (MOEX bonds carry)
│   │   ├── cbr_calendar.py      # CBRCalendarStrategy (macro calendar overlay)
│   │   ├── cbr_event.py         # CBREventListener (listens to CBR calendar)
│   │   ├── cbr_strategy_wrapper.py # CBRStrategyWrapper (applies CBR gating to strategies)
│   │   ├── ichimoku.py          # IchimokuStrategy (Ichimoku clouds)
│   │   ├── hurst.py            # HurstStrategy (Hurst exponent mean-reversion detector)
│   │   ├── vol_targeting.py    # VolTargetingStrategy
│   │   ├── hrp.py              # Hierarchical Allocation (HRP) utilities
│   │   └── presets/            # Per-segment YAML configuration files
│   │       ├── us_tech.yaml
│   │       ├── us_broad.yaml
│   │       ├── us_healthcare.yaml
│   │       ├── us_finance.yaml
│   │       ├── ru_blue_chips.yaml
│   │       └── ru_energy.yaml
│   │
│   ├── risk/                   # Layer 4: Risk management
│   │   ├── __init__.py
│   │   ├── position_sizer.py   # compute_position_size(), compute_realized_vol()
│   │   ├── stop_loss.py        # compute_atr_stop_loss()
│   │   ├── chandelier_exit.py  # Chandelier stop-loss (ATR + high/low tracking)
│   │   ├── position_sizing_pipeline.py # 6-step pipeline (Kelly → Vol → Regime → EVT → Copula → Caps)
│   │   ├── pre_trade_check.py  # PreTradeChecker (11-check validation)
│   │   ├── circuit_breaker.py  # CircuitBreaker (L1/L2/L3 daily DD limits) + CrossMarketCircuitBreaker
│   │   ├── layer_circuit_breaker.py # PortfolioLayerCircuitBreaker (per-layer drawdown)
│   │   ├── drawdown_monitor.py # DrawdownMonitor (tracks peak-to-trough)
│   │   ├── loss_limits.py      # LossLimitTracker (daily + cumulative loss caps)
│   │   ├── kelly.py            # RollingKelly (win rate + avg win ratio tracking)
│   │   ├── dv01_sizing.py      # DV01-aware position sizing for bonds
│   │   ├── regime.py           # RegimeProvider (HMM regime detection)
│   │   ├── hmm_regime.py       # HiddenMarkovModel regime classifier
│   │   ├── correlation.py      # Correlation calculation + caching
│   │   ├── copula.py           # Gaussian copula for tail dependency
│   │   ├── evt.py              # Extreme Value Theory (Generalized Pareto)
│   │   ├── garch.py            # GARCH volatility forecasting
│   │   ├── bocpd.py            # Bayesian Online Change Point Detection
│   │   └── commodity_currency.py # Commodity-currency correlations
│   │
│   ├── execution/              # Layer 5: Order execution
│   │   ├── __init__.py
│   │   ├── broker_base.py      # BrokerBase ABC + OrderRequest/OrderResult dataclasses
│   │   ├── alpaca_broker.py    # AlpacaBroker (Alpaca REST API)
│   │   ├── tinkoff_broker.py   # TinkoffBroker (Tinkoff gRPC API for MOEX)
│   │   ├── simulated_broker.py # SimulatedBroker (for backtests)
│   │   ├── broker_router.py    # BrokerRouter (dispatches to correct broker by market ID)
│   │   ├── retry_policy.py     # RetryPolicy (exponential backoff)
│   │   └── fill_reconciler.py  # Fill reconciliation (slippage, commission)
│   │
│   ├── backtest/               # Backtesting engine (all modules, Layer 5 conceptually)
│   │   ├── __init__.py
│   │   ├── engine.py           # BacktestEngine (main candle-replay loop)
│   │   ├── config.py           # BacktestConfig (strategy-specific parameters)
│   │   ├── performance.py      # PerformanceAnalyzer (Sharpe, DD, win rate, PF)
│   │   ├── walk_forward.py     # Walk-forward analysis (train/test splits)
│   │   ├── bond_engine.py      # Bond-specific backtest engine
│   │   ├── bond_walk_forward.py # Bond walk-forward analysis
│   │   ├── bond_metrics.py     # Bond-specific performance metrics
│   │   ├── journaling_combiner.py # JournalingStrategyCombiner (hook-based signal tracking)
│   │   ├── decision_journal.py # DecisionJournal (logs all trading decisions for analysis)
│   │   ├── costs.py            # TransactionCosts (commission, slippage)
│   │   ├── iteration_tracker.py # IterationTracker (versioning + comparison)
│   │   ├── monte_carlo.py      # Monte Carlo simulation
│   │   ├── stress_test.py      # Stress testing (market gap simulation)
│   │   ├── portfolio_aggregator.py # Multi-symbol portfolio aggregation
│   │   └── metrics.py          # Custom performance metrics
│   │
│   ├── api/                    # Layer 6: REST API
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py       # Main router combining all sub-routers
│   │       ├── auth.py         # API key authentication middleware
│   │       ├── system.py       # Health, mode management, metrics
│   │       ├── portfolio.py    # Portfolio state, positions, snapshots
│   │       ├── trades.py       # Trade history, P&L
│   │       ├── signals.py      # Current signals per symbol
│   │       ├── risk.py         # Exposure, circuit breaker, drawdown status
│   │       ├── news.py         # Recent articles, sentiment scores
│   │       └── ml.py           # Model metadata, feature importance, retraining
│   │
│   └── dashboard/              # Layer 6: Streamlit dashboard
│       ├── __init__.py
│       ├── app.py              # Main dashboard entry point
│       └── pages/
│           ├── portfolio.py    # Portfolio performance
│           ├── strategies.py   # Strategy signals, win rates
│           ├── risk.py         # Risk metrics, circuit breaker status
│           └── metrics.py      # Custom performance analysis
│
├── backtest/                   # Backtest-related utilities (not in src/)
│   └── ...
│
├── tests/                      # Test suite (unit, integration, e2e)
│   ├── unit/
│   │   ├── test_*.py           # Unit tests for each module
│   │   ├── test_config.py      # Config loading
│   │   ├── test_instruments.py # Instrument registry
│   │   ├── test_tinkoff_*.py   # Tinkoff integrations
│   │   ├── test_strategies.py  # Strategy signal generation
│   │   ├── test_position_*.py  # Position sizing
│   │   ├── test_engine.py      # Backtest engine
│   │   └── conftest.py         # Pytest fixtures + mocks
│   ├── integration/
│   │   └── test_*.py           # Integration tests (broker, data fetchers)
│   └── e2e/
│       └── test_*.py           # End-to-end backtest scenarios
│
├── scripts/                    # Standalone CLI scripts
│   ├── run_iteration.py        # Main backtest runner (phases)
│   ├── run_validation.py       # Validation backtest
│   ├── run_portfolio_iteration.py # Multi-symbol backtest
│   ├── run_batch_evaluation.py # Batch segment evaluation
│   ├── train_models.py         # ML model training
│   ├── tune_hyperparams.py     # Optuna hyperparameter tuning
│   ├── run_sandbox.py          # Sandbox trading simulation
│   ├── seed_historical_data.py # Download historical OHLCV
│   ├── build_event_data.py     # Build historical event dataset for news analysis
│   ├── compare_iterations.py   # Compare backtest metrics across iterations
│   ├── run_bond_iteration.py   # Bond portfolio backtest
│   └── ...
│
├── config/                     # Global configuration (Layer 1)
│   ├── settings.py
│   ├── modes.py
│   ├── segments.py
│   └── logging.py
│
├── models/                     # Trained ML model artifacts
│   └── <segment>/              # e.g., models/us_tech/
│       ├── xgb.pkl
│       ├── lgbm.pkl
│       ├── catboost.pkl
│       ├── calibrator.pkl
│       ├── meta_learner.pkl
│       ├── model_weights.json
│       ├── segment_meta.json
│       └── selected_features.json
│
├── results/                    # Backtest iteration output
│   └── iterations/
│       ├── baseline/
│       ├── sprint1-moex-quickwins/
│       ├── week5-final/
│       └── ...
│
├── docs/                       # Documentation
│   ├── architecture/           # (OVERVIEW.md, DEPENDENCY_LAYERS.md, DATA_FLOW.md, DECISIONS.md)
│   ├── design/                 # (MARKETS.md, SEGMENTS.md, STRATEGIES.md, RISK.md, NEWS.md, ML.md)
│   ├── api/                    # (ENDPOINTS.md, SCHEMA.md)
│   ├── quality/                # (GRADES.md, GAPS.md)
│   ├── operations/             # (DEPLOYMENT.md, MONITORING.md)
│   ├── plans/                  # (ROADMAP.md, PHASE_1.md, PHASE_2.md, etc.)
│   ├── research/               # (deep-research documents)
│   └── INDEX.md                # Master documentation index
│
├── alembic/                    # Database migrations (SQLAlchemy)
│   ├── env.py
│   ├── versions/
│   │   └── 001_initial_schema.py
│   └── alembic.ini
│
├── .claude/                    # Claude Code agents & skills
│   ├── agents/
│   │   ├── gsd-*.md            # GSD sub-agent definitions
│   │   └── ...
│   ├── skills/
│   │   ├── backtest-iteration/
│   │   ├── strategy-diagnose/
│   │   └── ... (6 total custom skills)
│   └── settings.json
│
├── .planning/                  # GSD planning state
│   ├── STATE.md                # Phase execution state
│   ├── ROADMAP.md              # Phase overview + status
│   ├── REQUIREMENTS.md         # Phase requirements
│   └── codebase/               # Codebase analysis (THIS FILE's siblings)
│       ├── ARCHITECTURE.md
│       ├── STRUCTURE.md
│       ├── CONVENTIONS.md
│       ├── TESTING.md
│       ├── CONCERNS.md
│       ├── STACK.md
│       └── INTEGRATIONS.md
│
├── .github/workflows/          # GitHub Actions CI
│   ├── lint.yml                # ruff check + format
│   ├── typecheck.yml           # mypy strict
│   └── test.yml                # pytest
│
├── docker/                     # Docker builds
│   ├── Dockerfile              # Production image
│   └── docker-compose.yml      # Dev/test services
│
├── pyproject.toml              # Python package manifest (uv)
├── uv.lock                     # Lock file (committed)
├── CLAUDE.md                   # Agent entry point + conventions
├── WORKFLOW.md                 # Development process doc
├── README.md                   # Project overview
└── .env.example                # Template for environment variables
```

## Directory Purposes

**`src/finalayze/`:**
- Purpose: Main Python package with 6-layer architecture
- Contains: All source code organized by layer (core, data, strategies, execution, etc.)
- Key files: Layer entry points, strategy implementations, ML pipeline, risk management

**`config/`:**
- Purpose: Global configuration (at project root, not under src/)
- Contains: Pydantic Settings, WorkMode enum, segment registry, logging setup
- Key files: `settings.py` (environment variables), `modes.py` (runtime safety), `segments.py` (market definitions)

**`tests/`:**
- Purpose: Test suite (unit, integration, E2E)
- Contains: pytest tests with fixtures, mocks, live broker sandbox tests
- Key files: Organized by layer, `conftest.py` provides shared fixtures

**`scripts/`:**
- Purpose: Standalone CLI scripts for backtesting, training, validation
- Key files: `run_iteration.py` (main backtest runner), `train_models.py` (ML), `run_sandbox.py` (paper trading)

**`models/`:**
- Purpose: Trained ML model artifacts (persisted on disk)
- Contains: Per-segment XGBoost, LightGBM, CatBoost, LSTM models + meta-learner + calibrator
- Key files: `.pkl` model files, `.json` metadata (weights, selected features)

**`results/`:**
- Purpose: Backtest iteration output + performance metrics
- Contains: CSV results, JSON metrics, portfolio snapshots per iteration
- Key files: Organized by iteration name (e.g., `results/iterations/week5-final/`)

**`docs/`:**
- Purpose: Architecture, design, API documentation
- Key files: `OVERVIEW.md` (system design), `ENDPOINTS.md` (REST API), `ROADMAP.md` (phases)

**`alembic/`:**
- Purpose: Database schema migrations (SQLAlchemy)
- Contains: Alembic environment + version files
- Key files: `versions/001_initial_schema.py` (creates tables + TimescaleDB hypertable)

**`.claude/`:**
- Purpose: Claude Code agent definitions and custom trading skills
- Contains: 18 sub-agent YAML definitions, 6 custom trading skills
- Key files: `agents/gsd-*.md` (GSD sub-agents), `skills/backtest-iteration/` (backtest skill)

**`.planning/`:**
- Purpose: GSD orchestrator state and codebase analysis
- Contains: Phase execution state, requirements, this codebase structure document
- Key files: `STATE.md` (current phase), `ROADMAP.md` (phase overview), `codebase/` (architecture docs)

## Key File Locations

**Entry Points:**
- `src/finalayze/main.py`: FastAPI application factory (`create_app()`, `app` instance)
- `src/finalayze/core/trading_loop.py`: Live trading loop (`TradingLoop` class, APScheduler-based)
- `scripts/run_iteration.py`: Backtest CLI runner
- `src/finalayze/dashboard/app.py`: Streamlit dashboard entry point

**Configuration:**
- `config/settings.py`: Pydantic Settings (env var loading via `FINALAYZE_` prefix)
- `config/modes.py`: WorkMode enum + ModeManager (safety gates)
- `config/segments.py`: Segment definitions (markets, instruments, strategy weights)
- `config/logging.py`: structlog setup (JSON output, per-mode levels)

**Core Logic:**
- `src/finalayze/strategies/combiner.py`: Signal aggregation (YAML presets, ADX routing)
- `src/finalayze/backtest/engine.py`: Main backtest loop (candle replay with signals + risk)
- `src/finalayze/risk/position_sizing_pipeline.py`: 6-step position sizing
- `src/finalayze/execution/broker_router.py`: Order routing to Alpaca/Tinkoff/Simulated

**Testing:**
- `tests/conftest.py`: Pytest fixtures (mock brokers, fake candles, settings)
- `tests/unit/test_engine.py`: Backtest engine tests
- `tests/unit/test_tinkoff_*.py`: Tinkoff integration tests

## Naming Conventions

**Files:**
- Strategy implementations: `src/finalayze/strategies/{name}_strategy.py` or `src/finalayze/strategies/{name}.py` (e.g., `momentum.py`, `mean_reversion.py`)
- Risk management: `src/finalayze/risk/{component}.py` (e.g., `position_sizer.py`, `circuit_breaker.py`)
- Data fetchers: `src/finalayze/data/fetchers/{source}.py` (e.g., `yfinance.py`, `tinkoff_data.py`)
- ML models: `src/finalayze/ml/models/{model_type}_model.py` (e.g., `xgboost_model.py`)
- Tests: `tests/unit/test_{module}.py` (mirrors source structure)

**Directories:**
- Layer-based: `src/finalayze/{layer_name}/` (core, data, strategies, risk, execution, backtest, api, dashboard)
- Sub-packages: `src/finalayze/{layer}/{component}/` (e.g., `data/fetchers/`, `ml/features/`, `ml/training/`, `api/v1/`)
- Segment-specific: `models/{segment_id}/` (e.g., `models/us_tech/`)

## Where to Add New Code

**New Strategy:**
- Implementation: `src/finalayze/strategies/{name}.py` (inherit from `BaseStrategy`, implement `generate_signal()`)
- Configuration: `src/finalayze/strategies/presets/{segment}.yaml` (add strategy name + weight)
- Tests: `tests/unit/test_{name}_strategy.py`
- Register in: `src/finalayze/core/trading_loop.py` (TradingLoop._strategies list)

**New Risk Component:**
- Implementation: `src/finalayze/risk/{component}.py`
- Pipeline integration: `src/finalayze/risk/position_sizing_pipeline.py` (add step if part of pipeline)
- Tests: `tests/unit/test_risk_{component}.py`

**New Data Fetcher:**
- Implementation: `src/finalayze/data/fetchers/{source}.py` (inherit from `FetcherBase`, implement `fetch()`)
- Rate limiting: Add `RateLimiter` if API has quota
- Tests: `tests/integration/test_{source}_fetcher.py`
- Register in: `src/finalayze/data/loader.py` (if used in backtest pipeline)

**New ML Feature:**
- Implementation: `src/finalayze/ml/features/technical.py` or new file in `src/finalayze/ml/features/`
- Feature selection: Add to `src/finalayze/ml/training/feature_selection.py`
- Tests: `tests/unit/test_ml_features.py`

**New API Endpoint:**
- Implementation: `src/finalayze/api/v1/{domain}.py` (e.g., `portfolio.py`, `risk.py`)
- Router inclusion: `src/finalayze/api/v1/router.py` (include_router())
- Tests: `tests/unit/test_api_{domain}.py`

**Utilities:**
- Shared helpers: `src/finalayze/{layer}/utils.py` or domain-specific module
- Cross-layer utilities: Avoid! If needed, place at lowest layer that uses it
- Example: `src/finalayze/core/clock.py` (Layer 0, used by all layers)

## Special Directories

**`models/`:**
- Purpose: Trained ML model artifacts (XGBoost, LightGBM, etc.)
- Generated: Yes (by `scripts/train_models.py`)
- Committed: No (added to `.gitignore`, but integrity checked via HMAC in `ml/integrity.py`)

**`results/`:**
- Purpose: Backtest iteration output (CSV metrics, JSON results)
- Generated: Yes (by `scripts/run_iteration.py`)
- Committed: No (large files, added to `.gitignore`)

**`.cache/`:**
- Purpose: Data caching (historical candles, news articles)
- Generated: Yes (by data fetchers)
- Committed: No (git-ignored)

**`alembic/versions/`:**
- Purpose: Database migration files (one per schema change)
- Generated: Yes (by `alembic revision --autogenerate`)
- Committed: Yes (track schema history)

**`.planning/`:**
- Purpose: GSD orchestrator state (phase execution, requirements, analysis docs)
- Generated: Yes (by GSD commands and mapping agents)
- Committed: Yes (state must persist across sessions)

---

*Structure analysis: 2026-03-14*
