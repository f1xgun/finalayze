# Finalayze — Regression & Audit Report

**Date:** 2026-06-28  
**Scope:** `origin/main` @ `3008e3a` (Phase 88 — latest shipped)  
**Method:** CI-equivalent regression battery + 7-dimension multi-agent audit (64 agents, every finding adversarially verified against real code; 5 findings rejected as not-real).  
**Companion to:** the onevoice regression/audit methodology (Makefile gate battery + golden-principles review).

---

## 1. Executive summary

- **Regression: GREEN.** All four CI gates pass (graph integrity, ruff lint, ruff format, mypy 284 files). Full suite **5996 passed / 92.82% coverage**; the only genuine test failure is the **documented pre-existing** `test_api_trades` 404-vs-500 todo. Operational validators + live sandbox e2e smoke (10/10) pass.
- **Audit: no CRITICAL, 3 HIGH, 16 MEDIUM, 26 LOW, 7 INFO.** The real-money LIVE hard stop is **robustly enforced** (independent triple gate + safe-by-default everywhere). No committed secrets; MOEX-via-Tinkoff invariant holds in the *runtime* path; no look-ahead bias in strategies/backtest.
- **Most important finding corroborates a live observation:** the deployed sandbox's `FATAL: too many clients` (which I hit during the Docker rollout) is a **real, root-caused async-engine connection leak** (HIGH).

### Severity tally (confirmed)

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 3 |
| MEDIUM | 16 |
| LOW | 26 |
| INFO | 7 |
| **Total** | **52** |

### By dimension (confirmed)

| Dimension | Confirmed |
|---|---|
| Architecture & layering | 11 |
| Tests & open debt | 10 |
| Risk & live-trade safety | 8 |
| Code quality | 8 |
| Security | 6 |
| Financial correctness | 5 |
| ML & data integrity | 4 |

---

## 2. Regression results

### 2.1 CI gates (run against `origin/main` in an isolated worktree)

| Gate | Command | Result |
|---|---|---|
| Graph integrity | `python scripts/graph_check.py` | ✅ OK (19 nodes, 19 files) |
| Lint | `ruff check .` | ✅ All checks passed |
| Format | `ruff format --check .` | ✅ 862 files formatted |
| Typecheck | `mypy src/finalayze/` | ✅ no issues, 284 files |
| Tests | `pytest --cov` | ⚠️ 5996 passed, 1 known failure, 92.82% cov |

### 2.2 Test-suite triage

Raw run surfaced 8 "failures"; re-running each in a clean CI-equivalent env (no `.env`, no DB) showed **7 were environment artifacts, 1 is a real known issue**:

| Test | Clean verdict | Cause |
|---|---|---|
| `test_api_trades::test_trade_detail_returns_500_without_db` | ❌ **real fail** | `assert 404 == 500` — documented pre-existing 404-vs-500 todo |
| 3× `test_saa_persistence_db` / `test_alembic_upgrade` / `test_stop_history` | ⏭️ skip | need a live DB (`FINALAYZE_DATABASE_URL`) |
| `test_settings_phase3::test_news_cycle_minutes_default` | ✅ pass clean | `.env` overrode the default under test |
| `test_meta_agent_status_endpoint::…defaults` | ✅ pass clean | same `.env` contamination |
| `test_executor::…fire_and_forget_envelope` | ✅ pass clean | `.env` contamination (RuntimeWarning persists — see F-arch leak) |

### 2.3 Operational validators & live e2e

| Check | Result |
|---|---|
| `validate_capital_ladder.py` | ✅ exit 0 — all instruments viable at the 2.5M tier (17 non-viable only at 50–150k tiers, expected) |
| `validate_ofz_data.py` | ✅ PASS — 12/12 OFZ bonds w/ candle+coupon+NKD via T-Bank gRPC |
| `smoke_test_sandbox.py` | ✅ 10/10 — full live sandbox execution path (buys/coupons/sells/drawdown/CBR) |
| Docker rollout | ✅ new version live, `/health` ok, SAA endpoints live, dashboard 8501 ok |

Smoke observations: `instrument_overwrite_distinct_figi` (two FIGIs collide on one MOEX symbol — see known WR-01) and `PostOrder RESOURCE_EXHAUSTED` ×5 (Tinkoff sandbox rate-limit; retries succeeded).

---

## 3. Audit findings

Every finding below was produced by a specialist finder and then **independently verified against the real code** by a skeptic agent (default-reject on non-reproduction). Severities are the *verified* (corrected) severities.

### 3.1 HIGH

#### [HIGH] Connection-pool leak: async engines are never disposed; per-event-loop engines accumulate on loop recreation

- **Dimension:** Architecture & layering
- **Location:** `src/finalayze/orchestration/db_persistence.py` 85-124 (creation); 76-83 of src/finalayze/core/db.py (reset_engine); no dispose anywhere)
- **Evidence:** _get_bg_session_factory() creates a NEW create_async_engine(pool_size=5, max_overflow=2) keyed by id(asyncio.get_running_loop()) and stores it in self._bg_session_factories (db_persistence.py:100-124). TradingPersistence has NO close()/dispose() method (verified via outline). The ONLY production engine.dispose() call in the entire src tree is data/loader.py:361 (verified by ast-index: all other .dispose() hits are tests). core/db.py reset_engine() (db.py:76-83) only does _engine_cache.clear()/_factory_cache.clear() -- it drops the Python references but never `await engine.dispose()`, so the underlying asyncpg pool connections are orphaned, not closed. TradingLoop.stop() (trading_loop.py:895-912) closes RedisCache and FXRateService and calls AsyncRuntime.shutdown() (async_runtime.py:154-171), which stops and nulls the loops but disposes NO engine. Because the engine is keyed by id(loop), any loop recreation (broker reconnect, stop/start, or meta-agent running on the uvicorn loop vs the APScheduler background loop -- the exact scenario the code comments at db_persistence.py:78-83) produces a fresh engine+pool while the prior engine's 5-7 server-side connections linger. pool_recycle=1800 only recycles within a LIVE pool, not an orphaned one. Over weeks this is exactly the deployed 'FATAL: sorry, too many clients already' exhaustion. With the FastAPI engine (db.py: pool_size=10+overflow=5=15) plus each background engine (7), the count climbs unbounded.
- **Recommendation:** Add a TradingPersistence.dispose_all() that `await engine.dispose()`s every engine in _bg_session_factories and clear the dict; call it from TradingLoop.stop()/AsyncRuntime.shutdown() before nulling the loops. Make core/db.py reset_engine() async (or add async dispose_engines()) that awaits engine.dispose() for every cached engine before clearing. Prefer NOT keying engines by loop id at all: SQLAlchemy async engines are loop-affine, so instead create one engine per long-lived loop and dispose it in that loop's teardown, or register an atexit/lifespan hook. Add a Prometheus gauge for len(_bg_session_factories) and for pool.checkedout() to make the leak observable.
- **Note:** finder claimed CRITICAL; verification corrected to HIGH (high confidence).

#### [HIGH] yfinance fallback for MOEX (ru_*) tickers violates Tinkoff-gRPC-only invariant

- **Dimension:** ML & data integrity
- **Location:** `scripts/training/data_loader.py` 326-339
- **Evidence:** fetch_symbol_candles tries Tinkoff first for MOEX segments (`if segment_id and is_moex_segment(segment_id): tinkoff_candles = fetch_tinkoff_candles(symbol)`), but fetch_tinkoff_candles silently returns [] on missing token, FIGI-resolution failure, or any gRPC exception (data_loader.py:286-305). When that happens the code falls through to `fetcher = YFinanceFetcher(market_id=market_id); return fetcher.fetch_candles(symbol, ...)` (lines 334-336) with the MOEX ticker (e.g. SBER, GAZP) and market_id='ru'. This directly violates the project hard invariant 'MOEX data = Tinkoff Invest gRPC only. Never yfinance for MOEX tickers.' (CLAUDE.md). yfinance cannot resolve plain MOEX tickers (per project memory), so the best case is empty data and the worst case is silently fetching a wrong/colliding US ticker of the same name into a MOEX training set. The fallback path is unconditional once Tinkoff yields nothing.
- **Recommendation:** Make the MOEX branch fail-closed: if is_moex_segment(segment_id) and the DB+Tinkoff paths yield no candles, return [] (and log loudly) — never call YFinanceFetcher for ru_* symbols. Restructure so the yfinance fallback is reachable only for US segments. Add a regression test asserting YFinanceFetcher.fetch_candles is never invoked for a ru_ segment symbol.

#### [HIGH] ML model files (calibrator.pkl / meta_learner.pkl) are joblib-deserialized with NO integrity verification, ever

- **Dimension:** Security
- **Location:** `src/finalayze/ml/loader.py` 127, 199 (and ensemble.py:285)
- **Evidence:** An HMAC integrity scheme exists (ml/integrity.py verify_model) and IS enforced inside XGBoostModel/LightGBMModel/CatBoostModel.load_from when a key is set. But the calibrator is loaded with a bare `loaded_cal = joblib.load(calibrator_path)` (loader.py:127) and the meta-learner via `self._meta_learner = joblib.load(path)` (ensemble.py:285, called from loader.py:199) -- neither ever calls verify_model, even when FINALAYZE_ML_MODEL_HMAC_KEY is configured. joblib/pickle deserialization executes arbitrary code embedded in the file, so any actor who can write into the models/<segment>/ directory (or substitute a malicious calibrator.pkl/meta_learner.pkl) achieves RCE in the trading process at boot. This is an inconsistency in an otherwise-present integrity control: the boosting models are protected, these two are not.
- **Recommendation:** Route calibrator.pkl and meta_learner.pkl loads through the same key-gated verify_model() check used by the boosting models (extract a single `_verified_joblib_load(path)` helper in ml/loader.py and use it for all four .pkl loads). Longer term, prefer a non-pickle serialization (e.g. skops or explicit JSON of the calibrator/LogReg coefficients) so deserialization is not arbitrary-code-capable at all.

### 3.2 MEDIUM

#### [MEDIUM] Dependency-layer invariant is not mechanically enforced (no import-linter, no test_architecture.py) despite docs claiming it is

- **Dimension:** Architecture & layering
- **Location:** `docs/architecture/DEPENDENCY_LAYERS.md` 86-93
- **Evidence:** DEPENDENCY_LAYERS.md:88-93 states 'A custom ruff rule or import-linter configuration WILL enforce these boundaries in CI' and 'An architectural test in tests/test_architecture.py CAN programmatically verify no upward imports exist.' Reality: ast-index finds zero references to 'import-linter'/'importlinter' anywhere, and tests/test_architecture.py does not exist. The only structural gate, scripts/graph_check.py, validates AGENTS.md<->.agents/manifest.jsonl coverage/links -- it does NOT parse imports. CLAUDE.md invariant #1 and AGENTS.md invariant #1 ('Imports flow downward only') are therefore aspirational. This is the meta-cause: a static AST scan of src/finalayze finds 4 module-level + 18 function-local + 23 TYPE_CHECKING upward references that no gate catches.
- **Recommendation:** Add import-linter (contracts: one layered contract for the 0->6 stack) to pyproject and the CI lint job, OR commit the tests/test_architecture.py the doc already promises (an AST walk asserting no module imports a higher-numbered layer). Fail CI on any new upward import. Treat TYPE_CHECKING and function-local imports as still-violating for L0 modules (they indicate misplacement) even if they avoid the runtime cycle.
- **Note:** finder claimed HIGH; verification corrected to MEDIUM (high confidence).

#### [MEDIUM] Cross-cutting infrastructure (TelegramAlerter, Prometheus metrics) misplaced at L6 (api/) but consumed by L0 and pervasively by L5

- **Dimension:** Architecture & layering
- **Location:** `src/finalayze/api/alerts.py` 195 (TelegramAlerter); api/metrics.py (counters); consumers across L0/L5
- **Evidence:** TelegramAlerter is defined in api/alerts.py:195 (L6) and the Prometheus counters (db_write_failures, etc.) in api/metrics.py (L6). These are pure infrastructure with no API dependency, yet they are imported UPWARD from L0 (core/kill_switch.py:79 `from finalayze.api.alerts import AlertPriority`; core/layer_ledger.py:21 TYPE_CHECKING TelegramAlerter) and from ~15 sites across the most critical L5 runtime module via function-local imports purely to dodge the static cycle: orchestration/trading_loop.py:968,1082,1189,1260,1711 (api.alerts/api.metrics), signal_executor.py:36-37, db_persistence.py:156,170, news_pipeline.py:163,193, daily_reporting.py:392, preset_applicator.py:339, equity_reconcile/anomaly_handler/broker_reconnect/ml_retraining/position_manager (TYPE_CHECKING). The hot path (the trading loop) cannot import its own alerting/metrics at module scope -- a clear sign the dependency points the wrong way.
- **Recommendation:** Move TelegramAlerter/AlertPriority/AlertQueue to a low layer (e.g. core/alerts_impl.py at L0, or a new L1 'infra' package) and move the Prometheus counter definitions to L0/L1 (a core/metrics.py). Then api/ can re-export for HTTP wiring. This eliminates the L0->L6 and L5->L6 upward imports and lets the trading loop import alerts/metrics at module scope, removing ~33 deferred-import workarounds.
- **Note:** finder claimed HIGH; verification corrected to MEDIUM (high confidence).

#### [MEDIUM] KillSwitch and LayerLedger live in L0 (core/) but depend on L4/L5/L6

- **Dimension:** Architecture & layering
- **Location:** `src/finalayze/core/kill_switch.py` 27-30, 79-80 (and core/layer_ledger.py:21)
- **Evidence:** core/kill_switch.py is explicitly self-described as 'KillSwitch orchestrator ... (Layer 0/6 boundary)' (docstring line 1) and depends on api.alerts (L6), execution.broker_router (L5), orchestration.trading_loop (L5) and risk.circuit_breaker (L4) -- TYPE_CHECKING at lines 27-30 plus real function-local imports at lines 79-80. core/layer_ledger.py:21 TYPE_CHECKING-imports api.alerts.TelegramAlerter (L6) and markets.instruments (L2) at line 24. A genuine L0 module must have zero project dependencies (DEPENDENCY_LAYERS.md:104 'Core importing anything'). KillSwitch is an orchestrator (it stops the loop, escalates breakers, cancels orders) -- it belongs in L5/orchestration, not L0.
- **Recommendation:** Move KillSwitch to orchestration/ (L5) where it can legitimately import execution/risk/orchestration, and inject the alerter. Move LayerLedger's TelegramAlerter usage out (inject an L0-located alerter interface) so layer_ledger stays a pure L0 data structure, or relocate it if it truly needs market/registry access.

#### [MEDIUM] L3 analysis imports FIFO P&L accounting from L6 api (api.v1._fifo)

- **Dimension:** Architecture & layering
- **Location:** `src/finalayze/analysis/portfolio_review_agent.py` 295
- **Evidence:** portfolio_review_agent.py:295 does `from finalayze.api.v1._fifo import fifo_pair` (function-local) -- L3 reaching UP into L6. fifo_pair is realized-P&L FIFO lot-matching, a core money-accounting primitive that has been placed inside the API router package. Besides being an upward import, this means FIFO accounting logic is owned by the presentation layer; any non-API consumer (analysis here, and potentially backtest/risk) must reach into api/ to do correct P&L.
- **Recommendation:** Move fifo_pair (and any sibling FIFO/lot-matching helpers) to a low layer -- core/ (L0) if pure, or data/ (L2) if it touches models -- and have both api.v1 and analysis import it downward. Money math like FIFO realized P&L must not live in L6.

#### [MEDIUM] Cross-symbol fundamental-feature contamination: features attributed to wrong ticker (train + inference)

- **Dimension:** ML & data integrity
- **Location:** `src/finalayze/ml/features/fundamental.py` 176
- **Evidence:** compute_fundamental_features derives the target symbol from `target = max(in_window, key=lambda s: s.as_of)` (line 176) and `symbol = target.symbol` (line 177) — i.e. whichever snapshot across the ENTIRE segment has the latest as_of. But the function is given a segment-wide MoexMarketData: loader.py:150-154 fetches `fundamentals` for the full `symbols` list, and MLStrategy.generate_signal (ml_strategy.py:103-107) plus _build_dataset_direction (scripts/training/dataset_builder.py:199-203) pass that SAME market_context for every per-symbol feature computation. compute_features (technical.py:190) calls compute_fundamental_features(_moex, as_of=...) with NO symbol argument, even though candles[-1].symbol is available. Net effect: when scoring LKOH, its feature row's earnings_yield / pe_zscore_vs_sector / revenue_growth_yoy / net_margin_trend / dividend_yield_z are computed for whichever segment peer (e.g. ROSN) has the most recent snapshot — not LKOH. This is consistent across train and inference, so it is not future-leakage, but the 5 fundamental features are semantically attached to the wrong instrument for all-but-one symbol per segment, defeating the per-symbol cross-sectional z-score intent and injecting noise. Feature-schema v4 is entirely these fundamental features (loader.py:27).
- **Recommendation:** Thread the scored symbol into compute_features (e.g. derive from candles[-1].symbol) and pass it to compute_fundamental_features so the target snapshot is selected by symbol identity (filter in_window to s.symbol == scored_symbol, then max by as_of), not by global latest as_of. Add a unit test that scores two symbols against the same segment-wide MoexMarketData and asserts each gets its own fundamentals. Bump FEATURE_SCHEMA_VERSION after the fix so stale models are rejected.
- **Note:** finder claimed HIGH; verification corrected to MEDIUM (high confidence).

#### [MEDIUM] Production normalize logic re-implemented in test and already drifted (silent loss of attribution fields)

- **Dimension:** Code quality
- **Location:** `tests/unit/test_moex_fixes.py` 120-140
- **Evidence:** `_normalize_trades_to_usd` is copied verbatim into the test as a local "replica" (docstring: "Replicate the normalize function from run_iteration.py") instead of importing the real one. The production version (scripts/run_iteration.py:788-798) was deliberately fixed under WR-01 to use `t.model_copy(update={...})` precisely so future TradeResult fields carry through and cross-segment aggregation cannot drop attribution data. The test's copy instead reconstructs TradeResult field-by-field (signal_id, symbol, side, quantity, entry_price, exit_price, pnl, pnl_pct, hold_bars) and OMITS `coupon_income` (and any other current/future field). So the test passes its own degraded copy while the real code carries the field — the test no longer exercises the shipped behavior and would stay green even if production regressed. `_normalize_snapshots_to_usd` (tests/unit/test_moex_fixes.py:143) and `_compute_segment_cash` (line 22) are copied the same way and can drift identically.
- **Recommendation:** Extract the pure helpers (`_normalize_trades_to_usd`, `_normalize_snapshots_to_usd`, segment-cash conversion) out of the script into a `src/finalayze/` module (Layer 2/4, e.g. markets/segment_normalize.py) and import them in BOTH run_iteration.py and the test. The script currently can't be imported cleanly because it calls `load_dotenv()` and mutates sys.path at module load (scripts/run_iteration.py:30,38-40) and pulls the full backtest stack — extracting the pure functions removes that obstacle and kills the drift.
- **Note:** finder claimed HIGH; verification corrected to MEDIUM (high confidence).

#### [MEDIUM] GET /positions/{symbol} is a permanent stub that returns 404 even when the position exists

- **Dimension:** Code quality
- **Location:** `src/finalayze/api/v1/portfolio.py` 523-530
- **Evidence:** The endpoint is registered (`@router.get("/positions/{symbol}", response_model=PositionDetail)`) and documented as "Return detail for a single open position. Returns 404 if not found." but the body is: if broker_router is None -> 404; then `# TODO: wire to real broker_router` followed by an UNCONDITIONAL `raise HTTPException(status_code=404, ...)`. So even when broker_router is wired and the position is genuinely held, the caller always gets 404. This is dead/stub logic exposed on a live API surface — an operator querying a real position is told it doesn't exist.
- **Recommendation:** Either implement the lookup against broker_router (mirror the position-detail logic used by api/v1/risk.py:140-149 `get_positions_detail`) or remove the route until it is implemented, so the API contract doesn't advertise a feature that is hard-wired to 404.

#### [MEDIUM] Pre-trade CircuitBreakerCheck is permanently bypassed -- _get_circuit_breaker_level always returns None

- **Dimension:** Risk & live-trade safety
- **Location:** `src/finalayze/orchestration/signal_executor.py` 949-957
- **Evidence:** The method _get_circuit_breaker_level(self, market_id) contains a comment 'For now, return None (check will skip)' and always returns None. This value is passed to CheckContext.circuit_breaker_level at line 1014. The CircuitBreakerCheck (pre_trade_check.py:189-195) returns None when circuit_breaker_level is None, so the check always passes. While the outer TradingLoop._process_market_cycle() gates HALTED/LIQUIDATE before calling process_instrument (line 1462-1465), this means the pre-trade check's CircuitBreakerCheck class is dead code in the live path, removing a documented defense-in-depth layer. If TradingLoop's outer gate is ever refactored or bypassed, trades would flow through unchecked.
- **Recommendation:** Pass the CircuitLevel that TradingLoop already computes as a parameter through SignalExecutor.process_instrument (it is already the 'level' param) into _run_pre_trade_check. Replace the stub _get_circuit_breaker_level method. This costs one line change and restores the defense-in-depth contract documented in pre_trade_check.py.
- **Note:** finder claimed HIGH; verification corrected to MEDIUM (high confidence).

#### [MEDIUM] StopLossRequiredCheck is semantically inverted -- never blocks orders missing a stop-loss

- **Dimension:** Risk & live-trade safety
- **Location:** `src/finalayze/orchestration/signal_executor.py` 1015-1016
- **Evidence:** Line 1016 sets require_stop_loss=self._position_tracker.has_stop(symbol). For a NEW BUY order on a symbol without an existing position, has_stop() returns False, so require_stop_loss=False, and StopLossRequiredCheck (pre_trade_check.py:293-295) passes unconditionally. The check was designed to enforce 'every order must have a stop-loss set' but the logic only checks 'if an existing stop exists, verify its price is not None' -- which is tautologically true (if has_stop is True, get_stop_loss_price is non-None). The actual stop-loss is set AFTER the order fills (_submit_order, lines 878-913), so it cannot be validated pre-trade. The check provides no protection.
- **Recommendation:** Either (a) remove StopLossRequiredCheck from the pre-trade pipeline since it is structurally impossible to validate a stop-loss before order submission, and document the post-fill stop setup as the true safety mechanism, or (b) redesign the check to validate that the sizing pipeline computed a valid stop-loss price before order submission by computing the stop earlier in the pipeline and passing it through CheckContext.
- **Note:** finder claimed HIGH; verification corrected to MEDIUM (high confidence).

#### [MEDIUM] Live correlation check (check 14) always passes -- _get_correlations returns empty dict

- **Dimension:** Risk & live-trade safety
- **Location:** `src/finalayze/orchestration/signal_executor.py` 613-623
- **Evidence:** _get_correlations() contains a TODO comment 'Wire returns history for live correlation computation in future phase' and always returns {}. CorrelationLimitCheck (pre_trade_check.py:367-383) calls count_correlated_positions with an empty correlation dict, which always returns 0. The documented portfolio constraint 'Max correlated (r>0.7): 3 positions' is NOT enforced in live trading. This means correlated positions can accumulate without limit, increasing portfolio tail risk.
- **Recommendation:** Implement correlation computation from cached historical returns in _get_correlations. Until implemented, add a warning log on each pre-trade cycle noting that correlation limits are not enforced, so the operator is aware of the gap.

#### [MEDIUM] Live parameter freshness check (check 13) always passes -- param_age_bars never populated

- **Dimension:** Risk & live-trade safety
- **Location:** `src/finalayze/orchestration/signal_executor.py` 1007-1028
- **Evidence:** CheckContext is constructed at lines 1007-1028 without setting param_age_bars (it defaults to None). ParamFreshnessCheck (pre_trade_check.py:348-360) requires param_age_bars is not None to fire. This means OU mean-reversion and pairs strategies can trade with arbitrarily stale parameters in live mode, which are exactly the strategies most sensitive to parameter drift (mean-reversion on a drifted mean can lead to large losses).
- **Recommendation:** Track parameter age for OU/pairs strategies and pass it through CheckContext.param_age_bars in _run_pre_trade_check. At minimum, surface a WARNING log that the freshness check is disabled.

#### [MEDIUM] KillSwitch uses /tmp path for persistent flag file -- not durable across reboots on some systems

- **Dimension:** Risk & live-trade safety
- **Location:** `src/finalayze/core/kill_switch.py` 65
- **Evidence:** Default flag_path is Path('/tmp/finalayze_killed'). On Linux systems with tmpfs mounted at /tmp, this file is cleared on reboot. If the kill switch was activated due to a severe issue and the system reboots (e.g., OOM-killer, power cycle), the flag would be lost and the system could restart trading. The flag is explicitly checked in TradingLoop.start() (trading_loop.py:796-799) and blocks restart, but only if the flag survives.
- **Recommendation:** Use a persistent path (e.g., in the project's data directory or a configurable path that maps to persistent storage in Docker deployments). The flag_path parameter is already injectable, so this is a configuration change.
- **Note:** finder claimed LOW; verification corrected to MEDIUM (high confidence).

#### [MEDIUM] Sector exposure calculation sums ALL open positions, not just same-sector positions

- **Dimension:** Risk & live-trade safety
- **Location:** `src/finalayze/orchestration/signal_executor.py` 971-983
- **Evidence:** _compute_sector_exposure iterates ALL positions in the portfolio (line 980: 'for pos_symbol, qty in portfolio.positions.items()') and sums their values, regardless of which sector/segment they belong to. This total is then compared to the 40% sector concentration limit by SectorConcentrationCheck (pre_trade_check.py:255: 'concentration = (ctx.sector_exposure_value + ctx.order_value) / ctx.portfolio_equity'). Since sector_exposure_value includes ALL positions (not just same-sector), the check would ALWAYS exceed 40% once total invested exceeds 40%, making it impossible to open new positions even in underrepresented sectors. This is effectively a duplicate of the cash reserve check rather than a sector concentration check.
- **Recommendation:** Fix _compute_sector_exposure to filter positions by segment: only sum positions that belong to the same segment as seg_id. This requires either a segment lookup per position symbol (via InstrumentRegistry) or tracking segment ownership in PositionTracker alongside strategy ownership.

#### [MEDIUM] ML pickle integrity verification is opt-in and disabled by default (HMAC key defaults empty)

- **Dimension:** Security
- **Location:** `src/finalayze/ml/models/xgboost_model.py` 144-150 (mirrored in lightgbm_model.py:146, catboost_model.py:143; settings.py:118)
- **Evidence:** load_from only verifies when `_get_hmac_key()` returns non-empty: `key = _get_hmac_key(); if key: verify_model(...)` then `return joblib.load(path)`. The key comes from Settings.ml_model_hmac_key which defaults to "" (settings.py:118) and is absent from .env.example, so in a default deployment NO model file is integrity-checked before joblib.load. The control is present but off-by-default, which usually means it is off in practice.
- **Recommendation:** Either (a) make the HMAC key mandatory whenever ml_enabled=True (fail-closed at Settings validation), or (b) fail-closed at load time when a .sha256 sidecar is missing once any signed model is detected, so the absence of a key cannot silently downgrade to unverified pickle loading. At minimum document the requirement to set FINALAYZE_ML_MODEL_HMAC_KEY in .env.example and the ML AGENTS.md.

#### [MEDIUM] Dashboard login uses non-constant-time password comparison with a default 'admin' credential in the deployment template

- **Dimension:** Security
- **Location:** `src/finalayze/dashboard/app.py` 37 (default injected by docker/streamlit-entrypoint.sh:7)
- **Evidence:** app.py:37 compares `if pwd == _PASSWORD` (plain ==, timing-observable). The Streamlit container entrypoint writes the secrets file with `password = "${DASHBOARD_PASSWORD:-admin}"` (docker/streamlit-entrypoint.sh:7), so a Docker deployment that forgets to set DASHBOARD_PASSWORD silently ships with a guessable default password of 'admin'. The dashboard itself fails closed when no password is set (app.py:32-34), but the entrypoint's :-admin default defeats that safeguard. The dashboard is read-only/preview, limiting blast radius, but it exposes portfolio/positions/trades data. (Same default-credential pattern: GRAFANA_PASSWORD:-admin in docker-compose.sandbox.yml.)
- **Recommendation:** Make DASHBOARD_PASSWORD required: change the entrypoint to `${DASHBOARD_PASSWORD:?DASHBOARD_PASSWORD must be set}` (matching the POSTGRES_PASSWORD pattern already used in the compose files) so the container fails to start without an explicit password. Replace `pwd == _PASSWORD` with hmac.compare_digest. Apply the same :? treatment to GRAFANA_PASSWORD.

#### [MEDIUM] All live-DB Alembic migration verification is skipped in default CI; migration 014 has only a weak substring-presence static guard

- **Dimension:** Tests & open debt
- **Location:** `tests/integration/migrations/test_014_rebalance_reason_text.py` 44-49
- **Evidence:** test_upgrade_alters_reason_to_text asserts only that the strings 'alter_column', '"saa_rebalance_orders"', '"reason"', and 'sa.Text()' each appear SOMEWHERE in the migration source. It never parses them into a single call nor runs the migration, so a migration that alters the wrong column/type (with an unrelated sa.Text() elsewhere) would still pass all four assertions. The only test that actually runs `alembic upgrade head` (tests/integration/test_alembic_upgrade.py:10-14 _db_url) calls pytest.skip whenever FINALAYZE_DATABASE_URL is unset -- i.e. in normal CI. Net effect: in default CI no migration is ever executed, and the migration data-integrity path (this system persists real rebalance/order audit rows) is guarded only by substring presence.
- **Recommendation:** Strengthen the static test to ast-parse the op.alter_column(...) call and assert column name + Text type as a single node (mirror what test_013 does for column adds). Run the live-DB migration suite in CI against an ephemeral Postgres/TimescaleDB service container so `alembic upgrade head` is exercised on every push, not only when a developer happens to set a DB URL.

### 3.3 LOW / INFO (condensed)

| Sev | Dimension | Title | Location |
|---|---|---|---|
| LOW | architecture | L0 core/ shim modules import L5/L6 at module level, creating core<->api and core<->orchestration import cycles | `src/finalayze/core/alerts.py` 12 (also core/trading_loop.py:12, core/bond_cycle.py:12, core/telegram_bot.py:12) |
| LOW | architecture | core/db.py is labelled 'Layer 2' but lives in L0/core and imports L1 config | `src/finalayze/core/db.py` 1, 49 |
| LOW | architecture | Per-call engine creation + fresh event loop in fundamental-snapshot loader | `src/finalayze/data/loader.py` 347, 368-370 |
| LOW | architecture | Redis client leaked on health-check failure (aclose skipped in except branch) | `src/finalayze/api/v1/system.py` 186-198 |
| LOW | architecture | Orphaned module: core/bond_math_quantlib.py has no production callers | `src/finalayze/core/bond_math_quantlib.py` n/a (whole file) |
| LOW | architecture | Documented Redis Streams EventBus (core/events.py) does not exist; stale docstrings reference it | `src/finalayze/data/cache.py` 8 |
| LOW | financial | Deposit interest NDFL bypasses the progressive 13/15% band | `src/finalayze/core/ndfl.py` 84-98 |
| LOW | financial | fund_underweight step 2 double-counts interest already inside tranche marks | `src/finalayze/orchestration/allocation.py` 230-235 |
| LOW | financial | BondSimulatedBroker.process_coupons lacks idempotency guard for sub-daily bars | `src/finalayze/execution/bond_simulated_broker.py` 136-163 |
| LOW | financial | Allocation friction applied as external drag, not reducing investable base | `src/finalayze/orchestration/allocation.py` 632-642 |
| LOW | ml_data | Gate-failed / force-saved models are loaded and trade live with only a log warning | `src/finalayze/ml/loader.py` 71-82 |
| LOW | quality | `news_cycle_minutes` default left at debug value 2 with TODO revert — 15x prod news/LLM call volume | `config/settings.py` 97 |
| LOW | quality | Startup component-wiring failures swallowed at DEBUG hide operationally significant losses | `src/finalayze/main.py` 89, 136 |
| LOW | quality | FX fallback constant `_FALLBACK_USDRUB = Decimal("90.0")` triplicated and placed mid-import | `src/finalayze/markets/currency.py` 16 |
| LOW | quality | Magic-number sentinel `Decimal(999999)` used to disable trailing-stop activation | `src/finalayze/execution/simulated_broker.py` 146-149 |
| LOW | quality | cancel_order(order_id) pops _stop_states by order_id but stops are keyed by symbol — a no-op that pretends to act | `src/finalayze/execution/simulated_broker.py` 345-347 |
| LOW | quality | Static CPI table is manually maintained and feeds ML/decision features (staleness is warn-only) | `src/finalayze/data/fetchers/cbr.py` 580-591 |
| LOW | risk_live | KillSwitch.activate uses type: ignore for broker method calls -- no compile-time safety | `src/finalayze/core/kill_switch.py` 89-90 |
| LOW | security | Telegram webhook secret-token comparison is not constant-time | `src/finalayze/api/v1/telegram.py` 42 |
| LOW | security | SANDBOX/DEBUG/TEST modes silently fall back to a default-credential database URL | `config/settings.py` 202-205 |
| LOW | tests_debt | Trade-detail endpoint returns 500 (not 404) for a missing trade; the 404 branch is entirely uncovered | `src/finalayze/api/v1/trades.py` 303-331 |
| LOW | tests_debt | Best-effort audit-persist error-swallow path on a completed rebalance has no test coverage | `src/finalayze/orchestration/rebalance_execution.py` 272-277 |
| LOW | tests_debt | run_with_active_budget (budget -> opening-notional money path) is covered only by a DB-gated integration test, skipped in default CI | `tests/integration/test_budget_driver_integration.py` 78-83 |
| LOW | tests_debt | NDFL realized-gain cross-check uses static weights, not the tilted weights the production path applies; passes only by coincidental weight equality | `tests/unit/test_allocation_orchestrator.py` 184-197 |
| LOW | tests_debt | Committed dividend snapshot is stale (fetched 2026-03-20) and has no freshness guard; under-credits dividends in total-return backtests | `src/finalayze/strategies/presets/moex_dividends.yaml` 1-2 |
| LOW | tests_debt | Migration-014 test is the only default-CI guard yet asserts substrings, not the alter_column semantics (partially hollow) | `tests/integration/migrations/test_014_rebalance_reason_text.py` 3, 44-49 |
| INFO | financial | Backtest Sharpe risk-free rate inconsistency between SAA and equity paths | `src/finalayze/orchestration/allocation.py` 104 |
| INFO | ml_data | RateLimiter under-credits tokens after waiting; documented non-thread-safe | `src/finalayze/data/rate_limiter.py` 45-50 |
| INFO | risk_live | PositionSizingPipeline floor is 15% of base_position via RegimeStep, not pipeline-level enforcement | `src/finalayze/risk/position_sizing_pipeline.py` 82-86 |
| INFO | security | run_rebalance.py --mode live passes the LIVE triple gate but routes to the sandbox endpoint (gate is theatrical from this CLI) | `scripts/run_rebalance.py` 215-225 |
| INFO | tests_debt | REQUIREMENTS.md traceability desync: documents only through Phase 70 while the codebase is at Phase 88 | `.planning/REQUIREMENTS.md` tail (Last updated line) |
| INFO | tests_debt | v10.x and v11.0 milestone archives missing (archival/tracking lag) | `.planning/milestones/` n/a |
| INFO | tests_debt | MEMORY.md is 7.2KB over its stated size limit | `MEMORY.md` n/a |

---

## 4. Rejected findings (transparency)

These were flagged by a finder but **disproven** during adversarial verification — listed so the audit is auditable:

- **[financial] CurrencyConverter inverse rate non-terminating Decimal division** — Cited line is accurate: currency.py:79 computes `self._rates[reverse_pair] = Decimal(1) / rate` (same pattern at line 34). BUT the finding's central technical claim is FALSE: I tested `Decimal(1)/Decimal(3)` in this repo's Python and it does NOT raise InvalidOperation — it returns a value rounded to
- **[risk_live] PDT rolling window uses 7 calendar days instead of 5 business days** — Confirmed the cited code at src/finalayze/risk/pre_trade_check.py:137: _count_recent_day_trades uses cutoff = as_of - timedelta(days=7) with an inclusive boundary (drops d < cutoff, counts d >= cutoff). The PDT path is genuinely live-wired (PDTTracker built+injected in trading_loop.py:354-359, recor
- **[risk_live] Circuit breaker check uses float for thresholds, mixes with Decimal drawdown** — Reproduced the cited code in /Users/f1xgun/finalayze/.claude/worktrees/regress-main/src/finalayze/risk/circuit_breaker.py. The factual claims hold: __init__ accepts float thresholds (lines 62-64), converts them via Decimal(str(...)) (lines 68-70), drawdown is pure Decimal arithmetic (line 110), and 
- **[risk_live] LayerCircuitBreaker allows intraday de-escalation unlike per-market CircuitBreaker** — The CODE behavior the finding describes is accurate: LayerCircuitBreaker.update() (src/finalayze/risk/layer_circuit_breaker.py:105-134) recomputes the level from scratch each call and CAN de-escalate (e.g. CAUTION->NORMAL) on recovery, whereas the per-market CircuitBreaker.check() (circuit_breaker.p
- **[ml_data] DataNormalizer does not explicitly reject NaN prices (Decimal NaN slips some checks)** — Verified against actual code at /Users/f1xgun/finalayze/.claude/worktrees/regress-main. The _validate code at normalizer.py:47-68 matches the citation, but the finding's core technical claims are factually wrong and the vulnerability is not reproducible.  (1) WRONG NaN SEMANTICS: The finding asserts

---

## 5. Per-dimension assessment

### Architecture & layering

The codebase has a real, root-caused PostgreSQL connection leak plus systemic layer-invariant erosion. The leak is structural: SQLAlchemy async engines are created in three places (core/db.py cached-by-URL, orchestration/db_persistence.py cached per-event-loop-id, data/loader.py per-call) but the ONLY production call to engine.dispose() is in data/loader.py. db.py's reset_engine() merely .clear()s the cache dicts without disposing; TradingPersistence has no close/dispose method; and TradingLoop.stop()/AsyncRuntime.shutdown() tear down event loops without disposing the per-loop engines keyed to them. Because db_persistence keys engines on id(asyncio.get_running_loop()), every loop recreation (reconnect, stop/start, meta-agent running on the uvicorn loop vs the background loop) spins a brand-new engine+pool (pool_size=5+overflow=2=7 connections each) while the prior engine's asyncpg connections stay open server-side -> "too many clients" after weeks of uptime. Separately, the dependency-layer invariant (imports flow 0->6 downward only) is NOT mechanically enforced: DEPENDENCY_LAYERS.md promises an import-linter config and a tests/test_architecture.py, but neither exists; graph_check.py only validates AGENTS.md<->manifest consistency, not actual imports. As a result ~33 upward import references have accumulated. The dominant cause is that cross-cutting infrastructure (TelegramAlerter in api/alerts.py and Prometheus counters in api/metrics.py) is misplaced at L6 yet consumed by L0 (kill_switch, layer_ledger) and pervasively across L5 orchestration via function-local "deferred" imports that dodge the static cycle. Four L0 sys.modules shim files import L5/L6 at module level, creating genuine core<->api and core<->orchestration import cycles (test-only callers). Async correctness is otherwise good: zero time.sleep/requests.* inside async functions, sync httpx.Client confined to fetchers, and asyncio.run is guarded against running-loop reentry. The documented "Redis Streams EventBus (core/events.py)" does not exist (stale docstrings reference it); Redis is cache+health-ping only.

### Security

The security posture of the live-trade and secrets paths is strong. The real-money LIVE hard stop is robustly enforced through multiple independent layers: a triple gate in rebalance_executor._enforce_live_gate (plan.mode==LIVE AND confirm=True AND WorkMode.REAL, which itself requires FINALAYZE_REAL_CONFIRMED=true), safe-by-default flags everywhere (TinkoffBroker sandbox=True, AlpacaBroker paper=True, Settings tinkoff_sandbox/alpaca_paper True, even docker-compose.prod defaults FINALAYZE_MODE=sandbox), and the only order-placing CLI (scripts/run_rebalance.py) hardcodes sandbox=True so its own --mode live still cannot reach the production endpoint. No code path defaults to live, paper=False appears nowhere, and AlpacaBroker is not wired into any runtime entry point. No hardcoded real secrets are committed (only .env.example/secrets.toml.example templates and env-var interpolation with :? fail-if-unset on critical DB passwords); the MOEX-via-Tinkoff invariant holds (YFinanceFetcher has zero usages). API auth is broadly correct: every mutating REST endpoint applies api_key_auth (constant-time hmac.compare_digest), POST /mode to REAL additionally requires a confirm_token, and CORS defaults to an empty allowlist (no wildcard). The two material weaknesses are in the ML pickle-deserialization integrity scheme (verification is opt-in by default and two artifacts are never verified at all) and a pair of timing-unsafe secret comparisons plus default 'admin' dashboard/grafana credentials in the deployment templates. The LLM prompt-injection surface (news text -> sentiment) is inherent but bounded: output is parsed into typed Pydantic models, not executed, and the headless-Claude client passes prompts as argv (no shell/command injection).

### Financial correctness

The codebase demonstrates strong financial discipline overall: all core money paths (deposit accrual, rebalance planning, order sizing, NDFL marginal band) use Decimal arithmetic end-to-end; the API serializes money as strings; the live execution path has a proper triple gate. Two real financial correctness issues exist: (1) the deposit interest NDFL uses a flat 13% rate instead of routing through the progressive 13/15% band that applies when total taxable income exceeds 2.4M RUB -- this under-taxes high-income scenarios and is inconsistent with how equity dividends are taxed in the same system; (2) the fund_underweight funding-order helper treats cumulative interest_income_net as an independent liquid pool available for withdrawal, but this income is already inside the tranche marks (accrued_net) and would be partially double-counted against matured-tranche withdrawals. The helper is currently unwired or advisory-only, limiting the blast radius. The allocation orchestrator applies rebalance friction as a cumulative drag on the reported curve rather than reducing the investable base, which is a minor conservatism gap. No look-ahead bias was found: strategies use history[:i+1], fills use candles[i+1].open, CBR rate lookups filter to meetings <= as_of, and CPI respects publication dates.

### Risk & live-trade safety

The risk management subsystem is architecturally sound with 14 pre-trade checks (exceeding the documented 11), proper circuit breaker escalation with sticky intraday levels, Decimal-based money math throughout the critical path, and a kill switch that correctly sequences cancel-orders/stop-scheduler/escalate-breakers/persist-flag/alert. However, the audit identified two critical findings that compromise live-trade safety: (1) the pre-trade circuit breaker check is permanently bypassed because SignalExecutor._get_circuit_breaker_level() is hard-coded to return None -- while the outer TradingLoop gates HALTED/LIQUIDATE before reaching the pre-trade pipeline, this means the CircuitBreakerCheck class is dead code in the pre-trade path, creating a false sense of defense-in-depth, and (2) the StopLossRequiredCheck is semantically inverted -- it only checks existing stops rather than enforcing that new BUY orders must have a stop-loss. Additionally, the live correlation check (check 14) and parameter freshness check (check 13) are structurally wired but permanently ineffective. Position sizing is well-bounded with a 15% RegimeStep floor, 20% hard cap, and min-position elimination. ATR stops and chandelier exits use segment-specific multipliers with MOEX 1.2x uplift and a grace-bar mechanism. The PDT tracker uses a 7-calendar-day window which approximately covers 5 business days but can undercount around US holidays.

### ML & data integrity

The core ML feature-engineering, labeling, temporal splitting, CPCV purging, calibration (Platt→isotonic→bypass+clamp + conformal), LSTM thread-safe inference, and ensemble graceful-degradation are largely correct and well-defended against look-ahead bias — no future/same-bar-close peeking was found in technical.py, and external (macro/CBR/fundamental) data is consistently as-of/lag-filtered and time-sliced per training window. However, I found two genuine data-integrity / live-trade-safety defects. (1) HIGH: a cross-symbol fundamental-feature contamination bug — compute_fundamental_features picks the target symbol as the globally-latest snapshot across the whole segment instead of the symbol actually being scored, so 5 of the ~70 features (earnings_yield, pe_zscore_vs_sector, revenue_growth_yoy, net_margin_trend, dividend_yield_z) are wrong for every symbol except one, in BOTH training and inference. (2) HIGH: the training/inference candle fetcher silently falls back to yfinance for MOEX (ru_*) tickers when Tinkoff returns empty, directly violating the hard "MOEX = Tinkoff gRPC only" invariant. Plus a MEDIUM live-trade-safety gap (force-saved / gate-failed models are loaded and traded with only a log warning, and us_tech ml_ensemble is enabled), and two LOW/INFO items (normalizer NaN gap, rate-limiter under-credit). MOEX ml_ensemble is currently disabled in all ru_* presets, which limits live blast radius of #1/#2 today, but both silently corrupt any future MOEX ML enablement.

### Code quality

Error handling across this codebase is genuinely disciplined for a real-money system: the order-execution and rebalance paths catch broadly only where isolation is intended (one leg's failure must not abort others; best-effort audit writes after orders are placed), almost always logging at error/warning with context and re-raising or returning a typed failure result. There are no bare `except:` in source, and the few `contextlib.suppress(Exception/BaseException)` uses are justified (channel cleanup, task-cancellation shutdown). The money math (deposit broker accrual, dividend NDFL, simulated fills) consistently uses Decimal and is well-commented. The real defects are in maintainability, not correctness: (1) the known production logic re-implemented in tests has already DRIFTED so the test no longer guards the shipped code; (2) the FX fallback constant is triplicated and hand-placed mid-import; (3) a debug config override (`news_cycle_minutes=2`) and a non-functional API stub were left in shippable code; and (4) several startup-wiring failures are swallowed at DEBUG, hiding operationally significant wiring loss. None are CRITICAL money-loss bugs, but the test-drift and config-override findings are HIGH because they erode the safety net and change production behavior silently.

### Tests & open debt

The high-risk SAA/allocation/gate/rebalance test surface -- the exact area with documented hollow-test history (the forced_leg_deltas hook that made cost/NDFL fire only in tests) -- is now genuinely robust. That hook is gone; cost and realized-gains NDFL are computed from the real per-leg rescale delta and verified by independent FIFO replays (test_allocation_orchestrator.py), the binding gate is conjunctive and pinned with flip-each-condition tests, and a dedicated regression test (test_phase76_duration_tilt.py:279 test_gate_candidate_applies_tilt_not_just_orchestrator) directly guards the Phase-76 bug where the tilt never reached the binding gate. The live-execution triple gate, deposit accrual/break-forfeit conservation, bond %-of-face price conversion, and the deposit-ladder optimizer all carry strong anti-hollow assertions. Only 2 skip markers exist (both legitimate env gates), no xfail, no flaky markers. The real weaknesses are: (1) a confirmed known-debt 404-vs-500 API mismatch whose 404 branch is entirely uncovered; (2) all live-DB migration verification is pytest.skip'd in default CI, leaving migration 014 covered only by a weak substring-presence static test; (3) the audit-persist best-effort error-swallow path and the run_with_active_budget DB path have no default-CI coverage; (4) one semi-hollow NDFL cross-check that passes only by a coincidental weight equality; and (5) confirmed housekeeping debt (stale dividend snapshot, MEMORY.md 7.2KB over limit, REQUIREMENTS.md desync, v10.x archival lag).

---

## 6. Recommended remediation priority

1. **F-arch (HIGH) connection leak** — add `TradingPersistence.dispose_all()` + dispose engines in `TradingLoop.stop()`/`AsyncRuntime.shutdown()`; make `db.py reset_engine()` actually dispose. *This is the production `too many clients` cause.*
2. **ml_data (HIGH) yfinance MOEX fallback** — make the MOEX branch fail-closed; never call `YFinanceFetcher` for `ru_*`. *Hard-invariant violation.*
3. **security (HIGH) unverified pickle loads** — route `calibrator.pkl`/`meta_learner.pkl` through the same HMAC `verify_model()` as the boosting models.
4. **MEDIUM cluster** — enforce the layer invariant mechanically (import-linter + `test_architecture.py`), fix the inert pre-trade checks (CB/stop-loss/correlation/freshness are dead wiring → either implement or remove the false safety net), progressive 13/15%% deposit NDFL, fundamental cross-symbol contamination, KillSwitch durable path, dashboard constant-time compare.
5. **tests_debt** — fix the 404-vs-500 endpoint (or its test), de-drift `test_moex_fixes.py`, restore default-CI coverage for migration/audit-persist paths, refresh stale `moex_dividends.yaml`, trim `MEMORY.md`, reconcile REQUIREMENTS.md / v10.x archival.

_No code was changed by this audit. Findings are recommendations; each fix should follow the normal TDD → review → PR cycle. The real-money hard stop remains intact._
