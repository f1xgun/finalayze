# Project Research Summary

**Project:** Finalayze — Autonomous MOEX Trading (bonds, stocks, LLM news, Telegram)
**Domain:** Autonomous multi-asset trading on MOEX — OFZ bonds, equities, CBR macro-driven strategies, LLM news analysis, Telegram alerting
**Researched:** 2026-03-14
**Confidence:** HIGH (stack: HIGH, features: HIGH, architecture: HIGH, pitfalls: HIGH)

## Executive Summary

Finalayze already has a mature, well-layered trading engine for US equities. The MOEX milestone is fundamentally an integration and completion project, not a greenfield build. The core codebase (Python 3.12, APScheduler, T-Invest gRPC SDK, anthropic client, FastAPI) provides the foundation; the gaps are specific: RUB position sizing is broken (MOEX positions size at 0.02% instead of 15%), bond cycle execution stubs are incomplete, the MacroCacheService macro injection path is not yet wired into the TradingLoop bond cycle, and no live Russian news source is connected. Four new dependencies are required: QuantLib (bond math), aiogram (Telegram bot alerts outbound), Telethon (Telegram channel reading inbound), and fastfeedparser (Russian news RSS). All four have high-confidence, actively maintained releases as of early 2026.

The recommended build order is dictated by a hard dependency chain: data layer correctness (RUB sizing, bond candle fetching, macro cache) must come first, because every downstream component — bond strategy calibration, risk wiring, sandbox validation — depends on receiving correctly sized, correctly sourced data. Bond cycle execution is the critical path to MOEX MVP; news pipeline and Telegram hardening are the next tier. The four-layer OFZ bond portfolio architecture (Core/Strategic/Tactical/Short with per-layer LayerLedger and circuit breakers) is the key differentiator versus simple MOEX trading bots; it must be wired completely before any live validation.

The highest-severity risks are financial: sizing against clean price instead of dirty price (NKD) can cause broker account overdraft; applying equity ATR stop-loss logic to bonds will trigger constant spurious exits; sharing one TinkoffBroker instance across concurrent APScheduler equity and bond cycles causes gRPC thread-safety failures; and the in-memory LayerLedger will diverge from actual broker state after any process crash, creating duplicate positions. Every one of these has a clear prevention strategy documented in the codebase design docs. Addressing them in the correct phase order eliminates the risk before real money is deployed.

---

## Key Findings

### Recommended Stack

The existing stack requires only four new dependencies for MOEX MVP. No existing libraries need replacement. The T-Invest gRPC SDK already exposes `GetBonds`, `GetBondCoupons`, and `GetAccruedInterests` methods — they just need to be wired into `TinkoffFetcher`. The `anthropic` client already handles Russian-language text via `sentiment_ru.txt` prompts. APScheduler, httpx, and lxml are already in the stack and handle scheduling, async HTTP, and HTML parsing respectively.

See `.planning/research/STACK.md` for full rationale, version pinning, and alternatives considered.

**Core technologies (new additions only):**
- **QuantLib >= 1.41**: Bond math — YTM, modified duration, convexity, NKD accrued interest. Industry-standard, precompiled wheels for Python 3.12, no C++ build required.
- **aiogram >= 3.26**: Outbound Telegram bot for trade alerts and P&L reports. Fully async, strict typing, matches existing async-first architecture. Preferred over python-telegram-bot for mypy compatibility.
- **Telethon >= 1.42**: Inbound Telegram channel reading via MTProto user account. Only maintained MTProto library (Pyrogram is abandoned). Needed to scrape financial channels that Bot API cannot access.
- **fastfeedparser >= 0.5.9**: Parse Russian news RSS (RBC, Interfax, TASS, Kommersant). 25x faster than feedparser, active maintenance (Mar 2026). Pair with existing httpx for concurrent async fetching.

**Russian news RSS endpoints confirmed (MEDIUM confidence — validate at implementation):**
- RBC: `http://static.feed.rbc.ru/rbc/logical/footer/news.rss` — primary financial source
- Interfax: `https://www.interfax.ru/rss.asp` — commercially independent wire service
- TASS: `http://tass.ru/rss/v2.xml` — state wire, useful for macro/geopolitical
- Kommersant: `https://www.kommersant.ru/RSS/main.xml` — business newspaper

---

### Expected Features

See `.planning/research/FEATURES.md` for dependency graph, prioritization matrix, and MVP definition.

**Must have (P1 — MOEX MVP):**
- RUB position sizing fix — confirmed bug; MOEX positions at 0.02% instead of 15%; blocks everything
- MOEX walk-forward backtest with positive PnL (ru_blue_chips + ru_energy minimum)
- MOEX strategy parameter tuning (ru_* YAML presets calibrated to Russian market volatility)
- OFZ bond cycle wired into TradingLoop (BondCycleProcessor fully connected, all 4 layers)
- MacroCacheService with daily CBR refresh and CBR-day force-refresh at 15:30 MSK
- Bond backtest with positive PnL (OFZ-PD duration rotation + OFZ-PK carry validated)
- Telegram daily P&L summary bug fix (currently shows zero)
- Coupon payment detection and Telegram alert
- MOEX trading hours gate validation (MOEX main session 07:00–15:40 UTC)
- Lot-size-aware ordering (MOEX lots enforce minimum order quantities)
- Sandbox validation: 5+ consecutive autonomous trading days without critical errors

**Should have (P2 — after sandbox validation):**
- Russian news feed integration (T-Invest API news or RBC/Interfax RSS via fastfeedparser)
- LLM-powered Russian news analysis wired to event_driven strategy
- Telegram /status command (on-demand position and P&L view)
- Real money deployment (small account, 500K RUB, 5+ sandbox days proven)

**Defer (v2+):**
- ML ensemble for MOEX ru_* segments (requires 3+ years of MOEX candle history and quality gates)
- PEAD strategy for MOEX (requires Russian earnings surprise data — not readily available)
- Pairs trading on MOEX (cointegration testing needed; sanctions-driven correlation breaks are a risk)
- Telegram financial channel monitoring (noise-to-signal filtering is a research problem)
- Portfolio optimization with HRP weights across MOEX segments

**Anti-features (explicitly excluded):**
- Real-time tick-level / HFT trading — out of scope, different API tier, different risk model
- Full Telegram trading commands (/buy, /sell) — bypass risk pipeline and circuit breakers; use T-Invest app directly for manual trades
- Automated ML model retraining in production — model drift risk; offline retraining + human review only
- Cryptocurrency trading — not on MOEX, violates broker constraint

---

### Architecture Approach

The architecture is a 6-layer dependency-ordered system (L0 types → L6 orchestration) with strict downward-only imports. New MOEX components slot into existing layers without restructuring. The system is organized around three independent APScheduler cycles: news (5 min), equity strategy (15 min), and bond (daily/on-event). Each cycle is fault-isolated. Bond strategies deliberately do not subclass `BaseStrategy` — they have a separate `generate_signal(symbol, candles, open_positions, bar_idx, **macro_kwargs)` interface because they need CBR/macro kwargs that have no meaning for equity strategies.

See `.planning/research/ARCHITECTURE.md` for layer diagram, data flow diagrams, component responsibility table, build order, and anti-patterns.

**Major components and current state:**

| Component | Layer | Status |
|-----------|-------|--------|
| TradingLoop (APScheduler, 3 cycles) | L6 | Done — needs bond cycle wired |
| BondCycleProcessor (4-layer pipeline) | L6 | Done — stubs incomplete (_size_and_execute, _process_yield_stops) |
| TelegramAlerter (9 alert types) | L6 | Done — daily P&L bug; no rate limiting |
| BrokerRouter → TinkoffBroker / AlpacaBroker | L5 | Done — needs separate "moex_bonds" TinkoffBroker instance |
| StrategyCombiner + 5 equity strategies | L4 | Done |
| Bond strategies (BondCarry, DurationRotation, CBREvent) | L4 | Done — need MacroCacheService injection |
| Risk pipeline (DV01BudgetStep, YieldStop, LayerCircuitBreaker) | L4 | Done — YieldStop stub incomplete |
| NewsAnalyzer + LLMClient (EN/RU) | L3 | Done — no live Russian news source connected |
| ML Ensemble (XGBoost + LightGBM + CatBoost) | L3 | Done for US; MOEX models untrained |
| TinkoffFetcher (candles, dividends, instruments) | L2 | Done — bond methods need wiring |
| CBRFetcher + MacroCacheService | L2 | Done — refresh schedule not wired into TradingLoop |
| InstrumentRegistry, CurrencyConverter | L2 | Done |

**Key patterns from architecture research:**
1. Scheduled cycle decomposition — each new periodic process (news polling, macro refresh) must be a separate APScheduler job, not embedded in the strategy cycle
2. 4-layer bond portfolio with independent LayerLedgers — Core/Strategic/Tactical/Short with per-layer circuit breakers
3. Macro context injection via MacroCacheService — bond strategies never call CBRFetcher directly; BondCycleProcessor injects MacroSnapshot as kwargs
4. News pipeline → sentiment cache → event-driven signals — LLM calls happen in the news cycle, never blocking the strategy cycle
5. Fire-and-forget alerting with exception suppression — all TelegramAlerter methods catch all exceptions internally; must be hardened with rate limiting before go-live

---

### Critical Pitfalls

See `.planning/research/PITFALLS.md` for all 12 pitfalls with warning signs, recovery costs, and phase mapping.

**Top 5 must-prevent pitfalls:**

1. **RUB position sizing with USD-derived Kelly figures** — Confirmed existing bug. MOEX positions at 0.02% instead of 15%. Fix: all MOEX risk checks must use RUB-denominated equity from TinkoffBroker.get_portfolio(); never convert MOEX equity to USD for sizing. Phase: first MOEX-specific backtest phase.

2. **Sizing against clean price instead of dirty price (NKD)** — Cash sufficiency check using clean price approves orders that overdraw account by 3–5% of face value. Fix: `_validate_orders()` in BondCycleProcessor must use `dirty_price = (clean_price_pct / 100) * face_value + nkd_per_bond`; DV01BudgetStep must accept dirty_price_per_bond for position cap. Phase: bond cycle execution completion.

3. **TinkoffBroker thread-safety across equity and bond cycles** — APScheduler runs each job in a separate thread; sharing one AsyncClient across threads causes intermittent gRPC failures and silent order drops. Fix: register a separate `TinkoffBroker` instance under key `"moex_bonds"` in BrokerRouter. Phase: bond cycle TradingLoop integration.

4. **LayerLedger diverging from actual broker portfolio after crashes** — In-memory ledger resets on restart; ghost positions cause duplicate bond exposure. Fix: reconcile LayerLedger.positions against TinkoffBroker.get_positions() on every startup before processing new signals. Phase: sandbox validation (must be resolved before go-live).

5. **Telegram message storm during circuit breaker liquidation** — 20 fills in 5 seconds triggers Telegram 429 rate limit; fire-and-forget drops all alerts at exactly the moment the operator needs them most. Fix: implement priority message queue with 1/second drain; batch liquidation fills into single message; respect retry_after on 429. Phase: Telegram hardening (before real-money deployment).

**Additional critical pitfalls to track:**
- Applying equity ATR stop-loss logic to bonds (YieldStop stub returns 0 — must be completed before live trading)
- MOEX holiday calendar not connected to bond cycle scheduler (currently only weekends are blocked)
- CBR announcement timing — extra bond cycle must trigger at 15:30 MSK (after press conference), not 13:30 MSK (spread spike window)
- Extended holiday macro staleness — macro refresh schedule must run 7 days/week, independent of trading day gate
- Coupon record date estimation (2-day buffer is wrong around holidays; use 3-day conservative buffer or MOEX ISS securityevents lookup)

---

## Implications for Roadmap

Based on combined research, the dependency chain dictates a clear 7-phase build order. No phase can safely be reordered because each phase's gate condition is an input to the next.

### Phase 1: RUB Sizing and MOEX Data Foundation

**Rationale:** RUB position sizing is the confirmed blocker for all MOEX work. Without correct sizing, every backtest, sandbox run, and live trade produces wrong results. This phase must come first — it has zero dependencies on other new work and unblocks everything.

**Delivers:** Correctly sized MOEX equity trades; validated TinkoffFetcher bond instrument discovery; InstrumentRegistry populated with OFZ bonds; MOEX trading hours gate connected.

**Addresses (from FEATURES.md P1):** RUB position sizing fix, MOEX trading hours gate, lot-size-aware ordering, FIGI-based instrument identification.

**Avoids (from PITFALLS.md):** Pitfall 7 (RUB/USD sizing bug).

**Gate:** MOEX backtest shows positions at 10–20% of MOEX RUB equity, not 0.02%.

**Research flag:** Standard patterns — no additional research needed. Fix is well-defined (use RUB-denominated equity from TinkoffBroker.get_portfolio()).

---

### Phase 2: MOEX Equity Backtest and Strategy Tuning

**Rationale:** Once sizing is correct, the equity backtest pipeline can produce trustworthy MOEX results for the first time. Strategy parameter tuning (ru_* YAML presets calibrated to Russian market volatility) requires known-good backtest infrastructure.

**Delivers:** Positive walk-forward backtest PnL on ru_blue_chips and ru_energy segments; calibrated ru_* strategy presets; ADX regime thresholds validated for MOEX volatility.

**Addresses (from FEATURES.md P1):** MOEX walk-forward backtest, MOEX strategy parameter tuning.

**Gate:** Positive walk-forward Sharpe > 0 on at least 2 MOEX segments over 2022–2025 out-of-sample period.

**Research flag:** May need `/gsd:research-phase` for MOEX-specific ADX threshold calibration and RUONIA-correlated volatility regime thresholds. Russian equity volatility patterns differ from US.

---

### Phase 3: Bond Data Pipeline and MacroCacheService Wiring

**Rationale:** Bond strategies require MacroCacheService providing live CBR data. MacroCacheService must be connected to TradingLoop's daily refresh job, separate from the bond cycle gate. Bond candle fetching (90-day OFZ series) must be validated against live T-Invest API. NKD computation and dirty price must be implemented before any bond sizing occurs.

**Delivers:** MacroCacheService wired with daily refresh + CBR-day force-refresh at 15:30 MSK; TinkoffFetcher bond methods fully implemented (GetBondCoupons, GetAccruedInterests); InstrumentRegistry populated with bond metadata; coupon record date buffering with 3-day conservative guard.

**Addresses (from FEATURES.md P1):** MacroCacheService, CBR key rate as macro input, coupon schedule tracking.

**Avoids (from PITFALLS.md):** Pitfall 1 (NKD-aware dirty price sizing), Pitfall 11 (extended holiday macro staleness), Pitfall 12 (coupon record date estimation errors).

**Gate:** Bond candle fetch returns non-empty series for OFZ-PD and OFZ-PK instruments; MacroSnapshot.key_rate matches cbr.ru within 24 hours; dirty price calculation validated against known OFZ settlement examples.

**Research flag:** Standard patterns for CBR API and T-Invest bond methods — both are documented in codebase design docs with high confidence.

---

### Phase 4: Bond Cycle Execution Completion and Calibration

**Rationale:** With data layer complete (Phase 3), the bond cycle execution stubs can be completed and calibrated. YieldStop must be implemented before any live bond trading. DV01BudgetStep must use dirty price. Separate TinkoffBroker instance for bond cycle must be wired. Bond backtest must prove positive PnL.

**Delivers:** BondCycleProcessor fully operational (no stubs); YieldStop evaluating positions with regime-adaptive thresholds; DV01BudgetStep using dirty price; separate "moex_bonds" TinkoffBroker in BrokerRouter; positive bond backtest on OFZ-PD + OFZ-PK.

**Addresses (from FEATURES.md P1):** OFZ bond autonomous trading, bond backtest with positive PnL, DV01-budget-aware sizing, yield stop-loss for bonds.

**Avoids (from PITFALLS.md):** Pitfall 1 (dirty price), Pitfall 2 (equity ATR stop on bonds), Pitfall 4 (TinkoffBroker thread-safety), Pitfall 8 (floater duration assumption at high rates).

**Gate:** BondCycleProcessor.run_cycle() executes orders in T-Invest sandbox without errors; bond backtest shows positive PnL with walk-forward validation; no equity stop-loss code path is invoked for bond orders.

**Research flag:** May need `/gsd:research-phase` for OFZ-PK effective duration calculation under 21% CBR key rate and floater reweighting logic. This is domain-specific fixed income math.

---

### Phase 5: TradingLoop Integration and Telegram Hardening

**Rationale:** With bond cycle and equity backtests independently validated, the full TradingLoop integration can proceed. This phase wires BondCycleProcessor into TradingLoop, connects MOEX holiday calendar to the bond cycle gate, adds the CBR-day extra cycle at 15:30 MSK, and hardens TelegramAlerter with rate-limited queuing before any live trades occur.

**Delivers:** BondCycleProcessor wired into TradingLoop; MOEX holiday calendar connected to bond cycle gate (not equity cycle); CBR-day extra bond cycle at 15:30 MSK; TelegramAlerter with priority message queue (circuit breaker alerts prioritized over fill alerts); daily P&L summary bug fixed (RUB MOEX equity shown correctly); coupon payment detection and on_coupon_received alert wired.

**Addresses (from FEATURES.md P1):** Telegram daily P&L bug fix, coupon Telegram alert, circuit breaker alerts.

**Avoids (from PITFALLS.md):** Pitfall 3 (MOEX holiday calendar gap), Pitfall 4 (TinkoffBroker thread-safety), Pitfall 5 (CBR announcement timing), Pitfall 10 (Telegram message storm), Pitfall 11 (macro staleness during holidays).

**Gate:** Concurrent equity + bond cycle integration test passes without gRPC errors; MOEX holiday dates return bond_cycle_skipped; macro refresh job runs on holidays; Telegram load test (20 fill alerts in 2 seconds) delivers all within 60 seconds.

**Research flag:** Standard patterns for APScheduler job isolation and Telegram rate limiting. The design doc (bond-tradingloop-integration-design.md) specifies the exact implementation.

---

### Phase 6: Sandbox Autonomous Validation

**Rationale:** End-to-end autonomous operation must be proven in T-Invest sandbox before any real money. This phase gates on Phases 1–5 being complete and validated. LayerLedger reconciliation against broker state must be implemented here — it is the last line of defense before go-live.

**Delivers:** 5+ consecutive autonomous trading days in T-Invest sandbox; LayerLedger reconciliation on startup; all circuit breakers tested and firing correctly; correct Telegram alerts verified for every trade type; no critical errors; sandbox drawdown < 5%.

**Addresses (from FEATURES.md P1):** Sandbox validation, all circuit breakers tested.

**Avoids (from PITFALLS.md):** Pitfall 6 (LayerLedger divergence), all integration gotchas verified under real network conditions.

**Gate:** 5 trading days without critical errors; drawdown < 5%; Telegram receives correct alerts for every fill, rejection, and circuit breaker event.

**Research flag:** No additional research needed. Verification and operational hardening.

---

### Phase 7: Russian News Pipeline and Event-Driven Strategy Activation

**Rationale:** News pipeline activation is deferred until the core autonomous trading (equity + bond) is stable in sandbox. This is a differentiator feature, not a table stake. It adds alpha but is not required for the system to be "autonomous" — the equity and bond strategies run without news signals.

**Delivers:** Russian news fetcher wired (RBC/Interfax/TASS RSS via httpx + fastfeedparser or T-Invest news API); NewsAnalyzer routing live Russian articles to sentiment_ru.txt prompt; event_driven strategy enabled on MOEX segments; per-source health monitoring with staleness detection; Telegram financial channel reading via Telethon (if prioritized).

**Addresses (from FEATURES.md P2):** Russian news feed integration, LLM-powered Russian news analysis, event_driven strategy, Telegram /status command.

**Avoids (from PITFALLS.md):** Pitfall 9 (Russian news source reliability — source weighting by independence: Interfax > RBC > Kommersant > TASS; staleness detection; stale cache fallback to neutral).

**Gate:** Sentiment signals influence combined MOEX signal in sandbox; source health monitoring alerts when feed silent > 30 minutes during market hours; no signal amplification from stale articles older than 4 hours.

**Research flag:** Needs `/gsd:research-phase` for Russian news RSS URL validation (MEDIUM confidence — URLs change), Telethon MTProto session management for production deployments, and T-Invest news API capability assessment. The news source integration has more unknowns than other phases.

---

### Phase Ordering Rationale

- Phases 1 → 2: Sizing correctness is a prerequisite for any meaningful backtest metric. Cannot tune MOEX strategies on data that produces wrong position sizes.
- Phases 2 → 3: MOEX equity backtest validates the engine works for Russian instruments before adding bond complexity.
- Phases 3 → 4: Bond data (NKD, coupon schedule, MacroCacheService) must be in place before bond strategies can be calibrated or their execution can be trusted.
- Phases 4 → 5: Bond cycle and equity cycle must each be individually validated before wiring them together in TradingLoop (otherwise integration bugs are indistinguishable from component bugs).
- Phases 5 → 6: All components must be integrated before sandbox testing; sandbox testing must complete before real money.
- Phase 7 is intentionally last: News pipeline adds complexity and LLM API cost; defer until core autonomous trading is proven. The event_driven strategy is disabled in the current codebase — enabling it should be the final enhancement layer.

---

### Research Flags

**Needs `/gsd:research-phase` during planning:**
- **Phase 2:** MOEX-specific ADX threshold calibration; RUONIA-correlated volatility regime differences vs US markets.
- **Phase 4:** OFZ-PK effective duration calculation under high-rate (21% CBR) environments; floater reweighting formula in EqualWeightBondSizer.
- **Phase 7:** Russian news RSS URL validation; Telethon MTProto session management for production; T-Invest news API capability (does it provide company-level news, or only market data?).

**Standard patterns — skip research-phase:**
- **Phase 1:** RUB sizing fix is a well-defined bug with a known solution (use TinkoffBroker.get_portfolio() denominated in RUB).
- **Phase 3:** CBR API and T-Invest bond methods are fully documented in the existing design doc (2026-03-10-moex-data-sources-design.md). MacroCacheService design is complete.
- **Phase 5:** TradingLoop integration pattern is documented. APScheduler job isolation and Telegram rate limiting are well-understood.
- **Phase 6:** Sandbox validation is an operational verification exercise, not a research problem.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | 4 new dependencies: all verified on PyPI with recent releases (2025–2026). aiogram 3.26, Telethon 1.42, QuantLib 1.41, fastfeedparser 0.5.9. Pyrogram abandonment confirmed. feedparser vs fastfeedparser speed claim is MEDIUM (no independent benchmark). |
| Features | HIGH | Derived from direct codebase inspection of 367 Python files + design documents + PROJECT.md known blockers. Feature dependency graph validated against existing code. MVP scope is realistic. |
| Architecture | HIGH | Derived entirely from direct codebase inspection. All layer boundaries, component responsibilities, and data flows are verified against actual code. Build order is dictated by real import dependencies. |
| Pitfalls | HIGH | Critical pitfalls (NKD sizing, ATR stop on bonds, thread-safety, LayerLedger divergence) verified against stub code in BondCycleProcessor and DV01BudgetStep. Telegram rate limit behavior verified against Telegram Bot API docs. |

**Overall confidence:** HIGH

### Gaps to Address

- **Russian news RSS URL stability (MEDIUM):** The 4 RSS endpoints are community-documented and confirmed working as of 2025 sources, but Russian news sites change paths without notice. Validate all 4 URLs at implementation time and add a liveness test to CI.
- **T-Invest news API capability (MEDIUM):** Research did not determine whether T-Invest's news API provides company-level fundamental news vs. only price/market data. Assess during Phase 7 planning — may require falling back to RSS only.
- **OFZ-PK floater effective duration formula (MEDIUM):** The research provides the direction (effective_duration = half the coupon reset period) but the exact formula for EqualWeightBondSizer reweighting at 21% CBR needs validation against a fixed-income textbook or QuantLib. Address in Phase 4 research.
- **Telethon session management in production (LOW):** Telethon requires a user account (not a bot) with an API_ID + API_HASH + session string. Production session persistence and rotation strategy needs design. Address in Phase 7 planning.
- **MOEX ISS securityevents pagination (LOW):** ISS returns max 100 rows per page with `start=` parameter. Coupon record date lookup may require pagination for bonds with long coupon histories. Validate in Phase 3.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase inspection — `src/finalayze/` (367 Python files, 2325+ tests), design docs `docs/plans/2026-03-12-bond-tradingloop-integration-design.md`, `docs/plans/2026-03-10-moex-data-sources-design.md`, `docs/design/RISK.md`, `docs/design/STRATEGIES.md`
- `.planning/PROJECT.md` — confirmed bugs and blockers
- [aiogram PyPI v3.26.0 Mar 2026](https://pypi.org/project/aiogram/) — version verification
- [Telethon PyPI v1.42.0 Nov 2025](https://pypi.org/project/Telethon/) — version verification, 11.5K GitHub stars
- [QuantLib PyPI v1.41 Jan 2026](https://pypi.org/project/QuantLib/) — bond math API verification
- [fastfeedparser PyPI v0.5.9 Mar 2026](https://pypi.org/project/fastfeedparser/) — RSS parsing capability
- [Tinkoff InvestAPI proto](https://github.com/Tinkoff/investAPI/blob/main/src/docs/contracts/instruments.proto) — GetBonds, GetBondCoupons, GetAccruedInterests methods

### Secondary (MEDIUM confidence)
- [Russian news RSS feeds (feedspot)](https://rss.feedspot.com/russian_news_rss_feeds/) — RSS URL discovery
- [Pyrogram maintenance status (Snyk)](https://snyk.io/advisor/python/pyrogram) — abandonment confirmation
- [MOEX trading calendar](https://www.moex.com/en/tradingcalendar/) — holiday calendar structure
- [Telegram Bot API rate limits](https://core.telegram.org/bots/faq) — 1 message/second per chat limit
- [MOEX OFZ settlement T+1](https://www.moex.com/n8973) — settlement convention verification
- EIDiamond invest-bot (GitHub) — MOEX bot feature comparison

### Tertiary (LOW confidence)
- Web search results for MOEX algorithmic trading 2025 — limited MOEX-specific results; inferences drawn from general algo-trading resources
- fastfeedparser 25x speed claim — stated by maintainer, no independent benchmark found

---
*Research completed: 2026-03-14*
*Ready for roadmap: yes*
