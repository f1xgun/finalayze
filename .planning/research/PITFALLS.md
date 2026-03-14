# Pitfalls Research

**Domain:** MOEX autonomous bond/coupon trading with LLM news and Telegram alerting
**Researched:** 2026-03-14
**Confidence:** HIGH (codebase analysis + verified domain knowledge)

---

## Critical Pitfalls

### Pitfall 1: Sizing Against Clean Price Instead of Dirty Price (NKD)

**What goes wrong:**
Bond position sizing computes how many bonds to buy using clean price (% of face value) but the actual cash deducted at settlement is the dirty price — clean price + NKD (накопленный купонный доход). A system that only checks `clean_price * quantity <= available_cash` will approve orders that actually overdraw the account by the NKD amount. For OFZ bonds mid-coupon period, NKD can be 3–5% of face value. At scale (buying 100+ bonds), this causes cash balance to go negative in the broker account.

**Why it happens:**
Market data and quoting APIs report clean price as the "price." Developers see the price field and use it directly. The NKD is a separate field that requires a separate API call (`get_accrued_interests`) or computation via `bond_math.nkd()`. The existing `DV01BudgetStep.compute_position_size()` uses `face_value` as the cash proxy — not dirty price — making this a latent bug in the sizing pipeline.

**How to avoid:**
Cash sufficiency check in bond pre-trade validation MUST use `dirty_price = (clean_price_pct / 100) * face_value + nkd_per_bond`. The `_validate_orders()` method in `BondCycleProcessor` already has this in its spec ("Cash sufficiency against dirty price"), but the actual sizing step in `DV01BudgetStep` uses `face_value` as the cash estimate — fix the sizer to accept `dirty_price_per_bond` and use that for the position cap, not face value.

**Warning signs:**
- Sandbox broker reports insufficient funds errors after position sizing passed pre-trade check
- Cash balance after bond buy is less than NKD × quantity bought
- Telegram alerts for rejected orders with "insufficient cash" on bonds that should fit in budget

**Phase to address:**
Bond execution wiring phase (the phase that completes `_size_and_execute()` in `BondCycleProcessor`).

---

### Pitfall 2: Applying Equity Stop-Loss ATR Logic to Bonds

**What goes wrong:**
The existing equity stop-loss pipeline (`risk/stop_loss.py`, `risk/chandelier_exit.py`) uses ATR-based trailing stops calibrated for equity volatility (1.5–2.5× ATR). OFZ bonds trade in price-percentage terms (e.g., 92.50 to 93.10), have very different volatility characteristics, and are affected by yield movements (interest rate sensitivity) rather than momentum. Applying ATR equity stops to bonds will either stop out on normal daily price noise (too tight) or never trigger (too wide). The existing `yield_stop.py` is the correct instrument but `_process_yield_stops()` currently returns 0 (stub).

**Why it happens:**
The existing equity pipeline is mature and the path of least resistance is to reuse it for bonds. Bond strategies are added to the same `Signal` schema and the same execution path, making it tempting to run them through the same risk pipeline.

**How to avoid:**
Keep bond and equity risk pipelines strictly separate. Bond exits must use `YieldStop` (threshold in basis points above entry YTM), not ATR. The `BondCycleProcessor._process_yield_stops()` stub must be completed before any live bond trading. Never pass bond orders through `PositionSizingPipeline` (equity-specific). The `DV01BudgetStep` and `EqualWeightBondSizer` are the correct bond sizing tools.

**Warning signs:**
- Bond positions getting stopped on 0.1–0.3% intraday moves (ATR too tight for bonds)
- No bond positions ever exiting despite yield rising substantially above entry (ATR too wide)
- `yield_stop` counter in logs always showing 0 after the first week of live trading

**Phase to address:**
Bond cycle completion phase (implementing full `_process_yield_stops()` and `_size_and_execute()`).

---

### Pitfall 3: Ignoring MOEX Holiday Calendar in Bond Cycle Scheduler

**What goes wrong:**
The bond cycle scheduler triggers at 10:30 MSK daily (`CronTrigger`). MOEX has 14–20 non-trading days per year (official public holidays plus official non-business days where bond market is still open for some instruments but not others). The current `MarketSchedule` only skips weekends. If the bond cycle runs on a MOEX holiday, it fetches zero candles for all bonds (empty responses from T-Invest API), generates no signals, and logs confusing "no candles" warnings — or worse, proceeds with stale data from the cache if CachingFetcher is used.

**Why it happens:**
The existing `MarketSchedule` class in `markets/schedule.py` only handles weekends (`weekday() >= 5`). Russian holiday calendar has irregular structure: some years have bridge holidays, some officially non-business days still trade, and the list changes annually. The MOEX design doc (`2026-03-10-moex-data-sources-design.md`) lists `config/moex_calendar.py` as the solution but it needs to be connected to the bond cycle scheduler.

**How to avoid:**
The bond cycle's `_bond_cycle()` method must check `is_moex_trading_day(today)` before proceeding. The `moex_calendar.py` static holiday list must be maintained and checked annually. Always have a fallback: if candle fetch returns empty for all bonds in a cycle, log `bond_cycle_skipped reason=holiday` — never proceed with zero-candle data.

**Warning signs:**
- `bond_candle_fetch_failed` warnings for ALL bonds simultaneously on specific dates
- `bond_cycle_complete total_signals=0 total_executed=0` on dates that should be national holidays
- Yield stops not evaluating on days adjacent to holidays when positions need monitoring

**Phase to address:**
Bond cycle integration phase (TradingLoop scheduler setup with `MacroCacheService`).

---

### Pitfall 4: TinkoffBroker Thread-Safety Across Equity and Bond Cycles

**What goes wrong:**
APScheduler `BackgroundScheduler` runs each job in a separate thread. The equity `_strategy_cycle()` and bond `_bond_cycle()` can execute concurrently. If both share the same `TinkoffBroker` instance, they share the same `AsyncClient` which is not thread-safe across different `asyncio.run()` calls — each `asyncio.run()` creates a new event loop, but the shared `AsyncClient`'s gRPC channel may have state from the previous loop. This causes intermittent gRPC errors, partial order submissions, or silent failures.

**Why it happens:**
The bootstrap script creates one `TinkoffBroker` instance and routes all MOEX traffic through it via `BrokerRouter`. The concurrency issue only manifests under load — in sandbox with slow cycles it appears to work fine. The design doc (`bond-tradingloop-integration-design.md` §3.6) already identified this and mandates a separate `TinkoffBroker` instance for bonds keyed as `"moex_bonds"`. The pitfall is forgetting to implement this separation.

**How to avoid:**
Register `tinkoff_broker_bonds = TinkoffBroker(...)` as a separate instance in `BrokerRouter` under key `"moex_bonds"`. Validate in integration tests that concurrent equity + bond cycles do not share gRPC channel state (`tests/integration/test_concurrent_cycles.py`). Never reuse an `AsyncClient` instance across threads.

**Warning signs:**
- Intermittent gRPC errors only during overlapping equity + bond cycle times
- "cannot schedule new futures after interpreter shutdown" Python errors
- Orders disappearing without error logs (silently dropped due to event loop conflict)

**Phase to address:**
Bond cycle integration phase (bootstrap wiring of `BondCycleProcessor`).

---

### Pitfall 5: CBR Key Rate Announcement Timing — Spread Spike Window

**What goes wrong:**
CBR announces key rate decisions at 13:30 MSK. OFZ bid-ask spreads spike 30–50 bps immediately after the announcement as market makers reprice. If the bond cycle runs at 13:30–14:30 MSK on CBR meeting days, it will buy/sell at inflated spreads, turning what should be a 5–10 bps cost into a 30–50 bps cost. The `CBREventStrategy` is designed to trade around CBR meetings but must avoid execution in this window.

**Why it happens:**
The CBR meeting day extra bond cycle is a useful feature, but its timing is critical. The design doc correctly identifies 15:30 MSK as the safe window (after the 15:00 press conference). Incorrectly implementing it at 13:30 or 14:00 MSK — or not gating execution at all on announcement day — wastes the alpha from the strategy on transaction costs.

**How to avoid:**
Extra CBR-day bond cycle must run at 15:30 MSK minimum, after the press conference with forward guidance. The `_cbr_day_refresh()` implementation in the design doc is correct. Validate by logging `fill_price` and comparing to last mid-price before execution — if spread exceeds 20 bps, flag a warning. Never schedule bond execution between 13:30 and 15:00 MSK on CBR meeting days.

**Warning signs:**
- Bond fills on CBR meeting days showing significantly worse prices than the pre-announcement mid
- `CBREventStrategy` generating negative alpha despite correct signal direction
- `MacroSnapshot.last_cbr_decision` showing "unexpected" decision when fill was taken before press conference

**Phase to address:**
TradingLoop bond cycle scheduling phase (CBR meeting detection + `_cbr_day_refresh()`).

---

### Pitfall 6: LayerLedger State Diverging from Actual Broker Portfolio

**What goes wrong:**
`LayerLedger` is in-memory and tracks per-layer positions, cash, and drawdown. If the process crashes and restarts, the ledger is reset to initial state. If an order executes but the result is not recorded (exception during `_log_signals`, network error during ledger update), the ledger shows no position while the broker holds one. This "ghost position" causes the bond cycle to repeatedly try to open the same position it already holds, accumulating unintended exposure up to the pre-trade position count limit.

**Why it happens:**
The design doc explicitly defers ledger persistence to a future phase ("In-memory for sandbox. Daily reconciliation against broker portfolio state at cycle start"). This is acceptable for sandbox validation, but the reconciliation step is critical before any real-money deployment. Without it, a single process restart doubles or triples bond exposure.

**How to avoid:**
Before every bond cycle run (or at minimum on process startup), reconcile `LayerLedger.positions` against `TinkoffBroker.get_positions()`. Any discrepancy — broker holds FIGI with no ledger entry — must be registered as an existing position in the ledger at conservative cost basis. Log discrepancies as `WARNING` alerts in Telegram. Do not proceed with new BUY signals until reconciliation is complete.

**Warning signs:**
- Bond position count in ledger is 0 but `TinkoffBroker.get_positions()` returns non-empty after restart
- Duplicate FIGI entries in broker portfolio for the same bond issue
- Aggregate DV01 budget exhausted faster than expected

**Phase to address:**
Sandbox validation phase (before real-money deployment). Must be resolved before go-live.

---

### Pitfall 7: RUB Position Sizing Using USD-Derived Kelly/Equity Figures

**What goes wrong:**
The existing equity position sizing pipeline uses portfolio equity in USD (Alpaca broker). The `PreTradeChecker` and `PositionSizingPipeline` were built with USD denominations. When applied to MOEX in RUB, if the system uses the total portfolio equity (USD + RUB converted) without careful currency separation, the MOEX position sizes become either absurdly small (if RUB equity is divided by USD-sized Kelly fractions) or dangerously large (if a USD target size is applied to RUB notional). The PROJECT.md notes this was a confirmed bug: "Position sizing in USD instead of RUB (MOEX positions ~0.02% instead of 15%)".

**Why it happens:**
`PreTradeChecker.check()` takes `portfolio_equity: Decimal` without a currency argument. The caller must ensure this is denominated correctly. In a multi-currency system (USD Alpaca + RUB Tinkoff), it is easy to pass total USD-converted equity instead of the per-market RUB-denominated equity. The circuit breaker uses baseline equity in whatever units it received at initialization.

**How to avoid:**
Every MOEX risk check must use RUB-denominated equity from `TinkoffBroker.get_portfolio()`. Never convert MOEX equity to USD for risk calculations — keep markets isolated. The `LayerLedger` for bonds must be initialized in RUB and use the `settings.bond_capital` (RUB) figure directly. Add a `currency` field assertion to position sizing inputs in tests.

**Warning signs:**
- MOEX bond positions showing 0.01–0.1% of equity (too small)
- DV01 budget exhausted with only 1–2 bonds (too large — USD equity treated as RUB)
- `PreTradeChecker` passing on bond orders whose notional exceeds the entire MOEX account value

**Phase to address:**
MOEX position sizing fix phase (first MOEX-specific phase). Confirmed existing bug.

---

### Pitfall 8: Floating-Coupon OFZ Duration Assumption Is Wrong at High Rate Environments

**What goes wrong:**
`EqualWeightBondSizer` (for OFZ-PK floaters) assumes near-zero duration because floating coupons reset to RUONIA. This is correct in stable rate environments. However, at CBR key rate = 21% (as of March 2026), OFZ-PK bonds have meaningful duration even between resets: the reset lag (quarterly or semi-annual) means the bond underperforms during rapid CBR rate hikes. Treating all floaters as zero-duration results in overweighting them during hiking cycles — exactly when they underperform most.

**Why it happens:**
The zero-duration assumption for floaters is a textbook simplification that breaks at high-rate/high-volatility regimes. In normal conditions (2015–2019 Russia) it was acceptable. At 21% key rate with potential for further hikes or cuts, even floater price moves of 3–5% can occur between coupon resets.

**How to avoid:**
For OFZ-PK floaters, compute "effective duration" as half the coupon reset period (in years). For semiannual-reset bonds, effective duration ≈ 0.25 years. Weight the Short layer's equal-weight allocation down by `effective_duration / target_duration_bucket`. Use the Short layer exclusively for floaters in the Core layer to minimize exposure. Do not allow the Short layer to exceed 15% of total bond capital.

**Warning signs:**
- OFZ-PK prices dropping 2%+ during CBR rate hike cycles despite "near-zero duration" assumption
- Short layer showing larger drawdown than Core layer (floaters underperforming)
- `DV01` budget calculation showing budget exhausted by floaters that "shouldn't count"

**Phase to address:**
Bond strategy calibration phase (OFZ-PK strategy parameter tuning).

---

### Pitfall 9: Russian News Sources Require Dedicated Latency and Reliability Handling

**What goes wrong:**
Russian financial news sources (RBC, Interfax, TASS, Kommersant) have irregular RSS update intervals, sometimes going silent for hours during major market events (which are exactly when news-driven signals matter most). TASS is state-owned and introduces official framing/delay on government-sensitive information. RSS feeds from Russian sources frequently change format without notice (Cyrillic encoding issues, malformed XML, redirects). A news pipeline that treats these sources as always-available will silently fail to generate signals during crises.

**Why it happens:**
Developers test the news pipeline during normal market conditions when feeds are reliable. Edge cases (MOEX suspension, geopolitical events, government communication embargoes) are not tested. State news agencies (TASS) have documented lag in covering market-moving events the government wants to control.

**How to avoid:**
Implement per-source health monitoring with staleness detection: if any source has not published in N minutes (e.g., 30 minutes during market hours), log `WARNING source_stale`. Weight Interfax (commercially independent) higher than TASS for market-sensitive company news. Always have a "no news = neutral" fallback — the `EventDrivenStrategy` must degrade gracefully, not amplify signals from the last cached article. LLM calls must include publication timestamp validation (reject articles older than 4 hours for intraday signals).

**Warning signs:**
- News pipeline generating signals from articles published during the previous trading session
- All signals on the same day pointing in the same direction (TASS echo chamber during controlled events)
- `NewsAnalyzer` producing identical sentiment scores across multiple instruments on the same day

**Phase to address:**
LLM news integration phase (enabling `event_driven` strategy with Russian media sources).

---

### Pitfall 10: Telegram Bot Message Storm During Circuit Breaker Events

**What goes wrong:**
During a circuit breaker LIQUIDATE event, the system may attempt to close 10–20 positions simultaneously. Each fill triggers `on_trade_filled()` → `send_alert()`. At 20 fills in 5 seconds, the Telegram Bot API returns 429 (Too Many Requests) with a retry-after delay. The current `send_alert()` in `TelegramAlerter` is fire-and-forget with a single `timeout=10` and swallows all exceptions — meaning 80% of circuit breaker alerts are silently dropped. During exactly the moment when the operator needs maximum alerting, they receive none.

**Why it happens:**
Telegram allows ~1 message/second per chat and rejects bursts exceeding this. The current implementation creates a new `httpx.AsyncClient` per message and has no rate limiting or queuing. Under normal conditions (1 fill every few minutes) this is invisible. Under liquidation events it fails completely.

**How to avoid:**
Implement a message queue with rate limiting in `TelegramAlerter`: buffer messages in a `deque`, drain at max 1/second using a background thread or asyncio task. On 429 responses, respect `retry_after` from the response header and re-enqueue. Prioritize circuit breaker alerts over trade fills (two priority queues: HIGH and NORMAL). For LIQUIDATE events specifically, batch fills into a single message: "LIQUIDATE: closed 12 positions" instead of 12 individual messages.

**Warning signs:**
- Zero Telegram messages received during a known high-activity period
- Logs show `TelegramAlerter failed to send message` with 429 status repeatedly
- Telegram shows single-digit messages on a day when dozens of fills occurred

**Phase to address:**
Telegram alerting hardening phase (before real-money deployment).

---

### Pitfall 11: MOEX Extended Holiday Windows Break NKD-Based Calculations

**What goes wrong:**
During the Russian New Year holiday window (January 1–8), MOEX is closed for 7+ consecutive calendar days. NKD computation uses `days_since_last_coupon` — if the system computes NKD on January 9 (first trading day) using calendar days since last coupon, it correctly includes the 8 holiday days. But if the bond cycle skipped running during the holiday (correctly), and the `MacroCacheService` snapshot is 8 days stale, the `MacroSnapshot.key_rate` used for signal generation is from December. For CBR decisions announced between December 20 and January 8 (rare but possible), signals would use the old rate.

**Why it happens:**
The `MacroCacheService` refresh is triggered by the bond cycle scheduler at 10:00 MSK daily. During a 7-day MOEX holiday, the bond cycle correctly skips (no trading), but the macro refresh should still happen to stay current. The design conflates "bond trading day" with "macro data refresh day."

**How to avoid:**
Separate the macro refresh schedule from the bond cycle schedule. The `macro_refresh` job should run 7 days/week (including holidays and weekends). Only the `bond_cycle` job should be gated on trading day status. The `MacroCacheService` should expose `snapshot_age_days` and the bond cycle should refuse to trade if age exceeds 2 business days.

**Warning signs:**
- `macro_refreshed key_rate=X` in logs where X is 7+ days old on the first post-holiday trading day
- `CBREventStrategy` not detecting a key rate change that occurred during holidays
- `bond_cycle_skipped reason=no macro data` on the first post-holiday trading day

**Phase to address:**
Bond cycle integration phase (TradingLoop scheduler setup).

---

### Pitfall 12: Coupon Record Date vs. Payment Date Confusion in Ex-Coupon Logic

**What goes wrong:**
OFZ bond coupons have three dates: (1) record date (cut-off for who receives the coupon), (2) payment date (when cash is credited), and (3) ex-coupon date (day after record date when price drops by coupon amount). The current `TinkoffFetcher._fetch_bond_coupons_async()` estimates record date as `coupon_date - 2 business days` since T-Bank does not provide it directly. This estimation can be wrong around holidays (e.g., if payment day falls after a 5-day holiday, the actual record date may be 4+ business days before payment, not 2). Trading on a day that the system thinks is pre-record but is actually post-record causes purchasing NKD that will not be received.

**Why it happens:**
T-Bank's `get_bond_coupons` API returns `coupon_date` (payment date) and `pay_one_bond` (amount). The record date requires separate lookup via MOEX ISS (`securityevents` endpoint) or manual calendar calculation. The 2-business-day estimate is the standard but breaks around MOEX holiday sequences.

**How to avoid:**
Cross-reference coupon record dates from MOEX ISS `securityevents` API (`https://iss.moex.com/iss/securities/{ticker}/events.json`) rather than estimating. If ISS lookup fails, use a conservative 3-business-day buffer instead of 2. Block bond purchases in the 3 business days before estimated payment date (not 2), and never purchase on payment day itself (NKD resets to zero, price impact).

**Warning signs:**
- Bond purchases immediately followed by NKD dropping to near zero the next day (bought just past record date)
- Coupon receipt alerts (`on_coupon_received`) not firing for bonds that were held through the payment date
- Position shows clean price jumping up then back down around coupon dates

**Phase to address:**
Bond data pipeline phase (before bond cycle execution is enabled).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| In-memory `LayerLedger` without persistence | Simpler sandbox setup | Ghost positions after crashes; cannot safely scale to real money | Sandbox validation only — never in production |
| Estimating record date as `coupon_date - 2d` | Avoids MOEX ISS `securityevents` lookup | Incorrect ex-coupon blocking around holidays | MVP with corporate bond exclusion; fix before adding non-OFZ bonds |
| Static `moex_calendar.py` holiday list | No external API dependency | Outdated after each year; requires annual manual update | Acceptable if calendar has automated test that fails when >1 year old |
| Single T-Invest API token for sandbox + live | Simpler config management | Accidental real orders from sandbox code path | Never — use separate sandbox/live tokens always |
| Fire-and-forget Telegram alerts | Simple implementation | Alerts lost during burst events (circuit breaker, liquidation) | Never in production — implement rate-limited queue before go-live |
| Floating coupon zero-duration assumption | Simpler DV01 calculation | Overweights floaters in high-rate/hiking environments | Only when key rate is stable and < 10%; not at 21% |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| T-Invest API gRPC | Using `AsyncSandboxClient` — forcibly overrides target to old `tinkoff.ru` domain | Use `AsyncClient` with `target="sandbox-invest-public-api.tbank.ru:443"` explicitly |
| T-Invest API gRPC | Sharing one `AsyncClient` across APScheduler threads | Create separate `TinkoffBroker` instance for bond cycle (`"moex_bonds"` key in `BrokerRouter`) |
| T-Invest API gRPC | Not setting `GRPC_DNS_RESOLVER=native` before importing grpc | Set env var before any `from t_tech.invest import ...` — already done in `tinkoff_data.py` |
| T-Invest API bond data | Using `get_candles` FIGI for a delisted/restructured bond | Check `bond.trading_status` field in `bond_by` response before adding to registry |
| CBR XML API | Parsing CBR XML without lxml (using stdlib `xml.etree`) | CBR XML uses non-standard encoding declarations; requires lxml with `recover=True` |
| MOEX ISS REST | Not handling 100-row pagination | ISS returns max 100 rows per page; must paginate with `start=` parameter |
| MOEX ISS REST | Assuming ISS timestamps are UTC | ISS returns MSK timestamps; must convert via `ZoneInfo("Europe/Moscow")` to UTC |
| Telegram Bot API | Sending messages without retry on 429 | Respect `retry_after` header; implement exponential backoff queue |
| Telegram Bot API | Long messages exceeding 4096 character limit | All alert methods must truncate or split messages over 4096 chars |
| Russian news RSS | Treating all sources as equally reliable | Weight by independence score: Interfax > RBC > Kommersant > TASS |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Creating new `httpx.AsyncClient` per Telegram message | Each `send_alert()` opens TCP connection + TLS handshake (~100ms) | Reuse client via connection pool; use fire-and-forget task queue | At > 5 messages/minute (circuit breaker events) |
| Creating new gRPC channel per candle fetch | Each `asyncio.run()` in `TinkoffFetcher` creates/destroys channel | Acceptable for low-frequency bond cycle (daily); would be a bottleneck at hourly equity cycle | Bond cycle: fine. If bond cycle ever goes sub-hourly, switch to persistent channel |
| Fetching 90-day candles for each bond separately | N sequential gRPC calls for N bonds in bond cycle | Low N (10–15 OFZ) keeps this acceptable. Add rate limiter between calls | Breaks if bond universe expands to 50+ instruments |
| LLM API calls for every news article | Each Claude API call = 0.5–2s latency | Batch news items; cache embeddings; skip articles older than 4h | At > 20 news items per cycle (high-activity market days) |
| In-process CachingFetcher for ISS/CBR data | Cache miss requires synchronous HTTP call in trading loop thread | Pre-warm cache on startup; use `GenericFileCache` with TTL | First trading day after restart when cache is cold |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Same T-Invest token for sandbox and live environments | Sandbox code triggers real orders | Separate `FINALAYZE_TINKOFF_TOKEN` (sandbox) and `FINALAYZE_TINKOFF_TOKEN_LIVE` (production) env vars with assertion that live token never used in sandbox mode |
| Logging T-Invest API token in error messages | Token exposed in log aggregation systems | `structlog` must never log `settings.tinkoff_token`; use `token[:4]+"***"` in debug logs |
| Telegram bot token in structured logs | Bot hijacking if logs are compromised | Same as above — never log raw tokens |
| No validation that CBR XML is from `www.cbr.ru` | DNS spoofing could inject fake key rate data | Validate SSL certificate against CBR's known cert fingerprint; flag if key_rate changes > 200bps between fetches |
| LLM prompt injection via news content | Malicious actor publishes news with injected instructions | Wrap all news content in explicit delimiters in LLM prompts; validate output schema before using sentiment score |

---

## "Looks Done But Isn't" Checklist

- [ ] **YieldStop**: `_process_yield_stops()` returns 0 (stub) — verify it actually evaluates positions before calling done
- [ ] **Bond execution**: `_size_and_execute()` returns `False` (stub) — verify actual broker order submission before calling done
- [ ] **NKD-aware sizing**: `DV01BudgetStep` uses `face_value` not `dirty_price` — verify pre-trade cash check uses dirty price
- [ ] **Telegram rate limiting**: `send_alert()` is fire-and-forget with no queue — verify burst scenario doesn't lose circuit breaker alerts
- [ ] **LayerLedger reconciliation**: in-memory ledger with no crash recovery — verify reconciliation against `TinkoffBroker.get_positions()` on startup
- [ ] **Holiday calendar wiring**: `moex_calendar.py` exists in design but must be connected to `_bond_cycle()` gate check
- [ ] **MOEX equity sizing in RUB**: confirmed existing bug — verify MOEX positions are 10–20% of MOEX equity, not 0.02%
- [ ] **Separate bond TinkoffBroker instance**: design requires `"moex_bonds"` key in `BrokerRouter` — verify it is wired in `run_sandbox.py`
- [ ] **CBR meeting day timing**: extra bond cycle must trigger at 15:30 MSK, not 13:30 — verify `_cbr_day_refresh` schedule
- [ ] **Coupon record date buffering**: 2-day estimate may be wrong around holidays — verify 3-day conservative buffer is used

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| NKD-aware sizing bug triggers overdraft | HIGH | Manual cancel of in-flight orders; manual cash top-up; fix sizer; validate all positions; restart bond cycle |
| ATR stop closes all bond positions incorrectly | HIGH | Manually re-enter positions at market; disable equity stop pipeline for bonds; implement `YieldStop`; run backtest to validate |
| LayerLedger diverges from broker portfolio | MEDIUM | `TinkoffBroker.get_positions()` audit; manual ledger reset; reconcile quantities; resume with fresh cycle |
| Telegram storm drops all circuit breaker alerts | MEDIUM | Check logs for `circuit_breaker_escalated` events; manually close positions if needed; implement queue before restarting |
| CBR meeting buy at 13:30 spread spike | LOW | Accept the bad fill; log as execution quality event; shift `_cbr_day_refresh` to 15:30 MSK |
| Stale macro data after holiday | LOW | Manual `macro_cache.refresh()` call via API; verify `snapshot.key_rate` matches CBR website; resume cycle |
| Duplicate bond positions from ledger reset | HIGH | Check broker portfolio for double positions; submit SELL for duplicate quantity; reset ledger to actual broker state |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| NKD-aware sizing (dirty price) | Bond cycle execution completion | Pre-trade check test: order with exact cash = clean_price * qty fails if NKD would overdraw |
| Equity stop-loss applied to bonds | Bond cycle execution completion | Assert no `chandelier_exit` or `stop_loss` code path is called for bond orders |
| MOEX holiday calendar gap | Bond cycle TradingLoop integration | Bond cycle returns `skipped=True` for every confirmed MOEX holiday in 2025 |
| TinkoffBroker thread-safety | Bond cycle TradingLoop integration | Integration test: concurrent equity + bond cycles complete without gRPC errors |
| CBR announcement timing | Bond cycle TradingLoop scheduling | Unit test: `_cbr_day_refresh` schedule is at 15:30 MSK, not earlier |
| LayerLedger divergence recovery | Sandbox validation phase | Process restart test: ledger state matches broker portfolio after restart |
| RUB position sizing bug | First MOEX-specific backtest phase | Assert MOEX position size is 10–20% of MOEX RUB equity in backtest |
| Floater duration assumption at high rates | Bond strategy calibration | Stress test: 300bps rate shock shows Short layer drawdown < 3% with corrected duration |
| Russian news reliability | LLM news integration phase | Source health monitoring alerts when feed silent > 30 minutes during market hours |
| Telegram burst failure | Telegram hardening phase (pre-go-live) | Load test: 20 fill alerts in 2 seconds — verify all delivered within 60 seconds |
| Extended holiday macro staleness | Bond cycle TradingLoop integration | Test: macro_refresh job runs on MOEX holiday; bond_cycle job skips on same day |
| Coupon record date estimation | Bond data pipeline phase | Test: ex-coupon gate uses 3-day buffer; validate against MOEX ISS `securityevents` for known OFZ coupon dates |

---

## Sources

- Codebase analysis: `src/finalayze/core/bond_cycle.py`, `src/finalayze/risk/dv01_sizing.py`, `src/finalayze/core/alerts.py`, `src/finalayze/data/fetchers/tinkoff_data.py`, `src/finalayze/execution/tinkoff_broker.py`, `src/finalayze/markets/schedule.py`
- Design documents: `docs/plans/2026-03-12-bond-tradingloop-integration-design.md`, `docs/plans/2026-03-10-moex-data-sources-design.md`
- MOEX settlement: [MOEX T+1 settlement for OFZs](https://www.moex.com/n8973)
- MOEX holiday calendar: [MOEX Trading Calendar](https://www.moex.com/en/tradingcalendar/) | [2025 trading schedule](https://www.moex.com/n73702)
- Telegram rate limits: [python-telegram-bot wiki](https://github.com/python-telegram-bot/python-telegram-bot/wiki/Avoiding-flood-limits) | [Telegram Bot FAQ](https://core.telegram.org/bots/faq)
- T-Invest API: [Tinkoff invest-python GitHub](https://github.com/Tinkoff/invest-python) | [investAPI issues: stream limit](https://github.com/Tinkoff/investAPI/issues/64)
- Russian news bias: [Interfax bias rating](https://mediabiasfactcheck.com/interfax-russia-bias/) | [TASS bias rating](https://mediabiasfactcheck.com/russian-news-agency-tass/)
- MOEX bond clean/dirty price: Wikipedia dirty price, BTRM Working Paper #14 on bond market clean price conventions
- Known project bugs: `docs/quality/GAPS.md`, `.planning/PROJECT.md` §Known Blockers
- MOEX bond market microstructure: [MOEX bonds market page](https://www.moex.com/s2264)

---
*Pitfalls research for: MOEX autonomous bond/coupon trading with LLM news and Telegram alerting*
*Researched: 2026-03-14*
