# Domain Pitfalls: MOEX Equity Profitability (v2.0)

**Domain:** Adding dividend gap, CBR regime overlay, and sector rotation strategies to existing MOEX multi-strategy trading system
**Researched:** 2026-03-20
**Confidence:** HIGH (codebase analysis) / MEDIUM (MOEX-specific domain knowledge from web research)

---

## Critical Pitfalls

Mistakes that cause rewrites, persistent negative Sharpe, or invalidate entire backtest results.

### Pitfall 1: Look-Ahead Bias in Dividend Gap Backtests

**What goes wrong:**
The dividend gap strategy uses `moex_dividends.yaml` (static file with 43 events) to know ex-dividend dates and amounts. In a backtest, the strategy "sees" all future dividends at bar 0. When the engine encounters bar N matching an ex-date, it generates a BUY. This is correct *if* the dividend was publicly announced before the ex-date. But on MOEX, dividend recommendations are announced by the Board of Directors 30-50 days before the shareholder meeting, and the ex-date is set at record date - 2 business days. A strategy that buys on the ex-date is fine (the date is known by then). But if gap_pct or confidence scaling uses the dividend amount, there is look-ahead risk: the *exact* dividend amount is often finalized only at the shareholder meeting, which can be 5-10 days before record date. In 2022, GAZP's board recommended 52.53 RUB but shareholders rejected it -- the YAML file shows the actual 51.03 RUB payment from October, but the strategy would have traded on the announced 52.53 RUB in June.

**Why it happens:**
The static YAML file conflates "announced" and "paid" dividends. The DividendGapStrategy.generate_signal() uses `matching_div.amount` from the YAML to compute `gap_pct = amount / pre_exdiv_close * 100`. If the YAML contains the final paid amount but the market priced in the announced amount (which may differ), the gap_pct calculation is wrong. Worse, some MOEX companies announce dividends then cancel them entirely (GAZP 2022, VTBR 2022) -- these events should NOT generate BUY signals but will if the YAML only contains successful payments.

**Consequences:**
- Backtest shows inflated win rate because only successful dividend payments are in the dataset
- Gap closure rate appears higher than reality (cancelled dividends = no gap to close, but also no loss counted)
- Strategy appears to have ~15 clean trades when real deployment would encounter 3-5 cancelled/reduced dividends per year

**Prevention:**
1. Add a `status` field to YAML entries: `paid`, `cancelled`, `reduced`. Include ALL announced dividends, not just paid ones
2. DividendGapStrategy must only use dividend data that would have been known at the ex-date: the amount from the Board recommendation (not shareholder vote), flagged with `confidence: preliminary|confirmed`
3. Include failed dividend events (GAZP 2022 June: recommended 52.53, cancelled) as entries with `status: cancelled` -- the strategy must NOT trade these in backtest
4. Expand YAML from 43 to 150+ events, including ALL board recommendations from 2020-2025, not just successful payments
5. Add test: no DividendEntry in calendar has ex_date before board_recommendation_date

**Detection:**
- Backtest shows 0 losing dividend gap trades (unrealistic -- ~20% of announced dividends are modified or cancelled)
- All gap_pct values in backtest exactly match final paid amounts (impossible to know pre-ex-date for ~30% of events)
- Win rate > 85% on dividend gap strategy (historically 65-75% when including cancellations)

**Phase to address:** Phase 1 (data preparation) -- must be resolved before any dividend gap backtesting.

---

### Pitfall 2: Survivorship Bias from 2022 Sanctions Structural Break

**What goes wrong:**
The backtest uses 2022-2025 data for walk-forward validation. February 2022 sanctions caused: MOEX closure Feb 28 - Mar 24 (25 trading days), foreign investor freeze (40% of equity ownership trapped), index drop of ~50% in Feb 2022, and permanent delisting/restructuring of several instruments (Polymetal -> Solidcore, Yandex -> YDEX). The current universe in `segments.py` includes symbols that either didn't exist pre-2022 (YDEX) or were fundamentally different entities (POLY -> new listing). Backtesting these symbols across the structural break treats pre-break and post-break as the same instrument.

**Why it happens:**
The segment definitions in `config/segments.py` use current ticker lists. Symbols like `GAZP`, `VTBR`, `SNGS` are in the universe but the PROJECT.md already identifies them as "toxic" (60% of negative PnL). The system treats 2022 as regular data, but the Feb-Mar 2022 period had: trading halt, forced margin calls, artificial circuit breakers by MOEX (20% daily limits), and CBR-mandated short-selling ban. Any strategy calibrated on this data learns "crisis = opportunity" which is survivorship bias -- you only see the survivors.

**Consequences:**
- Walk-forward folds that include Feb-Mar 2022 in training learn distorted patterns (artificial floor from MOEX circuit breakers)
- Strategies calibrated on 2022 data overestimate recovery speed (MOEX intervention prevented natural price discovery)
- Toxic symbols (GAZP, VTBR) show "mean reversion" in 2022-2023 that was actually government-supported price floors, not a repeatable pattern
- Vol estimates from 2022 data are 3-5x higher than normal, choking VolTargetStep for all subsequent bars

**Prevention:**
1. Universe surgery FIRST: remove GAZP, VTBR, SNGS, IRAO, ALRS from active segments before any backtesting
2. Purge Feb-Mar 2022 data from walk-forward training windows entirely (treat as gap, not data)
3. Use separate walk-forward configs: pre-sanctions (2020-2022-02) and post-sanctions (2022-04-2025). Never train across the break
4. For symbols that were restructured (POLY, YNDX), start data only from post-restructuring date
5. Add `exclude_periods` parameter to BacktestConfig: `[("2022-02-24", "2022-04-01")]`
6. Vol estimates must use adaptive window that excludes the crisis period or uses exponential decay

**Detection:**
- Walk-forward Sharpe swings wildly between folds (one fold includes 2022, another doesn't)
- VolTargetStep consistently scales MOEX positions to 25% (floor) because asset_vol > 0.50 from 2022 data
- Mean reversion strategies show artificially high win rate on GAZP/VTBR in 2022-2023 folds

**Phase to address:** Phase 1 (universe cleanup and data preparation) -- must be the FIRST thing done.

---

### Pitfall 3: Vol Target Calibrated for US (0.19) Destroys MOEX Position Sizes

**What goes wrong:**
The VolTargetStep in `position_sizing_pipeline.py` computes `raw_ratio = target_vol / asset_vol` and clamps to [0.25, 1.5]. The target_vol comes from strategy presets: `vol_target: 0.19` (19% annualized). MOEX blue chip volatility is typically 0.35-0.60 (35-60% annualized). The ratio becomes 0.19/0.45 = 0.42, clamped to 0.42. Combined with RegimeStep (scale 0.50 in ELEVATED), the position becomes 0.42 * 0.50 = 0.21 of base. Combined with the pipeline floor of 15%, many signals end up at exactly the 15% floor -- effectively a fixed tiny position regardless of signal strength.

**Why it happens:**
The 0.19 vol target was calibrated for US equities (SPY vol ~0.15-0.20). MOEX equities inherently have 2-3x higher vol due to: thinner liquidity, commodity/FX exposure, geopolitical risk premium, and smaller free float. The preset YAML files (`ru_blue_chips.yaml`) copy-pasted `vol_target: 0.19` from US presets without recalibrating.

**Consequences:**
- ALL MOEX equity positions are systematically undersized (21-42% of what they should be)
- The pipeline floor (15%) fires on 60-70% of MOEX trades, eliminating any signal-strength differentiation
- Position sizes too small to overcome transaction costs (MOEX commission + slippage > expected return on tiny positions)
- 104 backtest iterations ALL showing negative Sharpe partly because positions are too small to generate meaningful P&L

**Prevention:**
1. Set MOEX-specific vol_target: 0.35-0.40 for ru_blue_chips, 0.45-0.50 for ru_energy (match the market's natural vol level)
2. Alternative: disable VolTargetStep entirely for MOEX segments and rely on Kelly + RegimeStep for sizing
3. Never copy US preset parameters to MOEX presets without recalibrating -- create a separate calibration script
4. Add a test: for each segment, assert that VolTargetStep ratio > 0.50 for the median asset_vol of that segment's universe
5. Log the VolTargetStep ratio per trade in backtest -- if > 70% of trades hit the floor, the vol_target is miscalibrated

**Detection:**
- Backtest logs show `vol_target_ratio: 0.25` (clamped to floor) for > 50% of MOEX trades
- Pipeline floor firing rate > 50% for MOEX segments (should be < 10%)
- Position sizes for MOEX equities are 0.5-2% of equity (should be 5-15%)

**Phase to address:** Phase 1 (parameter recalibration) -- quick fix, high impact, do alongside universe cleanup.

---

### Pitfall 4: CBR Rate Regime Timing Error -- Using Announcement Date Instead of Market Pricing Date

**What goes wrong:**
The CBR announces key rate decisions 8 times per year. The `rub_oil_regime.py` exists but is not wired into sizing. When adding a CBR rate regime overlay, the natural approach is: "if CBR cut rate, go risk-on; if CBR hiked, go risk-off." But the market prices in CBR decisions 1-3 weeks BEFORE the announcement based on: inflation data (released monthly), CBR forward guidance (published in monetary policy reports), and OFZ yield curve movements. If the strategy waits for the actual announcement, the move has already happened. If the strategy trades on announcement, it buys at the post-announcement price (already adjusted).

**Why it happens:**
Developers treat CBR decisions as "events" (like earnings). But unlike earnings, CBR decisions are highly predictable -- the OFZ yield curve embeds the expected rate path. The "surprise" component (actual vs. expected) is what moves prices, not the absolute decision. A strategy that buys on "CBR cut" and sells on "CBR hike" is buying/selling AFTER the information is priced in.

**Consequences:**
- CBR regime overlay shows near-zero alpha in backtests (market already priced the decision)
- Worse: if the strategy buys on "cut" announcement, it buys AFTER the rally, then holds through the mean-reversion back to fair value -- negative alpha
- Look-ahead bias if the backtest uses actual CBR decisions to size positions on dates before the decision was announced

**Prevention:**
1. CBR regime must use OFZ yield curve slope (2Y-10Y spread) and RUONIA-OIS spread as *leading* indicators, not the CBR decision itself as a *lagging* indicator
2. Implement two distinct regime signals:
   - **Pre-decision positioning** (2 weeks before meeting): based on OFZ curve steepening/flattening trend
   - **Post-decision adjustment** (day after meeting): only on *surprise* component (actual vs. market-implied rate)
3. The existing `rub_oil_regime.py` (RUB-oil decorrelation) is actually a BETTER regime signal than CBR decisions -- wire it into sizing first
4. For backtesting: CBR decision dates must be from a historical calendar (available from cbr.ru), NOT embedded in the candle data. Ensure the strategy only "sees" the decision after the announcement bar
5. Add "CBR surprise" feature for ML: `actual_rate - implied_rate_from_OFZ_curve`. This has genuine predictive value

**Detection:**
- CBR regime overlay shows < 0.05 improvement in Sharpe vs. no overlay (signal already priced in)
- Positions opened on CBR decision day show negative average return over next 5 bars (bought the top)
- All CBR-gated trades cluster around 8 dates per year with no signal between meetings

**Phase to address:** Phase 3 (CBR regime integration) -- requires careful design; do NOT just gate on rate decisions.

---

### Pitfall 5: Dividend Gap Strategy Conflicts with Combiner ADX Routing

**What goes wrong:**
The `DividendGapStrategy` generates BUY signals on ex-dividend dates. These signals enter the `StrategyCombiner` where they are subject to ADX regime routing. If ADX > 35 (trend regime), only `_MOMENTUM_STRATEGIES` (momentum, dual_momentum) are allowed. Dividend gap is NOT in `_MOMENTUM_STRATEGIES` or `_MR_STRATEGIES` -- it is not classified in either pool. This means: (a) the signal may be filtered by regime routing (depending on implementation), or (b) it always passes routing but its weight gets diluted by whichever pool is active.

The deeper problem: dividend gap is an EVENT-DRIVEN strategy (triggers on calendar date, not price pattern). It should bypass ADX routing entirely. But the combiner's `generate_combined_signal()` applies routing to ALL strategies. The current code in combiner.py lines 35-36 defines:
```
_MOMENTUM_STRATEGIES = frozenset({"momentum", "dual_momentum"})
_MR_STRATEGIES = frozenset({"mean_reversion", "pairs", "ou_mean_reversion", "rsi2_connors"})
```
`dividend_gap` is in neither set, so it falls through to "ambiguous" classification and gets included regardless of ADX -- but this is ACCIDENTAL, not intentional. A future refactor that defaults unknown strategies to "blocked" would silently kill the dividend gap strategy.

**Why it happens:**
ADX routing was designed for US equity strategies (trend vs. mean-reversion classification). Event-driven strategies (dividend_gap, pead, cbr_calendar) don't fit this taxonomy. The combiner was not designed for strategies with fundamentally different signal generation logic.

**Consequences:**
- Dividend gap signals get weighted alongside trend/MR signals, diluting a high-conviction event signal with noisy technical signals
- On ex-div days, the combined confidence may fall below `min_combined_confidence` (0.15 for ru_blue_chips) because other strategies generate HOLD/SELL signals that average down the BUY
- If another strategy generates a SELL on the ex-div day (e.g., momentum sees the gap as a bearish breakdown), the signals cancel out and no trade occurs -- missing the highest-alpha signal

**Prevention:**
1. Create an `_EVENT_STRATEGIES` frozenset: `{"dividend_gap", "pead", "cbr_calendar", "event_driven"}` that bypasses ADX routing entirely
2. Event strategies should have a "standalone" mode: if dividend_gap confidence > 0.60, it should generate a trade independently of the combiner's weighted average
3. Alternative: give dividend_gap a combiner weight of 0.50+ on MOEX segments so it dominates the combined signal on ex-div days
4. Test: on a known ex-div date with ADX > 35, verify dividend_gap signal is NOT filtered
5. Test: on a known ex-div date, verify combined signal is BUY even if momentum generates SELL

**Detection:**
- Backtest shows 0 dividend_gap trades despite 43 ex-div events in calendar (signals filtered or diluted)
- Trade log shows "combined_confidence: 0.12" on ex-div dates (below 0.15 threshold)
- Dividend gap appears in strategy signal log but never in final trade list

**Phase to address:** Phase 2 (dividend gap strategy implementation) -- combiner modification required before dividend gap can work.

---

### Pitfall 6: Sector Rotation Creates Catastrophic Signal Conflicts in Multi-Strategy Combiner

**What goes wrong:**
Sector rotation generates allocation signals at the SECTOR level (overweight energy, underweight financials). But the existing combiner operates at the SYMBOL level (BUY SBER, SELL GAZP). Mixing these two levels creates contradictions: sector rotation says "overweight energy" (all energy symbols should get larger positions), but mean_reversion says "SELL ROSN" (overbought). The combiner weights these equally, and the result is incoherent -- neither a clean sector bet nor a clean technical signal.

Worse: sector rotation typically rebalances monthly. Technical strategies generate daily signals. If sector rotation weight is significant (e.g., 0.20), it produces a constant BUY/SELL bias for 30 days that drowns out daily technical signals. If sector rotation weight is small (e.g., 0.05), it has no effect and adds complexity without alpha.

**Why it happens:**
The combiner was designed for strategies that all operate at the same level (per-symbol, per-bar). Sector rotation is a different beast -- it operates at the portfolio level across symbols. Forcing it into the per-symbol combiner framework creates an impedance mismatch.

**Consequences:**
- Sector rotation "BUY energy" generates weak BUY signals for ALL energy symbols, even toxic ones (SNGS, GAZP) that should be excluded
- The constant sector bias reduces the combiner's ability to discriminate between strong and weak signals within a sector
- Monthly rebalancing creates whipsaw at transition dates: sector rotation flips from BUY to SELL on all symbols in a sector simultaneously
- Backtest optimization overfits the sector rotation timing to known macro events (2022 sanctions, 2024 rate hikes)

**Prevention:**
1. Sector rotation must NOT go through the per-symbol combiner. Implement it as a PORTFOLIO-LEVEL overlay in the sizing pipeline, not the signal pipeline
2. Correct architecture: sector rotation modifies position sizes (via a `SectorAllocationStep` in `PositionSizingPipeline`), not signal direction
3. The `SectorAllocationStep` scales positions: if sector = overweight (1.5x), the position for symbols in that sector is 1.5x base. If sector = underweight (0.5x), the position is 0.5x
4. Sector rotation rebalancing should use a 20-day rolling transition (linear ramp from old weights to new weights), not a hard flip on rebalance day
5. Never apply sector rotation to individual toxic symbols -- if a symbol is excluded from the universe, it stays excluded regardless of sector weight

**Detection:**
- Backtest shows sector rotation generating trades on excluded symbols (GAZP, VTBR)
- Trade log shows 20+ trades on the same day at month-end (sector rebalance whipsaw)
- Sector rotation + technical signals produce opposite directions on the same symbol simultaneously

**Phase to address:** Phase 3 (sector rotation implementation) -- requires architectural decision on where sector rotation lives (signal vs. sizing).

---

### Pitfall 7: Overfitting to 2022-2024 CBR Hiking Cycle

**What goes wrong:**
The backtest period (2022-2025) covers an extraordinary CBR hiking cycle: 9.5% (Feb 2022) -> 20% (Feb 2022 emergency) -> 7.5% (Sep 2022) -> 16% (Aug 2023) -> 21% (Oct 2024) -> 15.5% (Feb 2026). This is NOT a normal rate cycle -- it includes emergency wartime hikes, sanctions-driven currency crisis, and the most aggressive tightening in CBR history. Any regime strategy calibrated on this data will be overfit to extreme events that may not recur.

**Why it happens:**
The 2022-2025 window happens to be the available data with Tinkoff API. It is also one of the most volatile periods in Russian financial history. Walk-forward optimization on this data will find parameters that work well for crisis/recovery cycles but fail in "boring" markets (e.g., 2017-2019 when CBR rate was stable at 7.25-7.75%).

**Consequences:**
- CBR regime parameters tuned to detect 200bps+ emergency hikes will never fire in normal conditions (typical hikes are 25-50bps)
- RUB/oil decorrelation thresholds calibrated on 2022 sanctions shock (correlation went negative) will never trigger in normal markets (correlation typically 0.30-0.60)
- Walk-forward results look good because train/test both cover crisis periods -- but this is overfitting to crisis, not genuine alpha
- When deployed in a "normal" 2026 market (CBR cutting from 15.5% gradually), all regime signals will be permanently in NORMAL mode and add no value

**Prevention:**
1. If possible, source pre-2022 MOEX data from MOEX ISS (free, goes back to 2010) for longer backtest windows that include calm periods (2017-2019)
2. Apply Optuna overfitting guardrails (already implemented for US): DSR haircut, holdout validation, perturbation check
3. For CBR regime parameters: use academic/historical thresholds from CBR monetary policy research, not data-mined thresholds from 2022-2025
4. Test regime parameters on "synthetic calm" data: generate 2 years of flat rate environment and verify the strategy does NOT trade (no false positives)
5. Track "regime signal firing rate" -- if it fires < 5% of bars, it's a crisis-only detector (fine for risk overlay, useless for alpha generation)
6. The RUB/oil decorrelation regime (`rub_oil_regime.py`) with thresholds 0.3/0.1 was designed for normal markets -- validate these thresholds still make sense post-2022

**Detection:**
- Walk-forward Sharpe is positive ONLY on folds that include crisis/recovery periods
- Regime signal fires 0 times in 2024-2025 data (market normalized, parameters only detect 2022-level events)
- CBR regime strategy shows Sharpe > 1.0 in backtest but all alpha comes from 3-4 crisis trades

**Phase to address:** Phase 2-3 (regime strategy development) -- design the strategy for normal markets FIRST, then add crisis detection as an overlay.

---

## Moderate Pitfalls

### Pitfall 8: Dividend Calendar Sparse Data Creates Backtest Starvation

**What goes wrong:**
The current `moex_dividends.yaml` has 43 entries across 7 symbols over 3 years. That's ~6 events per symbol per year, but some symbols have only 1-2 events. For walk-forward with 12-month training windows, a training fold may contain 0-2 dividend events for a given symbol -- far too few to estimate win rate, expected return, or calibrate any parameters. The strategy appears to "work" in backtest only because the few events happen to be winners (survivorship from only including paid dividends -- see Pitfall 1).

**Prevention:**
1. Expand to 150+ events: add ALL blue chip + mid-cap dividends from 2020-2025 using Tinkoff API `get_dividends()`
2. Include symbols beyond current universe: MTSS, MGNT, CHMF, NLMK, PHOR -- all have regular dividends
3. For symbols with < 3 events in a walk-forward fold, fall back to a "MOEX average" gap closure model (pooled parameters)
4. Add interim dividends: many MOEX companies pay twice or thrice per year (TATN has 3/year, LKOH has 2/year)
5. Test: assert that every walk-forward training fold contains >= 10 dividend events across all symbols

**Detection:**
- Walk-forward folds show dividend_gap firing 0-1 times in 6-month test window
- Gap closure parameters (hold period, confidence) cannot be calibrated due to < 5 events per fold
- Strategy appears to have infinite Sharpe (2 trades, both winners) in some folds

**Phase to address:** Phase 1 (data preparation).

---

### Pitfall 9: event_driven Strategy Generates Phantom Trades in Combined Backtests

**What goes wrong:**
The `event_driven` strategy is enabled in `ru_blue_chips.yaml` with weight 0.15. In backtests, there is no real news feed, so `event_driven` generates 0 trades (expected). But its 15% weight allocation is still present in the combiner normalization. When `normalize_mode: "firing"` is used (only normalize across strategies that actually fire), event_driven's weight is excluded. This is correct. BUT: if event_driven is accidentally configured with `normalize_mode: "all"`, every combined signal is multiplied by 0.85 (because event_driven contributes 0, taking 15% of the weight). This systematically reduces all signal confidences by 15%, pushing more signals below `min_combined_confidence`.

Separately: the PROJECT.md notes "event_driven strategy fires without real news feed in backtests" as an issue. If the strategy has cached/stale sentiment data from a previous live run, it may generate phantom signals in backtest mode.

**Prevention:**
1. Explicitly disable `event_driven` in ALL segment presets used for backtesting: `event_driven: {enabled: false}` in backtest configs
2. Ensure `normalize_mode: "firing"` is the default for all MOEX presets (it is, but verify)
3. Add a guard in BacktestEngine: if `mode != "real"` and `event_driven.enabled`, log WARNING and auto-disable
4. Clear any cached sentiment state at backtest start (DividendGapStrategy already has `reset()`, event_driven needs the same)
5. Test: backtest with event_driven enabled vs. disabled produces identical results (zero contribution)

**Detection:**
- Backtest metrics differ between `event_driven.enabled: true` and `event_driven.enabled: false` even though no news feed exists
- Trade count changes when event_driven weight changes (shouldn't if it fires 0 times)
- Combiner logs show `event_driven: confidence=0.XX` in backtest (phantom signal from cached data)

**Phase to address:** Phase 1 (backtest configuration cleanup).

---

### Pitfall 10: Preferred Share Arbitrage (SBER/SBERP) Assumes Constant Spread

**What goes wrong:**
Preferred share arbitrage (buy SBERP when discount to SBER > historical mean) assumes a mean-reverting spread. But MOEX preferred shares have structurally different dynamics: (a) SBERP gets the SAME dividend as SBER but trades at 5-15% discount -- the discount IS the dividend yield premium, (b) the spread changes with interest rate environment (higher rates -> smaller discount, because prefs have higher yield), (c) during 2022 crisis, preferred shares became MORE expensive than common (SBERP > SBER briefly) due to retail buying of high-yield instruments.

**Prevention:**
1. Model the spread as a function of CBR rate and dividend yield, not a simple mean
2. Use spread z-score relative to a rolling window that EXCLUDES the 2022 crisis period
3. Entry threshold must be > 2 standard deviations (not 1.5 as in current pairs params)
4. Always check that BOTH legs (SBER and SBERP) are in the active universe and tradeable
5. Set max_hold_bars for pairs to match dividend cycle (if entering before ex-div, close before record date to avoid dividend tax complications)

**Detection:**
- Pairs strategy shows > 50% of trades entering during crisis periods (overfitting to 2022)
- Spread mean changes significantly between walk-forward folds (non-stationary)
- Win rate drops below 40% when 2022 data is excluded from training

**Phase to address:** Phase 3 (if implementing preferred arbitrage).

---

### Pitfall 11: Brent Gate for Energy Sector Uses Wrong Correlation Lag

**What goes wrong:**
MOEX energy stocks (ROSN, TATN, LKOH) correlate with Brent crude, but with a 1-3 day lag (Russian market closes before Brent final settlement). A Brent price gate that uses same-day Brent close to gate MOEX entries introduces a timing mismatch: if you gate on today's Brent, MOEX already reacted to yesterday's Brent. The signal is stale.

Worse: Brent data comes from yfinance (BZ=F), which reports in USD. The actual exposure is Brent-in-RUB (Brent * USDRUB). When RUB weakens (Brent in USD flat), Brent-in-RUB rises and MOEX energy stocks rally -- but a USD Brent gate would miss this entirely.

**Prevention:**
1. Use Brent-in-RUB (Brent * USDRUB) as the gate variable, not USD Brent
2. Apply 1-day lag: gate on yesterday's Brent-in-RUB change, not today's
3. Brent gate should be a SIZING modifier (SectorAllocationStep), not a signal filter -- don't block entries, scale positions
4. The existing `rub_oil_regime.py` already computes RUB/oil correlation -- use this as the Brent gate rather than building a separate one
5. Test: correlation between MOEX energy returns and Brent-in-RUB returns at lag=1 vs lag=0 (expect lag=1 is higher)

**Detection:**
- Brent gate correlation with energy stock returns is < 0.20 (using wrong lag or wrong currency)
- Energy sector positions are gated off on days when energy stocks actually rally (Brent-in-RUB up but USD Brent flat)

**Phase to address:** Phase 3 (sector rotation / Brent gating).

---

## Minor Pitfalls

### Pitfall 12: Dividend Gap max_hold_bars Set Too High (60 in preset, 15 in config)

**What goes wrong:**
`ru_blue_chips.yaml` sets `dividend_gap.params.max_hold_bars: 60` but `backtest/config.py` has `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"] = 15`. These are different code paths: the preset parameter controls the strategy's internal exit logic, while the config parameter controls the engine's forced exit. The engine will force-close positions at 15 bars even though the strategy expects 60 bars for gap closure. For MOEX blue chips, typical gap closure takes 20-40 bars (30-60 calendar days). The 15-bar engine limit will cut most dividend gap positions before gap closure, converting winners into losers.

**Prevention:**
1. Align `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"]` with the strategy's `max_hold_bars` parameter (60)
2. Better: dividend_gap should NOT use the engine's generic hold bar limit at all -- it has its own exit logic (gap closure OR max_hold_bars)
3. Add test: `assert DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"] >= preset_max_hold_bars`

**Phase to address:** Phase 2 (dividend gap implementation).

---

### Pitfall 13: T+1 Settlement Means Dividend Gap Entry Is One Bar Late

**What goes wrong:**
MOEX uses T+1 settlement for equities. The strategy buys on the ex-dividend date (current_candle.timestamp matches ex_date). But in MOEX terminology, the "last buy date" in the Tinkoff API is actually T-1 relative to the record date. If you buy at close on the "last buy date," you get the dividend. If you buy on the NEXT day (actual ex-date), you don't. The YAML comments confirm: "ex_date = last buy date (Tinkoff convention); actual ex-div is next trading day." But the strategy uses `div.ex_date.date() == current_date.date()` to trigger -- this triggers on the LAST BUY DATE, not the ex-date. This means the strategy buys BEFORE the gap occurs, at the pre-gap price, and the position immediately drops by the dividend amount.

**Prevention:**
1. Clarify terminology: rename YAML field from `ex_date` to `last_buy_date` to match Tinkoff convention
2. Strategy should buy ONE BAR AFTER `last_buy_date` (the actual ex-date when the gap appears)
3. OR: buy on `last_buy_date`, collect the dividend, and set gap closure target = pre-ex-date price + dividend (breakeven accounting for dividend received)
4. The current implementation uses `candles[-2].close` as pre_exdiv_close, which is correct IF triggered on the actual ex-date (bar after last_buy_date), but WRONG if triggered on last_buy_date itself

**Detection:**
- Dividend gap trades show entry price = pre-gap price (no gap at entry)
- Many trades show immediate 5-10% unrealized loss on day 1 (bought pre-gap, gap happened next bar)
- gap_pct in trade features doesn't match actual price drop

**Phase to address:** Phase 2 (dividend gap implementation) -- critical correctness issue.

---

### Pitfall 14: RegimeStep and Sector Rotation Step Compete for Position Scale

**What goes wrong:**
The sizing pipeline already has `RegimeStep` (scale by regime_scale, floor 0.15) and adding a `SectorAllocationStep` creates multiplicative scaling: final_size = base * vol_target_ratio * regime_scale * sector_scale. If regime_scale = 0.50 (ELEVATED) and sector_scale = 0.50 (underweight sector), the position is 0.25x base -- hitting the pipeline floor of 15%. Three or more multiplicative reduction steps guarantee that most MOEX positions end up at the floor, eliminating signal differentiation.

**Prevention:**
1. Cap total multiplicative reduction: `total_scale = max(product_of_all_scales, 0.25)` at the pipeline level
2. Or: make sector allocation ADDITIVE to a neutral baseline, not multiplicative: `sector_adjusted = base * (1.0 + sector_deviation)` where sector_deviation is [-0.30, +0.30]
3. Monitor floor-hit rate: if > 30% of trades hit the pipeline floor, the pipeline has too many reduction steps
4. Consider reducing the number of sizing steps for MOEX: Kelly -> MOEX-calibrated VolTarget -> HardCaps (3 steps instead of 7)

**Phase to address:** Phase 3 (sector rotation sizing integration).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 1: Universe cleanup | Removing toxic symbols changes walk-forward fold composition; old iterations incomparable | Start iteration history fresh for v2.0; do not compare v2.0 Sharpe to v1.0 iterations |
| Phase 1: Data expansion | Tinkoff API rate limits on historical dividend fetches (600 requests/minute) | Batch dividend fetches per symbol; cache to YAML; fetch once and store |
| Phase 2: Dividend gap | T+1 settlement timing confusion between last_buy_date and ex-date (Pitfall 13) | Write explicit test with known SBER 2024 dividend; verify entry is on gap day |
| Phase 2: Dividend gap | Combiner dilutes dividend signal below threshold (Pitfall 5) | Give dividend_gap standalone mode or weight >= 0.40 on MOEX segments |
| Phase 2: Dividend gap | Only 43 events -> insufficient for walk-forward calibration (Pitfall 8) | Expand to 150+ events BEFORE any backtest tuning |
| Phase 3: CBR regime | Using CBR decisions as signals instead of leading indicators (Pitfall 4) | Wire `rub_oil_regime.py` FIRST; add OFZ curve slope as leading indicator |
| Phase 3: Sector rotation | Portfolio-level strategy forced into per-symbol combiner (Pitfall 6) | Implement as SectorAllocationStep in sizing pipeline, NOT in combiner |
| Phase 3: Brent gate | Wrong currency denomination for energy correlation (Pitfall 11) | Use Brent * USDRUB, not USD Brent; test correlation at lag=0 vs lag=1 |
| Phase 3: Parameter tuning | Overfitting to 2022 crisis regime (Pitfall 7) | Source pre-2022 data from MOEX ISS; use academic thresholds for regime |
| Phase 4: ML ensemble | MOEX features trained on 2022 crisis -> model learns crisis-only patterns | Exclude Feb-Mar 2022 from training; validate on 2024-2025 calm data |
| Phase 4: ML ensemble | Too few training samples (3 years MOEX vs 10+ years US data) | Pool features across sectors; use transfer learning from US model |
| Phase 5: Portfolio assembly | OFZ + equity allocation creates currency double-counting (both RUB) | Track OFZ and equity allocations separately with independent circuit breakers |

---

## Integration Gotchas Specific to v2.0

| Integration Point | Common Mistake | Correct Approach |
|-------------------|----------------|------------------|
| DividendGapStrategy + Combiner | Dividend signals diluted by technical strategy HOLD/SELL signals | Create `_EVENT_STRATEGIES` bypass in ADX routing; or give dividend_gap standalone trade capability |
| CBR Regime + VolTargetStep | CBR regime reduces position AND vol target reduces position -> double penalty | Choose ONE of: regime overlay OR vol target for MOEX. Not both multiplicatively |
| Sector rotation + existing segments | Sector rotation implies variable universe; existing segments have fixed symbol lists | Sector rotation modifies WEIGHTS within fixed universe, does not add/remove symbols |
| rub_oil_regime.py + RegimeStep | Both produce a regime_scale; which one wins? | Compose: `effective_regime_scale = min(rub_oil_scale, vix_regime_scale)` -- use the more conservative |
| dividend_gap + ATR stop | ATR trailing stop triggers on gap day (price drops by dividend amount) | Exempt dividend_gap positions from trailing stop for first 5 bars (grace period for gap to stabilize) |
| MOEX ISS data + Tinkoff API data | Timestamps from different sources in different timezones (ISS = MSK, Tinkoff = UTC) | Normalize all timestamps to UTC at fetch time; assert timezone-aware datetimes in all data pipelines |

---

## "Looks Done But Isn't" Checklist for v2.0

- [ ] **vol_target recalibration**: YAML files still show `vol_target: 0.19` for MOEX -- verify updated to 0.35-0.40 before any backtest
- [ ] **dividend calendar completeness**: `moex_dividends.yaml` has 43 events -- verify expanded to 150+ before dividend gap backtest
- [ ] **ex-date vs. last_buy_date alignment**: `DividendGapStrategy` triggers on `ex_date` from YAML -- verify this is the actual gap day, not the day before
- [ ] **rub_oil_regime.py wired into sizing**: file exists but is not referenced in `BacktestEngine` or `PositionSizingPipeline` -- verify wired before claiming regime overlay works
- [ ] **event_driven disabled in backtest**: `ru_blue_chips.yaml` has `event_driven.enabled: true` -- verify disabled or confirm 0 phantom signals in backtest
- [ ] **DEFAULT_STRATEGY_HOLD_BARS for dividend_gap**: config says 15, strategy expects 60 -- verify aligned
- [ ] **2022 data handling**: walk-forward includes Feb-Mar 2022 -- verify either excluded or handled with purge gap
- [ ] **toxic symbols removed**: GAZP, VTBR still in `ru_blue_chips` segments.py definition -- verify removed from MOEX equity backtests
- [ ] **sector rotation architecture**: decided as combiner strategy vs. sizing step -- verify sizing step approach before implementation
- [ ] **Brent-in-RUB not USD Brent**: Brent gate uses correct currency -- verify BZ=F * USDRUB, not raw BZ=F

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Look-ahead bias in dividend data | HIGH | Rebuild entire dividend calendar with announcement dates and cancelled events; re-run all backtests; previous iterations invalidated |
| Survivorship bias from 2022 | HIGH | Source pre-2022 data from MOEX ISS; rebuild walk-forward with separate pre/post periods; all 104 iterations invalid for comparison |
| Vol target miscalibration | LOW | Update YAML `vol_target` values; re-run backtests; quick fix, no architectural change |
| CBR regime timing error | MEDIUM | Redesign regime from "react to decision" to "lead with OFZ curve"; requires new data pipeline for OFZ yields |
| Combiner dilutes dividend signals | MEDIUM | Add `_EVENT_STRATEGIES` bypass; requires combiner refactor but isolated change |
| Sector rotation in wrong layer | HIGH | If implemented in combiner, must refactor to sizing pipeline; significant code move |
| T+1 settlement date confusion | LOW | Fix date matching in DividendGapStrategy; rename YAML field; re-run backtests |
| Multiplicative sizing reduction | MEDIUM | Reduce pipeline steps for MOEX or add total-scale floor; requires sizing pipeline redesign |

---

## Sources

- Codebase analysis: `src/finalayze/strategies/dividend_gap.py`, `src/finalayze/strategies/combiner.py`, `src/finalayze/risk/position_sizing_pipeline.py`, `src/finalayze/risk/rub_oil_regime.py`, `src/finalayze/strategies/presets/ru_blue_chips.yaml`, `src/finalayze/strategies/presets/moex_dividends.yaml`, `config/segments.py`, `src/finalayze/backtest/config.py`
- PROJECT.md: v2.0 milestone context, known issues, 104 REJECT iterations, toxic symbol identification
- MOEX dividend calendar: [MOEX Dividend Yield Listing](https://www.moex.com/ru/listing/dividend-yield.aspx) | [Smart-Lab Dividend Calendar](https://smart-lab.ru/dividends/)
- CBR key rate history: [CBR Key Rate](https://cbr.ru/eng/hd_base/KeyRate/) | [CBR Monetary Policy Guidelines 2024-2026](https://www.cbr.ru/eng/about_br/publ/ondkp/on_2024_2026/)
- MOEX 2022 crisis: [Bloomberg - Russian Stocks Slump Most on Record](https://www.bloomberg.com/news/articles/2022-02-24/russian-stocks-slump-most-on-record-on-ukraine-attack-chart)
- Sector rotation pitfalls: [Common Pitfalls of Sector Rotation - GIGAPRO](https://www.gwcindia.in/gigapro/blog/common-pitfalls-of-sector-rotation-and-how-to-avoid-them/) | [Sector Rotation Myth - Molchanov 2024](https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2882)
- Dividend gap closure statistics: [T-Bank Invest - MOEX_TRADE analysis](https://www.tbank.ru/invest/social/profile/MOEX_TRADE/d9c7bf45-730e-4294-b376-557ca790fcdb/) | [Alfa Investor - Dividend Gap Closure](https://alfabank.ru/alfa-investor/t/moskovskaya-birzha-kak-bystro-zakroetsya-dividendnyy-gep/)
- MOEX market data: [Trading Economics - Russia Stock Market](https://tradingeconomics.com/russia/stock-market) | [Statista - Weekly MOEX Performance](https://www.statista.com/statistics/1254381/weekly-performance-moex/)

---
*Pitfalls research for: v2.0 MOEX Equity Profitability -- dividend gap, CBR regime, sector rotation*
*Researched: 2026-03-20*

---
---

# Production Readiness Pitfalls: v3.0 Monitoring, Go/No-Go Gates, and Gradual Rollout

**Domain:** Adding sandbox monitoring, go/no-go gates, gradual capital scaling, and production ops to existing autonomous MOEX trading system
**Researched:** 2026-03-21
**Confidence:** HIGH (confirmed against existing codebase) / MEDIUM (industry patterns from research)

---

## Critical Pitfalls

### Pitfall P1: Sandbox Gives False Confidence Because Tinkoff Fills Are Synthetic

**What goes wrong:**
The Tinkoff Invest sandbox fills market orders at the "last price" -- not the actual order book. When the system moves to live with 50-100K RUB, actual fills on illiquid MOEX mid-caps (TATN, LKOH, preferred shares) will be 0.3-1.5% worse than sandbox. If go/no-go thresholds are calibrated against sandbox slippage, the system will appear to pass and then immediately underperform on live capital. The existing `SandboxPortfolioTracker` compensates for missing dividends and coupons, but it does NOT compensate for the execution quality gap.

**Why it happens:**
Developers trust sandbox metrics at face value. The shadow ledger in `sandbox_tracker.py` correctly adjusts for coupon/dividend income but leaves fill quality uncorrected. Sandbox fill rate is always 100% (synthetic). Live fill rate on limit orders for MOEX bonds can be 60-80% in thin markets.

**How to avoid:**
- Track two separate fill-quality metrics during sandbox: (a) sandbox reported fill price and (b) independently captured MOEX ISS mid-price at order submission time. Measure the spread as "simulated slippage."
- Set go/no-go fill-rate threshold based on limit orders only (ignore market order fill rate, which is trivially 100% in both modes).
- Add an explicit "slippage budget" metric: expected slippage from `impact.py` vs. observed spread. Flag any divergence > 0.3%.
- Do NOT use sandbox P&L as the primary go/no-go gate. Use it only for signal correctness validation.

**Warning signs:**
- Sandbox Sharpe > backtest walk-forward Sharpe by more than 20%: likely means sandbox is not applying realistic execution costs.
- Zero fill rejections in sandbox over 30 days: impossible in real MOEX limit order markets.
- OFZ bond trades show 100% fill on thin coupon days in sandbox.

**Phase to address:** Sandbox monitoring phase (Phase 1). Define slippage simulation methodology before any go/no-go thresholds are finalized.

---

### Pitfall P2: Go/No-Go Thresholds Are Invented, Not Calibrated Against Backtest Distributions

**What goes wrong:**
The project defines go/no-go thresholds as: uptime >= 99%, fill rate >= 95%, DD < 5%, signal divergence < 50%. These numbers are plausible but arbitrary. If backtest walk-forward data shows the system naturally produces 6% peak drawdown during sideways MOEX markets, a 5% DD gate will permanently block go-live even when the system is behaving as designed. Alternatively, thresholds too loose let a broken system through.

**Why it happens:**
Teams set thresholds based on intuition or generic benchmarks without referencing actual backtest distribution of the same metrics. They then either never reach go-live (thresholds too tight) or pass a broken system (thresholds too loose).

**How to avoid:**
- Derive each threshold from walk-forward backtest percentiles, not from generic guides. For DD: use the 90th-percentile 30-day drawdown from `PortfolioBacktestOrchestrator` walk-forward results as the go/no-go cap, not a round number.
- For signal divergence: define "divergence" precisely -- is it the difference in signal direction (buy/sell/hold) or in position size? A 50% divergence in direction is catastrophic; 50% divergence in position size is recoverable.
- Run the go/no-go gate logic against historical backtest periods first to validate the thresholds produce the right verdict on known-good and known-bad periods.
- Separate "blocking" thresholds (any failure = no-go) from "advisory" thresholds (failures trigger review, not automatic block). Uptime and signal-direction divergence should be blocking; slippage and fill rate should be advisory.

**Warning signs:**
- First sandbox run fails go/no-go on day 1: thresholds likely too tight relative to expected variance.
- Signal divergence never exceeds 5% in sandbox: threshold is too loose to detect bugs.
- The go/no-go report is binary pass/fail with no trend data: cannot distinguish "improving" from "stuck."

**Phase to address:** Go/no-go gate design phase (Phase 2). Calibrate thresholds against existing walk-forward results before writing the gate logic.

---

### Pitfall P3: Kill Switch Has Activation Lag That Allows Additional Losses

**What goes wrong:**
The system already has a 3-level `CircuitBreaker` (CAUTION/HALTED/LIQUIDATE) and Telegram `/stop` command. Adding a "kill switch" for production sounds straightforward. The failure mode is that the kill switch (a) requires human Telegram action during a fast-moving loss, (b) has no guaranteed propagation to the `TradingLoop` between cycles, or (c) kills positions but leaves open limit orders sitting in the order book. A 30-second cycle gap on MOEX during a flash crash is not negligible.

**Why it happens:**
Kill switches are built for normal conditions. The code path from Telegram command to `TradingLoop` cancellation to open order purge to position flattening has multiple async handoffs. Each one adds latency. If the loop is mid-cycle processing bonds when the kill arrives, it may complete the current bond cycle before honouring the kill. Industry case studies confirm real-world kill switch response times of 30+ minutes without automated backing.

**How to avoid:**
- Implement the kill switch as a shared asyncio `Event` checked at the top of EVERY cycle iteration, not just at cycle entry. The existing `TradingLoop` must have a cooperative cancellation point after each instrument processed, not just at loop start.
- Kill switch must call `cancel_all_open_orders()` via `TinkoffBroker` as its FIRST action, before attempting position flattening. Open limit orders on MOEX persist until explicitly cancelled.
- Test kill switch activation while a bond cycle is mid-execution (not just during idle state).
- Log the time from kill signal received to last order cancelled and last position flat -- this is the "kill latency" metric. Target < 5 seconds.
- The existing LIQUIDATE circuit level is close to this but is triggered by drawdown, not by operator command. The kill switch is a separate code path that must be tested independently.

**Warning signs:**
- Telegram `/stop` command acknowledged but positions not flat 60 seconds later.
- Kill switch tested only in idle state, never mid-cycle.
- Open limit orders visible in Tinkoff dashboard after kill switch fires.

**Phase to address:** Production operations phase (Phase 3). Kill switch must be implemented before any live capital is deployed.

---

### Pitfall P4: Gradual Capital Scaling Creates Position-Size Discontinuities

**What goes wrong:**
Starting at 50-100K RUB with tightened risk limits (3% max position, 1% daily loss cap) and then scaling to 500K-2.5M RUB seems like a simple parameter change. In practice, the `PositionSizingPipeline` has a 15% pipeline floor. At 50K RUB with 3% max position, the minimum position is 225 RUB (fractional lots on MOEX). At 500K RUB with the same pipeline floor, minimum position is 2,250 RUB. The floor arithmetic changes the effective minimum lot count. Some MOEX instruments (OFZ bonds) have a minimum lot of 1,000 RUB nominal, so the viable instrument set changes as capital scales.

**Why it happens:**
Risk parameters are designed for a target capital level, not for a 10x capital range. Teams test at target capital and assume smaller capital "just works." The pipeline floor, half-Kelly sizing, and MOEX lot size constraints interact in non-obvious ways at small capital.

**How to avoid:**
- Define a capital-scaling ladder with explicit parameter sets for each step (50K, 150K, 500K, 2.5M RUB), not a single "tightened limits" mode.
- For each capital level, run the position sizing pipeline against the full instrument universe and verify: (a) no instrument produces a sub-lot position that would be rounded to zero, (b) pipeline floor does not produce positions that violate the per-instrument max position %, (c) OFZ bond lot sizes are respected.
- Include a scaling test in the go/no-go gate: simulate one trading cycle at the next capital tier before authorizing the move up.
- Document the minimum viable capital for each instrument in the universe. Instruments below threshold should be gated out at small capital levels.

**Warning signs:**
- Position sizes round to 0 for instruments after pipeline floor is applied at small capital.
- Bond orders rejected by Tinkoff due to minimum lot violation.
- Consecutive profitable days required for capital scaling increase are never reached because trade count is too low at small capital to generate a statistically meaningful sample.

**Phase to address:** Gradual rollout phase (Phase 4). Capital ladder must be defined and position-size arithmetic validated before first live deposit.

---

### Pitfall P5: Metric Collection Misses the Sandbox-to-Live Drift Window

**What goes wrong:**
The team collects metrics during sandbox validation: P&L, uptime, fill rate, slippage. Then they go live and shut down sandbox. The critical missing metric is the sandbox-to-live signal divergence during the transition period itself -- the window when live capital is deployed but the sandbox is still running in parallel. If the system generates a BUY signal in sandbox but a HOLD in live for the same symbol on the same bar, that divergence must be captured immediately. Without parallel operation for at least 10-20 trading days, teams have no way to distinguish "live underperformance from execution" versus "live underperformance from signal bug."

**Why it happens:**
Running sandbox and live in parallel requires careful state management (separate circuit breaker instances, separate equity tracking, separate Telegram channels). Teams shut down sandbox when they go live to reduce complexity, losing the most valuable diagnostic tool.

**How to avoid:**
- Design the monitoring system to support simultaneous sandbox + live operation with separate metric streams but a shared signal comparison report.
- Define "signal divergence" as a first-class metric: for each bar, compare sandbox signal direction with live signal direction for all symbols. Log divergences immediately, not in batch.
- Keep sandbox running for at least 20 trading days after go-live before decommissioning it.
- Signal divergence > 5% on any single day should trigger automatic investigation, not just logging.

**Warning signs:**
- Sandbox is shut down on the day of go-live.
- Signal comparison is done manually by reading logs, not via automated divergence report.
- No dedicated Telegram channel or dashboard panel for sandbox vs. live comparison.

**Phase to address:** Sandbox monitoring phase (Phase 1) -- parallel operation must be designed in from the start, not retrofitted.

---

### Pitfall P6: Anomaly Alerts Produce Alert Fatigue and Get Ignored

**What goes wrong:**
Production anomaly monitoring for a daily-bar trading system on MOEX will naturally produce alerts during: (a) MOEX market holidays, (b) T-Invest API maintenance windows (typically Sunday mornings), (c) CBR rate announcement days with elevated volatility, (d) extended trading sessions. If all anomalies produce the same-priority Telegram alert, operators tune them out within a week. The system already has a `TelegramMonitor` with a priority queue -- but the anomaly detection layer must respect that queue hierarchy, not dump all alerts at CRITICAL.

**Why it happens:**
Anomaly detection is added as a layer on top of existing monitoring without integrating into the existing priority queue. All new alerts default to the highest priority to ensure visibility. Industry data shows > 71% of anomaly alerts in production financial systems are false positives. Within days, operators are ignoring all Telegram messages.

**How to avoid:**
- Classify anomaly alerts into three tiers before implementation: (1) CRITICAL = kill switch needed (equity loss > threshold, order rejected by broker), (2) WARNING = investigation needed within 1 hour (slippage > budget, signal divergence spike), (3) INFO = note for end-of-day review (metric drifted within acceptable range).
- Integrate with the existing `TelegramMonitor` priority queue instead of creating a separate alert path.
- Add "quiet hours" suppression for known maintenance windows: T-Invest API maintenance (Sunday 02:00-06:00 MSK), MOEX pre-market hours.
- Implement alert deduplication: if the same anomaly fires every 5 minutes, collapse to one alert per hour with a count.

**Warning signs:**
- More than 5 Telegram alerts per day during normal operation.
- Same alert type firing repeatedly without operator action.
- Operators disabling Telegram notifications to reduce noise.

**Phase to address:** Production health monitoring phase (Phase 3). Alert taxonomy must be defined before any alerts are implemented.

---

### Pitfall P7: Circuit Breaker Thresholds Are Not Adjusted for Small Capital

**What goes wrong:**
The existing `CircuitBreaker` uses L1=5%, L2=10%, L3=15% daily drawdown thresholds. These were calibrated for a system with 500K+ RUB where a 5% loss is 25,000 RUB -- a meaningful signal. At 50-100K RUB with tightened risk limits, a single bad trade on a volatile MOEX instrument can produce a 5% intraday loss from pure mark-to-market noise (a 3% position in TATN dropping 1.5% intraday = 4.5% portfolio impact). The circuit breaker will fire CAUTION on normal volatility, halving position sizes and raising confidence thresholds -- which may prevent the system from generating any recovery trades.

**Why it happens:**
Circuit breaker thresholds are designed once and never revisited for different capital levels. The go-live documentation says "tightened limits" but teams focus on max-position-size and daily-loss-cap without updating the circuit breaker calibration.

**How to avoid:**
- Define circuit breaker thresholds as a function of capital tier, not fixed values. At 50K RUB: L1=3%, L2=6%, L3=10%. At 500K+ RUB: use existing L1=5%, L2=10%, L3=15%.
- Or: use the daily loss cap (1% of capital) as the L1 threshold by construction -- if the 1% daily cap is the hard stop, the circuit breaker CAUTION should trigger at 0.7% to give early warning before the cap is hit.
- Validate circuit breaker sensitivity during sandbox by deliberately injecting drawdown scenarios and verifying the system responds proportionally.

**Warning signs:**
- Circuit breaker fires CAUTION on the first day of live trading without a clear P&L cause.
- System halts trading after a single losing position on a normal MOEX day.
- CAUTION triggers every day, making the metric meaningless.

**Phase to address:** Gradual rollout phase (Phase 4). Circuit breaker thresholds must be in the capital-ladder parameter set.

---

### Pitfall P8: Health Check Does Not Cover the Full Dependency Stack

**What goes wrong:**
Teams add a `/health` endpoint that returns "OK" if the FastAPI app is responding. This passes while: (a) the T-Invest gRPC connection is silently broken, (b) the CBR XML API is returning stale rate data, (c) the Redis cache has stale IMOEX data, (d) the MOEX ISS connection is timing out and the system is trading on yesterday's index levels. The system appears healthy but is making decisions with stale inputs. Stale data does not cause exceptions -- it silently produces wrong signals.

**Why it happens:**
Health checks are added at the API layer. The data dependency stack (gRPC, REST, Redis, PostgreSQL) is only partially exercised. This is a well-known pattern: liveness check (is the process alive?) vs. readiness check (are all dependencies fresh?).

**How to avoid:**
- Implement a deep health check that verifies data freshness, not just connectivity: (a) T-Invest gRPC: last successful candle fetch < 30 minutes ago during market hours, (b) CBR rate: data age < 24 hours, (c) MOEX ISS: IMOEX data age < 60 minutes during market hours, (d) PostgreSQL: last successful write < 10 minutes during active cycle.
- Distinguish "liveness" (process is running) from "readiness" (all data sources fresh and within tolerance).
- Add a staleness alert that fires BEFORE the next trading cycle if any data source is stale. A trade executed on stale data is harder to recover from than a missed trade.
- The existing `structlog` already captures data fetch events -- add a last-fetch timestamp tracker per data source to feed the health check.

**Warning signs:**
- Health check passes but no trades executed for 2+ trading days without explanation.
- CBR rate shown in dashboard has not updated since last CBR meeting date.
- Health check response time < 10ms (it is not actually checking any downstream dependencies).

**Phase to address:** Production health monitoring phase (Phase 3). Deep health check architecture must be designed before any live deployment.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use sandbox P&L as go/no-go primary metric | Simple, single number | Overestimates real performance due to synthetic fills | Never -- always separate execution quality from signal quality |
| Single alert channel for all anomaly types | Fast to implement | Alert fatigue within 1 week | Never in production |
| Fixed circuit breaker thresholds across capital tiers | No code change needed | Fires too early at small capital, too late at large capital | Never for MOEX where lot sizes vary significantly |
| Shut down sandbox when going live | Reduces complexity | Lose parallel comparison baseline | Acceptable after 30+ trading days of stable live operation |
| Binary pass/fail go/no-go gate | Simple to communicate | Cannot distinguish "improving" from "stuck at 49% pass" | Acceptable if trend data is logged separately |
| Health check at API layer only | Fast to build | Misses data staleness, the most common silent failure mode | Only during initial sandbox phase |
| One capital level for gradual rollout | Simpler parameter set | Position arithmetic breaks at extreme ends | Never -- define at least 3 capital tiers |

---

## Integration Gotchas (v3.0 Production Ops)

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| T-Invest gRPC (sandbox) | Treat sandbox fills as equivalent to live fills | Capture MOEX ISS mid-price at order time; compute simulated slippage separately |
| T-Invest gRPC (live) | Assume market orders fill instantly at current price | Log fill_price from OrderResult and compare to signal_price; measure actual slippage per trade |
| T-Invest gRPC (live) | Forget that the old `invest-public-api.tinkoff.ru` domain is deprecated | Always pass `target="invest-public-api.tbank.ru:443"` -- already fixed in `tinkoff_data.py` but must be verified in every new broker instantiation |
| MOEX ISS REST API | Use ISS data for real-time price during trading hours | ISS has 15-minute delay; use T-Invest streaming for intraday prices, ISS only for EOD validation |
| CBR XML API | Poll CBR on every cycle | CBR rates change at scheduled meetings; cache with 6-hour TTL and alert on cache miss |
| Telegram Bot | Send all alerts via the same bot token without rate limiting | Telegram rate limits (30 messages/second) will be hit if anomaly detection fires in burst; use the existing priority queue and throttle |
| Redis cache | Assume Redis holds current data if no exception thrown | Check TTL on cached keys in health check; expired but not-yet-evicted keys can return stale data |
| PostgreSQL TimescaleDB | Use synchronous writes in the monitoring loop | Async write queue must be non-blocking; a DB write failure must not halt the trading cycle |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Collecting all metrics on every bar | Monitoring adds 200ms+ per bar to cycle time | Collect high-frequency metrics (order fill, equity) every bar; collect aggregate metrics (Sharpe, PF) daily | At > 50 symbols in universe |
| Storing raw signal data in PostgreSQL without retention policy | DB grows unbounded; query latency increases | TimescaleDB automatic retention: keep raw bars 90 days, aggregated daily metrics permanently | After 6 months of operation |
| Comparing sandbox vs. live signals in-process | Shared state risks contamination between modes | Use separate process or async task for comparison; communicate via database or queue | Immediately -- single off-by-one bug corrupts both comparisons |
| Telegram metric reports generated synchronously | Report generation blocks trading cycle | Pre-compute reports on a background schedule; Telegram send is fire-and-forget | During heavy market open volatility |
| Kill switch implemented as a database flag | DB latency (5-50ms) means kill is not truly immediate | Use asyncio Event checked in the hot loop; database flag is for persistence only | During flash crashes with rapid position changes |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Kill switch accessible without authentication | Any Telegram user can halt trading | Restrict `/stop` and kill switch Telegram commands to specific user IDs; verify for new kill switch implementation |
| Metric dashboard exposes position sizes and P&L publicly | Reveals trading strategy and sizing to market participants | Require authentication on Streamlit dashboard; add API key header to all `/api/v1/` endpoints |
| Go/no-go gate report contains raw T-Invest API responses | Leaks FIGI codes, position details | Sanitize report output; log raw API responses to secure internal log only |
| Capital scaling automation without audit trail | Cannot reconstruct why capital was increased | Every capital scaling event must write to an immutable audit log with: timestamp, trigger metric, value at trigger, operator confirmation |
| Storing T-Invest token in monitoring config | Token rotation requires config redeploy | Load token from environment variable at runtime; monitor token expiry and alert 7 days before |

---

## UX Pitfalls (Operator Experience)

| Pitfall | Operator Impact | Better Approach |
|---------|-----------------|-----------------|
| Go/no-go report is a wall of numbers | Operator cannot quickly determine pass/fail status | Use traffic-light summary at the top: GREEN/YELLOW/RED per metric, detail below |
| Sandbox dashboard shows sandbox P&L without noting it excludes slippage | Operator over-trusts sandbox results | Always label sandbox P&L as "adjusted P&L (excludes live slippage)" |
| Capital scaling requires manual code change | Friction causes delayed scaling or delayed de-scaling | Provide a CLI command: `finalayze scale-capital --tier 2` that validates arithmetic before applying |
| Kill switch is only accessible via Telegram | Kill switch unavailable if Telegram API is down | Add a secondary kill switch via the existing REST API `/api/v1/kill` protected by API key |
| No "what happened today" summary | Operator must read raw logs to understand daily activity | Automated end-of-day Telegram summary: trades executed, P&L, circuit breaker events, data quality issues |

---

## "Looks Done But Isn't" Checklist (v3.0)

- [ ] **Sandbox go/no-go gate:** Often missing calibration against backtest percentiles -- verify thresholds derived from `PortfolioBacktestOrchestrator` walk-forward data, not from generic benchmarks.
- [ ] **Kill switch:** Often missing open-order cancellation -- verify `cancel_all_open_orders()` is called before position flattening, not after.
- [ ] **Health check:** Often missing data staleness verification -- verify each data source has a last-fetch timestamp checked against a freshness threshold.
- [ ] **Circuit breaker for small capital:** Often uses production thresholds -- verify thresholds are defined per capital tier.
- [ ] **Alert taxonomy:** Often all alerts at same priority -- verify CRITICAL/WARNING/INFO tiers are defined and respected by `TelegramMonitor` priority queue.
- [ ] **Parallel sandbox + live operation:** Often sandbox is shut down at go-live -- verify sandbox continues running for 20+ trading days post go-live for signal comparison.
- [ ] **Slippage tracking:** Often relies on sandbox-reported fills -- verify independent mid-price capture at order submission time.
- [ ] **Capital scaling:** Often a single parameter change -- verify position sizing arithmetic validated for each capital tier before each step up.
- [ ] **Audit trail:** Often missing for scaling decisions -- verify every capital scaling event writes to immutable log with trigger metrics and operator confirmation.

---

## Recovery Strategies (v3.0)

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Sandbox metrics used as go/no-go baseline, live underperforms | MEDIUM | Run 10-day live parallel with sandbox; compute slippage delta; adjust thresholds; no code change needed if delta is stable |
| Go/no-go thresholds too tight, system never reaches go-live | LOW | Derive new thresholds from walk-forward backtest; document rationale; re-run gate |
| Kill switch has activation lag, additional losses incurred | HIGH | Immediately halt all trading; review order book for uncancelled orders; reconcile positions manually via Tinkoff dashboard; add sync cancellation before next deployment |
| Capital scaling too fast, drawdown exceeds L2 circuit breaker | MEDIUM | Activate HALTED level manually; reduce capital to previous tier; wait 2 profitable days for circuit breaker reset per existing `_L2_PROFITABLE_DAYS_REQUIRED` logic |
| Alert fatigue, critical alert ignored | HIGH | Review last 7 days of ignored alerts for missed signals; redesign alert taxonomy before re-enabling monitoring |
| Health check misses stale data, wrong signals executed | HIGH | Cross-check executed trades against manual MOEX ISS data for the affected period; quantify PnL impact; add data staleness check as blocking pre-cycle check |

---

## Pitfall-to-Phase Mapping (v3.0)

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Sandbox false confidence from synthetic fills (P1) | Phase 1: Sandbox monitoring | Verify slippage simulation methodology documented; mid-price capture implemented |
| Arbitrary go/no-go thresholds (P2) | Phase 2: Go/no-go gate design | Run gate logic against historical backtest periods; verify calibration rationale |
| Kill switch activation lag (P3) | Phase 3: Production ops | Test kill switch mid-cycle; measure kill latency; verify open order cancellation |
| Position-size discontinuities at capital tiers (P4) | Phase 4: Gradual rollout | Run sizing pipeline against full universe at each capital tier before deploying |
| Missing sandbox-to-live comparison window (P5) | Phase 1: Sandbox monitoring | Verify sandbox continues post go-live; signal divergence report automated |
| Alert fatigue (P6) | Phase 3: Production ops | Verify alert taxonomy; count daily alerts in first 5 days of operation |
| Circuit breaker miscalibrated for small capital (P7) | Phase 4: Gradual rollout | Validate circuit breaker sensitivity per capital tier in sandbox before live deposit |
| Health check misses data staleness (P8) | Phase 3: Production ops | Each data source has freshness check; health check fails if any source stale |

---

## Sources (v3.0)

- Existing codebase: `src/finalayze/execution/sandbox_tracker.py`, `src/finalayze/risk/circuit_breaker.py`, `src/finalayze/risk/pre_trade_check.py` -- confirms what infrastructure exists and where gaps are
- `.planning/PROJECT.md` -- v3.0 milestone requirements and system constraints
- [False Confidence in Systematic Trading (SSRN/ML Quants 2025)](https://mlquants.substack.com/p/false-confidence-in-systematic-trading) -- sandbox/live performance gap patterns
- [FIA Best Practices for Automated Trading Risk Controls (2024)](https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf) -- kill switch implementation requirements
- [Systemic failures in algorithmic trading (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8978471/) -- kill switch activation lag case study (30-minute real-world response time)
- [Alert Fatigue Mitigation in Anomaly Detection](https://ciajournal.com/index.php/jcia/article/download/28/29) -- alert taxonomy and deduplication approaches
- [Algorithmic Trading Overfitting -- PickMyTrade](https://blog.pickmytrade.trade/algorithmic-trading-overfitting-backtest-failure/) -- sandbox-to-live divergence patterns
- [QuantStart: Successful Backtesting Part II](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-II/) -- slippage and fill rate measurement methodology

---
*Pitfalls research for: v3.0 Production Monitoring, Go/No-Go Gates, and Gradual Capital Rollout*
*Researched: 2026-03-21*
