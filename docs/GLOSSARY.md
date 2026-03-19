# Glossary

Domain terminology used across the Finalayze codebase. Agents should reference this
when encountering unfamiliar terms.

## Trading & Strategy

| Term | Definition |
|------|-----------|
| **ADX routing** | Regime detection via ADX(14) indicator. ADX > 30 = trend-only pool, ADX < 20 = mean-reversion-only, 20-30 = dominant pool wins. See `strategies/adx.py`. |
| **Combiner** | `StrategyCombiner` — aggregates signals from all enabled strategies using weighted average. Each segment has its own weight preset (YAML). |
| **Confidence threshold** | Minimum combined signal confidence (0.38) to generate a trade. Below this → HOLD. |
| **Event-driven strategy** | `EventDrivenStrategy` — generates BUY/SELL from news sentiment. Weight 0.15 on all ru_* segments. |
| **Grace bar** | Backtest engine skips stop-loss check on the fill candle (entry_bars+1 == i) to avoid immediate stop-out on intraday volatility. |
| **Isolation test** | Running a single strategy in backtest without other strategies. Used to measure individual strategy contribution. |
| **Pipeline floor** | Minimum position size = 15% of base_position. Prevents cascading sizing steps from zeroing out positions. |
| **Sanctions proximity** | Per-ticker score (0-1) measuring exposure to sanctions risk. Higher scores reduce confidence on geopolitical/sanctions events. See `strategies/event_driven.py`. |
| **Walk-forward** | Backtest methodology: 12-month training window + 6-month out-of-sample test, sliding forward. |

## Risk Management

| Term | Definition |
|------|-----------|
| **Half-Kelly** | Position sizing: Kelly fraction × 0.5. Default gives ~8.33% position size. MOEX uses three-quarter Kelly (0.75). |
| **Circuit breaker** | 3-level drawdown protection. L1 (5%) = reduce size, L2 (10%) = stop new trades, L3 (15%) = close all. |
| **Pre-trade pipeline** | 11 sequential checks before order submission: cash, allocation, PDT, correlation, etc. See `risk/pre_trade_check.py`. |
| **ATR stop** | Trailing stop-loss based on ATR (Average True Range). Strategy-specific multipliers in `backtest/config.py`. |
| **DV01 budget** | Bond position sizing based on dollar value of 1bp yield change. Uses dirty price (not face value). |

## Bonds

| Term | Definition |
|------|-----------|
| **NKD** | Накопленный купонный доход (accrued coupon interest). Added to clean price to get dirty price. |
| **Dirty price** | Clean price + NKD. What you actually pay for a bond. |
| **OFZ** | Облигации федерального займа — Russian government bonds. OFZ-PD = fixed coupon, OFZ-PK = floating coupon. |
| **Carry strategy** | Hold bonds to collect coupon income. `ru_ofz_pk` segment uses this (Sharpe +1.14). |
| **YieldStop** | Bond exit logic — closes position when current YTM crosses regime-adaptive thresholds. |
| **LayerLedger** | Bond portfolio tracker with 4 layers: Core, Tactical, Opportunistic, Buffer. |

## ML Pipeline

| Term | Definition |
|------|-----------|
| **Meta-learner** | Stacking model that combines XGBoost + LightGBM + CatBoost predictions. |
| **Conformal calibration** | Non-parametric calibration of ML confidence scores using holdout set. |
| **Sequential bootstrapping** | Sample weight method (Marcos López de Prado) that reduces overlap between training labels. |
| **Trend-scanning labels** | Dynamic label generation that finds the optimal lookback for each sample's trend. |
| **Brier score** | Calibration metric for probability predictions. Lower = better. Used as quality gate. |

## News Pipeline

| Term | Definition |
|------|-----------|
| **Entity extraction** | LLM-based mapping of news article text to MOEX ticker symbols. Uses 29 company→ticker mappings. |
| **Sentiment cache** | In-memory EMA-smoothed sentiment scores per ticker. Formula: `new = 0.7 * old + 0.3 * article`. |
| **t.me/s/ scraping** | Fetching Telegram channel messages via public web preview (no API auth needed). |

## Infrastructure

| Term | Definition |
|------|-----------|
| **Work mode** | System operating mode: `debug` (no broker), `sandbox` (T-Invest sandbox), `test` (unit tests), `real` (live trading). |
| **real_confirmed** | Boolean guard in Settings. Must be `True` to start in REAL mode. Prevents accidental live deployment. |
| **FIGI** | Financial Instrument Global Identifier. Used by T-Invest API to identify instruments. |
| **Segment** | A group of tickers with shared strategy parameters. E.g., `ru_blue_chips`, `us_tech`. Defined in `config/segments.py`. |

## GSD Workflow

| Term | Definition |
|------|-----------|
| **Phase** | Major work unit in a milestone (e.g., "Bond Data Pipeline"). Contains multiple plans. |
| **Plan** | Executable work package within a phase. Contains tasks, assigned to a wave. |
| **Wave** | Parallel execution group. Plans in the same wave run simultaneously. |
| **Checkpoint** | Human verification point within a plan. Blocks execution until approved. |
| **Backtest-iteration** | Mandatory skill gate: run backtest after any strategy/risk/ML change, compare metrics. |
