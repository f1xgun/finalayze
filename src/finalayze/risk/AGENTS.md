# Risk

## Purpose
Risk management: position sizing pipeline, pre-trade checks (11 checks), circuit breakers, stop losses, regime detection, drawdown monitoring, and correlation-based exposure limits.

## Layer
Layer 4 -- Strategy / Risk. Can import from layers 0-3. Never import from layers 5-6.

## Key Files
- `position_sizing_pipeline.py` -- PositionSizingPipeline: chain of steps (Kelly -> VolTarget -> Regime -> MetaLabel -> Copula -> EVT -> HardCaps). Pipeline floor = 15% of base_position.
- `position_sizer.py` -- Half-Kelly position sizing, vol-adjusted sizing, realized volatility computation
- `pre_trade_check.py` -- PreTradeChecker: 11+ risk checks (exposure, drawdown, market hours, PDT, correlation, etc.). Returns PreTradeResult with pass/fail and violation list.
- `circuit_breaker.py` -- CircuitBreaker: 3-level escalation (CAUTION >= 5%, HALTED >= 10%, LIQUIDATE >= 15% daily drawdown). Sticky escalation (no intraday de-escalation).
- `stop_loss.py` -- ATR-based trailing stop loss computation
- `chandelier_exit.py` -- Chandelier exit (ATR-based with strategy-specific multipliers)
- `kelly.py` -- RollingKelly: rolling win rate and payoff ratio tracker
- `regime.py` -- RegimeProvider: VIX/volatility-based regime detection (risk-on/risk-off)
- `drawdown_monitor.py` -- Peak-to-trough drawdown tracking
- `correlation.py` -- Cross-asset correlation computation for position limits
- `loss_limits.py` -- Daily/weekly loss limit tracking
- `garch.py` -- GARCH volatility forecasting
- `hmm_regime.py` -- Hidden Markov Model regime detection
- `evt.py` -- Extreme Value Theory tail risk estimation
- `copula.py` -- Copula-based dependency modeling for portfolio risk
- `bocpd.py` -- Bayesian Online Change Point Detection
- `layer_circuit_breaker.py` -- Per-portfolio-layer circuit breakers (bond layers)
- `turnover_budget.py` -- Transaction cost budget tracking
- `dv01_sizing.py` -- Duration-based bond position sizing
- `yield_stop.py` -- Yield-based stop for bond positions
- `rub_oil_regime.py` -- RUB/oil correlation regime (MOEX-specific)
- `commodity_currency.py` -- Commodity-currency correlation sizing
- `bond_equity_correlation.py` -- Bond-equity correlation for cross-asset hedging

## Public API
- `PositionSizingPipeline` -- `run(context: SizingContext) -> Decimal`
- `PreTradeChecker` -- `check(signal, equity, positions, ...) -> PreTradeResult`
- `CircuitBreaker` -- `update(equity) -> CircuitLevel`, `reset_daily()`, `reset_manual()`
- `compute_position_size()` -- standalone Half-Kelly function
- `compute_atr_stop_loss()` -- ATR-based stop price

## Contracts
- Input: Signal, current equity (Decimal), open positions, candle history
- Output: position size (Decimal), PreTradeResult, CircuitLevel
- Invariants: Pipeline floor prevents cascading reductions from zeroing positions. Circuit breaker levels are sticky (only escalate intraday, never de-escalate). LIQUIDATE requires manual reset. Pre-trade checks must ALL pass for order submission. MOEX gets 1.2x ATR stop uplift.

## Testing
- Test location: `tests/unit/test_risk.py`, `tests/unit/test_circuit_breaker.py`, `tests/unit/test_pre_trade_check.py`, `tests/unit/test_position_sizing_pipeline.py`
- Run: `uv run pytest tests/unit/ -k risk -v`

## Common Patterns
- Pipeline uses Protocol-based `PositionSizingStep.adjust(size, context) -> Decimal`
- All monetary calculations use `Decimal` (never float)
- PreTradeChecker returns a list of all violations (not just first failure)
- Circuit breaker tracks `_consecutive_profitable_days` for L2 recovery (requires 2 consecutive profitable days)

---

## Graph

- **Parent:** [`src/finalayze/AGENTS.md`](../AGENTS.md)
- **Agent owner:** `risk-agent` (review: `risk-officer`)
- **Layer:** 4
- **Imports from:** `core/`, `config/`, `data/`, `markets/`, `ml/` (MetaLabeler)
- **Imported by:** `backtest/`, `execution/` (pre-trade), `orchestration/`, `api/`
- **Keywords:** `Half-Kelly`, `ATR_stop`, `circuit_breaker`, `pre_trade_check`, `position_sizing_pipeline`, `chandelier_exit`, `drawdown_monitor`, `regime`, `VIX`, `correlation`, `GARCH`, `HMM`, `EVT`, `copula`, `BOCPD`, `DV01`, `yield_stop`, `MOEX_1.2x_uplift`
