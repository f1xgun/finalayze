# Risk Management Design

## Overview

Every order must pass an 11-check pre-trade risk pipeline before execution.
Risk management operates both per-market and cross-market.

## Position Sizing: Half-Kelly

```
f* = (win_rate * avg_win/avg_loss - (1 - win_rate)) / (avg_win/avg_loss)
position = portfolio_value * (f* * 0.5)    clamped to max 20%
```

Applied per-market using that market's portfolio value and currency.

## Stop-Losses: ATR-Based

- `stop_loss = entry_price - (ATR(14) * multiplier)`
- US multiplier: 2.0, MOEX multiplier: 2.5 (higher volatility)
- Trailing stop activates at +1 ATR profit

## Portfolio Constraints

| Rule | Limit | Scope |
|------|-------|-------|
| Max open positions | 10 per market | Per-market |
| Max single position | 20% of market portfolio | Per-market |
| Max segment allocation | 40% of market portfolio | Per-market |
| Max correlated positions | 3 with r > 0.7 | Per-market |
| Min cash reserve | 20% of market portfolio | Per-market |
| Max total invested | 80% across all markets | Cross-market |

## Circuit Breakers (Per Market)

| Level | Trigger | Action | Recovery |
|-------|---------|--------|----------|
| L1 Caution | -5% daily | Reduce size 50%, raise min confidence | Auto-reset next day |
| L2 Halt | -10% daily | Stop new trades in that market | Auto after 2 profitable days |
| L3 Liquidate | -15% daily | Close ALL positions in that market | Manual reset only |

**Cross-market breaker:** Combined portfolio drops -10% in base currency -> halt ALL markets.

## Pre-Trade Check Pipeline (11 Checks)

1. Market hours (per market)
2. Symbol valid (per market)
3. Mode allows order
4. Circuit breaker (per market)
5. PDT compliance (US only)
6. Position size (Kelly + cap)
7. Portfolio rules (per market)
8. Cash sufficient (per market/currency)
9. Stop-loss set
10. No duplicate pending order
11. Cross-market exposure limit

ALL must pass. Any failure -> order rejected with full explanation.

## Status

**Phase 1:** Basic risk checks (position sizing, stop-loss, portfolio rules).
**Phase 3:** Circuit breakers, PDT tracker, currency risk.
