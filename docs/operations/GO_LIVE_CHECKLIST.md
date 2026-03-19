# Go-Live Checklist (MOEX Real Trading)

## Prerequisites
- [ ] Phase 6 sandbox validation report passed (AUT-04)
- [ ] All Phase 7 tests green (`uv run pytest tests/ -x`)
- [ ] Backtest-iteration on all ru_* segments shows positive WF Sharpe

## Environment Configuration
- [ ] Set `FINALAYZE_MODE=real` in `.env`
- [ ] Set `FINALAYZE_REAL_CONFIRMED=true` in `.env` (without this, system refuses to start)
- [ ] Set `FINALAYZE_TINKOFF_TOKEN` to real account token (not sandbox)
- [ ] Set `FINALAYZE_LLM_API_KEY` for OpenRouter entity extraction
- [ ] Set `FINALAYZE_TELEGRAM_BOT_TOKEN` and `FINALAYZE_TELEGRAM_CHAT_ID` for alerts
- [ ] Set `FINALAYZE_TELEGRAM_CHANNELS` with target channel list for news reading
- [ ] Verify starting capital: 500K RUB in T-Invest account

## Safety Verification
- [ ] Circuit breaker thresholds: L1=5%, L2=10%, L3=15% (same as sandbox)
- [ ] Risk limits unchanged from sandbox-validated configuration
- [ ] Telegram /stop command tested in sandbox mode first
- [ ] All ru_* preset weights sum to 1.00

## Launch Procedure
1. Start system: `uv run python -m finalayze`
2. Verify startup log shows "mode=real, real_confirmed=true"
3. Monitor first news cycle via Telegram alerts
4. Monitor first strategy cycle -- confirm order submission logged
5. Check T-Invest dashboard for executed orders

## Emergency Procedures
- **Telegram /stop**: Halts all trading cycles immediately
- **Circuit breaker**: Auto-triggers at L1/L2/L3 drawdown thresholds
- **Manual kill**: `docker stop finalayze` or Ctrl+C
- **Rollback**: Set `FINALAYZE_MODE=sandbox` and restart
