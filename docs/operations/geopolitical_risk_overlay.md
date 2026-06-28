# Geopolitical-risk overlay (live, advisory)

A **forward-only, advisory** risk-awareness signal built on the live news/sentiment
pipeline. It exists because the active-equity experiment
([active_equity_sleeve_experiment.md](../research/active_equity_sleeve_experiment.md))
confirmed the MOEX equity drawdown was geopolitical/sanctions-driven — but showed
the catastrophic core (the 2022-02-24 -26% invasion gap + the 27-day trading
halt) was structurally **un-catchable**, and that a news overlay **cannot be
honestly backtested** (no historical point-in-time sentiment data).

## What it is — and is NOT
- **Is:** a live signal that maps aggregate market sentiment + news intensity
  (+ optional sanctions/geopolitical event counts) into a risk **level**
  (`normal` / `elevated` / `high`) and a **recommended equity trim** toward the
  deposit/OFZ anchor, surfaced as an **alert**.
- **Is NOT:** a backtested edge, an auto-trader, or a real-money action. It
  **informs**; the operator decides. Real-money changes are a hard stop.

## Components
- `src/finalayze/analysis/geopolitical_risk.py` — pure mapping brain
  (`assess_geopolitical_risk`), pre-registered transparent bands/weights.
- `src/finalayze/orchestration/geo_risk_monitor.py` — aggregates the live
  `SentimentStore` across the MOEX bellwethers (`assess_live`).
- `scripts/geo_risk_alert.py` — CLI: print the assessment; with `--notify`, send a
  Telegram alert (via `notify_telegram.py`) when the level is ELEVATED/HIGH.

## Running
```bash
uv run python scripts/geo_risk_alert.py            # print only
uv run python scripts/geo_risk_alert.py --notify   # + Telegram on ELEVATED/HIGH
```
Cron (weekday mornings), Telegram via `FINALAYZE_TELEGRAM_BOT_TOKEN` +
`FINALAYZE_TELEGRAM_CHAT_ID`:
```
13 6 * * 1-5  cd /path/to/finalayze && uv run python scripts/geo_risk_alert.py --notify
```
Fail-soft: with no live sentiment data (DB down / empty store) it reports `normal`
and never crashes the cron.

## Scoring (transparent, not fitted)
`score = 0.5*bearish_sentiment + 0.4*sanctions/geo-event_intensity + 0.1*news_volume`
(volume only counts when sentiment is negative). Bands: `<0.33 normal` (trim 0%),
`0.33-0.66 elevated` (trim 25%), `>=0.66 high` (trim 50%). The trim is a
*recommendation* toward the deposit/OFZ anchor — applied only by the operator.

## Honest limitations
- Cannot be validated on history → treat as a judgment aid, not a proven edge.
- The biggest tail (an instant invasion gap + a market halt) is un-catchable by
  any signal; the structural defense remains **allocation** (a bounded equity
  sleeve anchored in deposit/OFZ, which the allocator already does).
- v1 uses sentiment + news volume; sanctions/geopolitical **event counts** are a
  clean enhancement once an event-type-tagged query is exposed (default 0 today).
