# Stack Research

**Domain:** Autonomous MOEX trading — bonds/coupons, LLM news analysis, Telegram integration
**Researched:** 2026-03-14
**Confidence:** MEDIUM-HIGH (T-Invest bond API: HIGH via existing SDK; QuantLib: HIGH via PyPI; Telegram libs: HIGH via PyPI; RSS: HIGH via PyPI; news RSS URLs: MEDIUM — verified existence, not freshness)

## Context: What Already Exists

The existing codebase (Python 3.12, async-first, uv) already provides:
- `t-tech-investments` gRPC SDK — candles, instruments, dividends wired (`tinkoff_data.py`)
- `anthropic` + `openai` clients — LLM analysis pipeline wired (`analysis/llm_client.py`)
- `httpx` — async HTTP client already in stack
- `apscheduler>=3.10.4` — scheduler already in stack
- `structlog` — logging already wired
- `fastapi` + Prometheus — monitoring already wired

This research covers only what is **missing** for the MOEX milestone.

---

## Recommended Stack

### New Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| QuantLib | `>=1.41` | Bond math: YTM, modified duration, convexity, NKD accrued interest | Industry-standard quant finance library. No MOEX-specific alternative exists. Provides `BondFunctions.duration()`, `BondFunctions.convexity()`, `FixedRateBond` for OFZ, floating-rate support for OFZ-PK/OFZ-IN. Precompiled wheels on PyPI — no C++ build needed. Latest: 1.41 (Jan 2026). |
| aiogram | `>=3.26` | Outbound Telegram bot — trade alerts, P&L reports, circuit breaker events | Fully async (asyncio + aiohttp). Only async-native Bot API framework. Matches project's async-first architecture. Supports Finite State Machine and webhook/polling modes. Bot API 9.5 support. Latest: 3.26.0 (Mar 2026). python-telegram-bot v20+ is also async but aiogram has better typing and lighter footprint. |
| Telethon | `>=1.42` | Inbound Telegram channel reading — scrape financial channels as MTProto user client | Only mature, actively maintained MTProto library. Enables reading public/private Telegram channels as a user account (not a bot). Required because Bot API cannot read channels not added as admin. 11.5K GitHub stars. Latest: 1.42.0 (Nov 2025). Pyrogram is abandoned — do not use. |
| fastfeedparser | `>=0.5` | Parse RSS feeds from RBC, Interfax, TASS, Kommersant | 25x faster than feedparser, familiar API, supports RSS 2.0 / Atom 1.0 / RDF. Actively maintained (v0.5.9, Mar 2026). No async fetch built in — pair with existing `httpx` for concurrent feed fetching. feedparser 6.0.x is the alternative but slower and poorly maintained. |

### Supporting Libraries (No New Dependencies Needed)

These capabilities come from libraries already in the stack:

| Existing Library | New Use for MOEX Milestone |
|-----------------|---------------------------|
| `t-tech-investments` SDK | Call `GetBonds()`, `GetBondCoupons()`, `GetAccruedInterests()` proto methods — already present in proto, just not wired in Python yet |
| `httpx` | Async RSS feed fetching (pair with fastfeedparser); already in stack |
| `apscheduler` | Schedule bond cycle, coupon calendar checks, daily P&L reports; already in stack |
| `anthropic` | Russian-language news analysis prompts; Claude Sonnet handles Russian text |
| `lxml` | HTML scraping fallback for news sources without RSS; already in stack |

### Russian News RSS Endpoints

| Source | RSS URL | Update Frequency | Notes |
|--------|---------|-----------------|-------|
| RBC | `http://static.feed.rbc.ru/rbc/logical/footer/news.rss` | ~15 min | Business/financial focus, most relevant |
| Interfax | `https://www.interfax.ru/rss.asp` | ~15 min | Wire service, company-level news |
| TASS | `http://tass.ru/rss/v2.xml` | ~10 min | State wire, macro/geopolitical |
| Kommersant | `https://www.kommersant.ru/RSS/main.xml` | ~30 min | Business newspaper |

Confidence: MEDIUM. These URLs are community-documented and confirmed working as of 2025 sources, but Russian news sites occasionally change RSS paths. Validate at implementation time.

### T-Invest API Bond Methods (No New Library)

The `t-tech-investments` gRPC proto already exposes:

| Proto Method | Data Returned | Bond Use |
|-------------|--------------|---------|
| `GetBonds` | Bond instrument list with FIGI, coupon rate, maturity, currency | Instrument discovery |
| `GetBondCoupons` | Coupon schedule: date, amount, period | Coupon income calendar |
| `GetAccruedInterests` | NKD (накопленный купонный доход) by date range | Clean vs dirty price calculation |
| `GetBondByFigi` | Single bond details including duration, yield to offer | Yield analytics |

These require implementing new methods on `TinkoffFetcher` — no new SDK needed.

---

## Installation

```bash
# Bond math
uv add "QuantLib>=1.41"

# Telegram bot (alerts out)
uv add "aiogram>=3.26"

# Telegram channel reader (news in)
uv add "Telethon>=1.42"

# RSS feed parser
uv add "fastfeedparser>=0.5"
```

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| QuantLib | Manual bond math in numpy | QuantLib has battle-tested day count conventions (ACT/365, 30/360), yield curve bootstrapping, and Russian OFZ-specific settlement rules. Manual implementation is error-prone for accrued interest on coupon dates. |
| QuantLib | `bond-pricing` PyPI package | Tiny library (last release 2022), no active maintenance, missing convexity and yield curve support. |
| aiogram | python-telegram-bot | Both are async in 2025. aiogram has strictly typed API models, lighter deps, and better mypy compatibility — matches existing coding conventions. |
| aiogram | Direct Telegram REST via httpx | Extra maintenance burden; aiogram handles retry, rate-limiting, and update routing. |
| Telethon | Pyrogram | Pyrogram is effectively abandoned (no PyPI release in 12+ months). Multiple community forks (pyrofork, pyroblack) exist but introduce instability. Telethon 1.42.0 was released Nov 2025 and is actively maintained. |
| fastfeedparser | feedparser 6.0.x | feedparser's last release was 2023, no async support, significantly slower. fastfeedparser provides 25x speedup with identical API and was released Jan 2026. |
| httpx (existing) | aiohttp for RSS fetching | httpx already in stack — avoid adding aiohttp as a second async HTTP client. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Pyrogram | Abandoned. No PyPI release since 2023. Forks are unstable community projects. | Telethon 1.42 |
| feedparser | Last major release 2023, no async, 25x slower than fastfeedparser | fastfeedparser 0.5.9 |
| yfinance for MOEX bonds | Cannot fetch MOEX instruments — OFZ FIGIs are not on Yahoo Finance | T-Invest gRPC API via existing `TinkoffFetcher` |
| TKSBrokerAPI | Third-party REST wrapper over T-Invest — adds abstraction layer over gRPC SDK already wired | Direct t-tech-investments SDK (already in stack) |
| Pyrogram forks (pyrofork, pyroblack) | Community forks with no stability guarantees, API diverges from upstream | Telethon |
| APScheduler 4.0 | Explicitly marked pre-release / not production ready by maintainer | APScheduler 3.x (already in stack at `>=3.10.4`) |
| Celery for bond scheduling | Over-engineered for scheduled trading loops; requires Redis worker management overhead | APScheduler (already in stack) |

---

## Stack Patterns by Variant

**For Telegram alerts (outbound only, no user interaction):**
- Use aiogram in polling or webhook mode
- Only needs `BOT_TOKEN` — no Telegram API app credentials
- Simpler: one bot account sends to a monitoring channel

**For Telegram channel reading (inbound news scraping):**
- Use Telethon with a dedicated user account (not bot)
- Requires `API_ID` + `API_HASH` from my.telegram.org + user session string
- Store session string encrypted in environment variables
- Do NOT reuse the trading account — use a dedicated monitoring account

**For bond yield curve analysis:**
- Use QuantLib `FlatForward` for simple YTM discounting
- Use `ZeroCurve` or `PiecewiseYieldCurve` for OFZ zero-coupon curve from CBR data
- CBR publishes OFZ zero-coupon yields daily at `cbr.ru/hd_base/zcyc_params/`

**For Russian news ingestion:**
- Fetch RSS with `httpx` (async, concurrent) → parse with `fastfeedparser` → deduplicate by GUID/link → send to `NewsAnalyzer` (existing Claude Sonnet pipeline)
- Telegram channels supplement RSS for real-time signals not covered by wire services

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| QuantLib 1.41 | Python 3.12 | Precompiled wheels available; no Cython/SWIG build needed |
| aiogram 3.26 | Python 3.10+ | Requires Python 3.10+; project uses 3.12, no conflict |
| Telethon 1.42 | Python 3.8+ | Pure Python asyncio; no binary deps |
| fastfeedparser 0.5.9 | Python 3.8+ | Pure Python; uses lxml internally — lxml already in stack |

---

## Sources

- [aiogram PyPI](https://pypi.org/project/aiogram/) — version 3.26.0, Mar 2026 (HIGH confidence)
- [aiogram docs](https://docs.aiogram.dev/) — feature verification (HIGH confidence)
- [Telethon PyPI](https://pypi.org/project/Telethon/) — version 1.42.0, Nov 2025 (HIGH confidence)
- [Telethon GitHub](https://github.com/LonamiWebs/Telethon) — 11.5K stars, active maintenance (HIGH confidence)
- [QuantLib PyPI](https://pypi.org/project/QuantLib/) — version 1.41, Jan 2026 (HIGH confidence)
- [QuantLib Python Docs](https://quantlib-python-docs.readthedocs.io/en/latest/instruments/bonds.html) — BondFunctions API (HIGH confidence)
- [fastfeedparser PyPI](https://pypi.org/project/fastfeedparser/) — version 0.5.9, Mar 2026 (HIGH confidence)
- [fastfeedparser GitHub](https://github.com/kagisearch/fastfeedparser) — 25x faster than feedparser claim (MEDIUM confidence — no independent benchmark found)
- [Pyrogram status](https://snyk.io/advisor/python/pyrogram) — maintenance inactive (HIGH confidence)
- [Tinkoff InvestAPI proto](https://github.com/Tinkoff/investAPI/blob/main/src/docs/contracts/instruments.proto) — GetBonds, GetBondCoupons, GetAccruedInterests methods (HIGH confidence)
- [Russian news RSS feeds](https://rss.feedspot.com/russian_news_rss_feeds/) — RSS URL discovery (MEDIUM confidence — URLs may change)
- [piptrends aiogram vs python-telegram-bot](https://piptrends.com/compare/python-telegram-bot-vs-aiogram) — download trends comparison (MEDIUM confidence)

---
*Stack research for: Autonomous MOEX trading — bonds, news analysis, Telegram*
*Researched: 2026-03-14*
