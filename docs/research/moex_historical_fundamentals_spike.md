# MOEX Historical Fundamentals — Feasibility Spike

Research only. NO production code was written. Goal: decide whether honest,
look-ahead-safe HISTORICAL fundamental data for our MOEX equities can be sourced
from somewhere OTHER than T-Bank (which is point-in-time/current only), so Stage 3
(fundamentals-in-ML) need not wait months for live-forward capture.

Date: 2026-05-31. Probes were run against the public MOEX ISS HTTP API (no token)
and public web pages. No secrets used. Nothing committed.

---

## Summary verdict

**VIABLE — partially, with a clear winner and a clear caveat.**

- **Best source: SmartLab (smart-lab.ru)** — it is the ONLY source we probed that
  exposes BOTH (a) per-quarter fundamentals (revenue, net income, EPS, ROE,
  margins, P/E) AND (b) an explicit per-report **disclosure date** ("Дата отчёта"),
  for both blue chips (SBER) and growth-tech (OZON). The disclosure date is the
  make-or-break field for look-ahead safety, and SmartLab carries it directly.
  Effort tier **M** (HTML scrape + parse + ToS risk).
- **MOEX ISS gives us only two useful pieces**: (1) a rich **dividend history**
  (real, dated, all our symbols) and (2) the static **shares outstanding**
  (`ISSUESIZE`), from which historical **market_cap** can be reconstructed as
  `close_price × ISSUESIZE`. ISS carries **NO issuer financials / ratios / P/E /
  revenue history**. Effort tier **S** for what it does cover.
- **e-disclosure.ru** (Interfax) is the canonical, legally-authoritative source and
  filings DO carry a publication date — but the machine API ("Шлюз") is a
  **paid, contract-only** gateway (pricing on request, phone +7 495 787-5213), and
  the free public site must be scraped + the РСБУ/МСФО PDFs/XBRL parsed. Effort
  tier **L**.
- **Paid vendors** (LSEG/Refinitiv, Bloomberg, FactSet) carry true point-in-time
  fundamentals but cost/licensing is prohibitive for this project (one paragraph
  below). Effort tier **L** + recurring cost.

**Recommendation:** use **SmartLab as the primary fundamentals backfill source**
(it is the only free/cheap source with disclosure dates) and **MOEX ISS for
dividends + reconstructed market_cap**. Where SmartLab lacks a row's date or a
symbol, fall back to the **CPI-style conservative publication-lag** machinery
already in `cbr.py` (a `+publication-lag` map keyed on fiscal-period end). This is
an honest, look-ahead-safe path that does NOT require months of live-forward
accumulation. Caveat: **growth-tech history is short** (OZON quarterly fundamentals
only visible back to ~2025Q2), so deep history exists only for established blue
chips.

---

## Per-source table

| Source | Covers our symbols? | Has history (multiple dated points)? | Has publication/as_of date? | Access method | Effort | Legality |
|---|---|---|---|---|---|---|
| **MOEX ISS** (dividends) | YES — all 4 segments resolve; dividend rows: blue chips rich (SBER 6, LKOH 25, MGNT 24, GMKN 21, ROSN 19, TATN 23, SIBN 23, NVTK 14, SNGS 12), growth-tech sparse (OZON/VKCO/CIAN/YNDX = 0, YDEX = 2, TCSG = 0, CBOM 1) | YES for dividends | **NO** — only `registryclosedate` (ex-div record date), not declaration date | Public REST JSON, no token | **S** | OK (public read API; MOEX owns redistribution rights — internal use only) |
| **MOEX ISS** (financials/ratios) | n/a | **NO endpoint exists** | n/a | — | — | — |
| **MOEX ISS** (market cap) | YES — `ISSUESIZE` static; live `ISSUECAPITALIZATION` current-only | Reconstructable: `close × ISSUESIZE` from price history | as_of = trade date (clean, look-ahead-safe) | Public REST JSON | **S** | OK |
| **SmartLab** (smart-lab.ru) | YES — SBER and OZON both confirmed; site covers MOEX issuers broadly | YES — quarterly IFRS/РСБУ tables; blue-chip history deep, **tech history short (~1yr)** | **YES — "Дата отчёта" row = disclosure date** (e.g. SBER 28.04.2025/29.07.2025/28.10.2025/26.02.2026; OZON 07.08.2025/10.11.2025/26.02.2026/28.04.2026) | HTML scrape (no public API); 403s on naive fetch → needs browser-like headers | **M** | GREY — no public API; ToS/robots must be checked; data ultimately derived from MOEX-owned disclosures |
| **e-disclosure.ru** (Interfax) | YES — all RU public issuers (authoritative) | YES — every filing | **YES — each disclosure event carries its publication timestamp** | Paid contract API ("Шлюз", REST/JSON, Swagger) OR scrape free public site + parse РСБУ/МСФО PDF/XBRL | **L** | API: contractual/licensed. Public site: read-only, but parsing reports is heavy |
| **Conomy / Финам screeners** | Partial | Screeners show current + some history | Mostly **fiscal-period only**, disclosure date not consistently exposed | Scrape | **M–L** | GREY (ToS) |
| **Paid vendors** (LSEG/Refinitiv, Bloomberg, FactSet) | YES, full | YES, true point-in-time DB | **YES** (vendor PIT product) | Licensed API | **L** + cost | Prohibitive license cost |

Empirical probes that back the table:
- `GET /iss/securities/SBER.json` → static reference (SECID, ISIN, ISSUESIZE
  21,586,948,000, FACEVALUE, listing dates). No financials.
- `GET /iss/securities/{SECID}/dividends.json` → dated dividend history, columns
  `[secid, isin, registryclosedate, value, currencyid]`. **registryclosedate is the
  ex-div record date, NOT the announcement date.**
- `GET /iss/.../boards/TQBR/securities/SBER.json` marketdata → `ISSUECAPITALIZATION`
  = 6.93e12 RUB **but live/current only**; the `/history/...` endpoint columns have
  price/volume but **no CAPITALIZATION column** → historical mkt cap must be
  reconstructed `CLOSE × ISSUESIZE`.
- `/iss/analyticalproducts.json` → 404 (no fundamentals product).
- SmartLab `/q/SBER/f/q/MSFO/` and `/q/OZON/f/q/MSFO/` → quarterly tables with a
  **"Дата отчёта"** (report/disclosure date) row plus fiscal-quarter labels.
- e-disclosure.ru "Шлюз (API)" page → REST/JSON, HTTPS, Swagger, **authorized users
  only, pricing by contract** (contact +7 495 787-5213).

---

## Look-ahead handling per source (the decisive question)

Can we stamp each historical fundamental with the date it BECAME PUBLIC?

- **SmartLab → YES (best).** The "Дата отчёта" row IS the disclosure/publication
  date. We can set `FundamentalSnapshot.as_of = utc(Дата отчёта)` directly → an
  honest point-in-time backtest with no lag fudge. (Spot-checked SBER and OZON;
  the dates match real reporting calendars.)
- **e-disclosure.ru → YES.** Each disclosure event carries a publication timestamp;
  if we ever licensed the gateway, `as_of` = event publish time. Cleanest in theory,
  highest effort/cost in practice.
- **MOEX ISS dividends → PARTIAL.** `registryclosedate` is the ex-div record date,
  not the declaration date. For a `dividend_yield`-style feature the record date is
  acceptable (the market already knows by then) and is itself look-ahead-safe to use
  as `as_of`. Do NOT treat it as the announcement date.
- **MOEX ISS market_cap (reconstructed) → YES.** `as_of` = the trade date of the
  close used; trivially look-ahead-safe.
- **Conomy/Финам → typically NO.** Mostly fiscal-period-end stamps → would require
  the publication-lag compromise.
- **Paid vendors → YES** (point-in-time by design).

**Fallback for any row lacking a publication date** (mirror `cbr.py` CPI machinery):
build a `FUNDAMENTAL_PUBLICATION_DATES` map keyed on `(symbol, fiscal_period)`, and
for missing entries derive a conservative effective date = fiscal-period-end +
publication lag. Russian issuers file IFRS roughly **45–75 days** after
quarter-end; use a conservative **+75 days** (analogous to
`_CPI_PUBLICATION_LAG_MONTHS` / `_effective_cpi_publication_date`). Clearly label
such snapshots as lag-approximated. SmartLab's actual dates above (Q1 reported late
Apr, Q2 early Aug, Q3 late Oct/early Nov, Q4 late Feb) validate the ~45–75d window.

---

## Field mapping to `FundamentalSnapshot`

`FundamentalSnapshot(symbol, as_of, pe_ratio, ev_ebitda, revenue_ttm, net_margin,
roe, eps_ttm, dividend_yield, market_cap, currency)` — every field Optional,
never fabricated.

| Schema field | SmartLab (primary) | MOEX ISS | Notes |
|---|---|---|---|
| `symbol` | row context | SECID | — |
| `as_of` | **"Дата отчёта"** (disclosure date) → UTC | trade date / registryclosedate | look-ahead key |
| `pe_ratio` | P/E column | — | direct |
| `ev_ebitda` | EV/EBITDA column (if present) | — | not all issuers (banks: n/a) |
| `revenue_ttm` | sum trailing 4 quarters or LTM column | — | banks report net interest+fee income instead |
| `net_margin` | net income / revenue, or margin column | — | derived |
| `roe` | ROE column | — | direct |
| `eps_ttm` | EPS column (LTM) | — | direct |
| `dividend_yield` | — (or compute) | **dividends.json** `value` / price | best from ISS dividends + price |
| `market_cap` | — | **`CLOSE × ISSUESIZE`** | reconstructed historically; or live `ISSUECAPITALIZATION` |
| `currency` | RUB | `currencyid` / FACEUNIT (SUR→RUB) | — |

Bank issuers (SBER, VTBR, TCSG, MOEX, CBOM) lack revenue/EV-EBITDA in the
conventional sense — leave those fields `None` and rely on net income, EPS, ROE,
P/E, P/B, NIM. This matches the "never fabricate, Optional defaults to None" schema
contract.

---

## Concrete recommendation

1. **Primary source: SmartLab** for per-quarter fundamentals, using the
   **"Дата отчёта" as `as_of`** → honest point-in-time, no lag needed where present.
2. **MOEX ISS** for (a) **dividend history** (`/securities/{SECID}/dividends.json`,
   `as_of = registryclosedate`) and (b) **reconstructed historical market_cap**
   (`CLOSE × ISSUESIZE`, `as_of = trade date`). Both free, no token, extend the
   existing `MoexISSFetcher` httpx pattern.
3. **Publication-date strategy:** prefer the explicit SmartLab disclosure date.
   For any (symbol, period) missing a date, fall back to a
   `FUNDAMENTAL_PUBLICATION_DATES` static map + conservative **+75-day lag** from
   fiscal-period end, reusing the exact `cbr.py` CPI pattern
   (`CPI_PUBLICATION_DATES`, `_effective_cpi_publication_date`,
   `get_latest_published_*`). Label lag-derived rows distinctly.
4. **Scope for a backfill phase (rough):**
   - New Layer-2 fetcher(s): `smartlab_fundamentals.py` (scrape+parse quarterly
     tables → `FundamentalSnapshot`), extend `moex_iss.py` with `fetch_dividends`
     + market-cap reconstruction.
   - A `FUNDAMENTAL_PUBLICATION_DATES` map + lag helper (mirror CBR/CPI).
   - One-off backfill script over our 4 segments → cache to parquet/DB keyed by
     `(symbol, as_of)`, filtered downstream by `as_of <= D`.
   - TDD + ruff + mypy as usual; honesty/no-fabrication tests.
   - Deep history available for blue chips; tech (OZON/VKCO/CIAN/YDEX) limited to
     ~1–2 years → Stage 3 should weight/segment accordingly or restrict
     fundamentals features to segments with adequate history.

---

## Risks / unknowns

- **SmartLab is a scrape, not an API.** Naive WebFetch returns 403 → needs
  browser-like headers; HTML layout can change and break parsers; **ToS/robots.txt
  must be checked before any automated pull** (legal grey zone — data is derived
  from MOEX-owned disclosures; MOEX restricts redistribution). Treat as
  internal-research use; do not redistribute.
- **Data provenance & accuracy:** SmartLab is a community/aggregator site, not the
  primary filer. Values should be sanity-checked against ISS price/dividends and,
  for a sample, against e-disclosure filings.
- **"Дата отчёта" semantics:** confirmed it is the report/disclosure date for the
  two symbols probed (SBER, OZON); should be verified across more issuers before
  trusting it blindly as the universal `as_of`.
- **Growth-tech history is shallow** (recent IPOs/redomiciliations: YNDX→YDEX,
  OZON, VKCO, CIAN) — deep fundamental history simply does not exist; live-forward
  capture remains the only way to lengthen it over time.
- **ISS dividend date is record date, not announcement** — acceptable but must be
  documented; do not mislabel as declaration date.
- **e-disclosure paid API** pricing unknown (contract-only); could become the
  authoritative upgrade path later if budget allows.
- **Survivorship/restatement bias:** quarterly tables may show restated figures
  rather than as-first-reported numbers; SmartLab's disclosure date mitigates but
  does not fully eliminate this — a known limitation of any non-PIT-vendor source.

---

### Sources
- MOEX ISS public API probes (live, this spike): `/iss/securities/SBER.json`,
  `/iss/securities/{SECID}/dividends.json`, `/iss/.../TQBR/securities/SBER.json`,
  `/iss/history/...`, `/iss/analyticalproducts.json` (404).
- [SmartLab SBER MSFO quarterly](https://smart-lab.ru/q/SBER/f/q/MSFO/),
  [SmartLab OZON MSFO quarterly](https://smart-lab.ru/q/OZON/f/q/MSFO/)
- [e-disclosure.ru API gateway ("Шлюз")](https://www.e-disclosure.ru/poluchenie-informacii/shlyuz-api)
- [Interfax disclosure product](https://group.interfax.ru/products/systems/disclosure/)
