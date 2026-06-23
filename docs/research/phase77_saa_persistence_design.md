# Phase 77 (v11.2 W2) — SAA Persistence Layer

**Status:** DESIGN (consilium-approved). **Branch:** `gsd/phase-77-saa-persistence-layer` (off origin/main 836b259 = #274).
**Route item 2 of the Asset-Allocation MVP** (persistence so a real portfolio survives a restart).

## Goal / Why

Today the whole allocation product is in-memory/config: a user's chosen `RiskProfile`, budget, and the live
deposit ladder (`DepositTranche` with daily-mutating `accrued_net`/`accrued_gross`/`broken`) have **zero DB
backing** (no User/account/deposit/target table in `core/models.py`). Phase 77 persists the **two pieces of
genuinely-mutable, non-reconstructable user state** so the portfolio is reloadable: the portfolio
identity+risk choice, and each living deposit tranche's accrued marks. Everything else is derivable
(target weights from `load_allocation_profiles(risk_profile)`), reconstructable by replay (NDFL/floor
accumulators), or a query (ASV per-bank) — so persisting it now would create stale dual-source-of-truth.

## Scope

**In:** 2 ORM models (`core/models.py`, L0) + 1 hand-written Alembic migration `012` (off `011`, plain
`op.create_table` style — NOT a hypertable) + a write/upsert seam through the existing `TradingPersistence`
+ a restart-reload loader + the replay-equivalence gate. SQLAlchemy 2.0 typed `Mapped[...]`,
`from __future__ import annotations`, `Decimal` (never float) for money.

**Out (deferred, with reasons):** User/auth/multi-tenancy (single-operator MVP); cost-basis lot table (no
live broker yet, equity is passive — Phase 4-5); separate equity/OFZ holdings table (passive index, no
per-name positions); separate BankAllocation table (`bank_id` inline, ASV is a `GROUP BY` query); persisted
NDFL/floor accumulators (reconstructable by replay — the replay-equivalence test is the gate); persisted
target vector (DERIVED from `risk_profile`); rebalance-history audit table; any hypertable/retention/
compression; live broker sync; any optimizer or gate softening.

## Operator decisions taken (consilium open questions — low-stakes MVP defaults, all reversible/additive)

1. **Single-operator MVP** — no User/account model, bare portfolio UUID, no `owner_id` column (a nullable
   `owner_id` is one additive migration if multi-user arrives).
2. **Derive target weights** from `risk_profile` (single source of truth) — no `TargetAllocationSnapshot`
   table (auditable-snapshot fork deferred).
3. **`ON DELETE RESTRICT`** on `deposit_tranches.portfolio_id` (safety, no silent data loss).

## Tables (minimal)

**`saa_portfolios`** — identity + the single SAA choice. Low-cardinality plain table.
`id UUID PK (uuid.uuid4)` · `risk_profile String(12) NOT NULL` (RiskProfile value, app-validated) ·
`budget_rub Numeric(20,2) NOT NULL` (large-cash precision) · `is_active Boolean NOT NULL default True` ·
`created_at/updated_at DateTime(tz) NOT NULL default now(UTC)`.

**`deposit_tranches`** — one mutable row per ladder rung, mirroring the `DepositTranche` dataclass 1:1 +
FK + `bank_id` + `updated_at`. `id UUID PK` · `portfolio_id UUID FK→saa_portfolios.id NOT NULL (ON DELETE
RESTRICT)` · `principal Numeric(20,2)` · `term_months Integer` (3/6/12) · `annual_rate Numeric(8,4)` ·
`open_date Date` · `maturity_date Date` · `accrued_net Numeric(20,2) default 0` · `accrued_gross
Numeric(20,2) default 0` · `broken Boolean default False` · `bank_id String(50) NULL` (ASV slice) ·
`updated_at DateTime(tz)`. `accrued_*` are mark-only (CR-01: stay IN the tranche, never a cash column).
Indexes: `ix_deposit_tranches_portfolio_id`, `ix_saa_portfolios_is_active`.

## TDD subtasks (P2-01 … P2-07)

- **P2-01 RED:** static AST migration test `tests/integration/migrations/test_012_saa_persistence.py`
  (mirror `test_010_agent_decisions.py` — parse `revision=='012'`/`down_revision=='011'`, assert both
  table-name + FK + downgrade literals; NO DB).
- **P2-02 GREEN:** add `SaaPortfolioModel` + `DepositTrancheModel` to `core/models.py` (typed, UUID
  `uuid.uuid4`, `Numeric(20,2)` money, `Date`, `DateTime(tz)`, real `ForeignKey`).
- **P2-03 GREEN:** hand-write `alembic/versions/012_saa_persistence.py` (plain `op.create_table` per
  `001_initial.py`; types byte-match the ORM; indexes; symmetric `downgrade` reverse-FK order). P2-01 → green.
- **P2-04 (CI live-DB):** `pytest.mark.integration`, skip-gated on `FINALAYZE_DATABASE_URL`: `upgrade head`,
  `information_schema` precision/tz introspection, FK enforcement (bogus `portfolio_id`→IntegrityError),
  ORM round-trip preserving Decimal, `downgrade 011` clean.
- **P2-05 GREEN:** write/upsert seam in `orchestration/db_persistence.py` (`session.add` for new rows;
  `pg_insert(...).on_conflict_do_update(index_elements=['id'], set_={accrued_net,accrued_gross,broken,
  updated_at})` for per-bar accrual). No new Repository/DAO class.
- **P2-06 GREEN:** restart-reload loader (active portfolio + non-broken unmatured tranches → rehydrate a
  `DepositSimulatedBroker`); reconstruct NDFL/floor accumulators by **replay from Jan-1**. **Gate:**
  replay-equivalence test — reloaded-then-replayed broker reproduces bit-identical
  `accrued_net`/`accrued_gross`/`_total_tax_paid` across the Jan-1 boundary (justifies NOT persisting
  the accumulators; if not bit-exact, fall back to a JSONB accumulator column).
- **P2-07:** `ruff check .` + `ruff format --check` + `mypy src/` green; static AST test green locally;
  live-DB tests in CI.

## Success criteria

`012` applies onto `011` + reverts cleanly; static AST test passes with no DB; live-DB round-trip preserves
`Numeric(20,2)/(8,4)` Decimal precision + tz; FK enforced; per-bar upsert UPDATEs in place (no row
duplication) via `TradingPersistence`; ruff + mypy green; every new file has
`from __future__ import annotations` + `Decimal` for money.

---

## OUTCOME — reload reconstruction (P2-06) reworked: DIRECT LOAD, no replay

Adversarial code review (CR-01) caught a real correctness bug in the FIRST (replay-based) reload: it
replayed `accrue()` over **calendar days**, but the live ladder accrues on **trading days only**
(`backtest/engine.py:840` iterates candle timestamps, ~252/yr) and `accrue()` compounds `(1+annual)^(1/252)`
**per call** — so a calendar replay over-compounds the mark (~365 steps where live had ~252). The original
"replay-equivalence" gate missed it (its oracle also used calendar days: replay-vs-replay, trivially equal).

**Fix (operator-directed): reload is now a DIRECT LOAD, never a replay** — cadence-independent and bit-exact.
The broker's mutable state is persisted in two places and restored verbatim: per-tranche accrued marks →
`deposit_tranches` (already); the broker-level year-scoped accumulators (`_ytd_deposit_gross` /
`_running_max_key_rate` / `_current_year`) + totals + last-accrual-date → a new
`saa_portfolios.deposit_accumulators` JSONB column. `load_deposit_broker_from_db` loads tranches (marks) +
`serialize/restore_deposit_accumulators`, with NO replay. The binding gate is real: after persist→restore,
the NEXT `accrue()` resumes BIT-IDENTICALLY to a never-restarted broker
(`test_restore_then_next_accrue_matches_live`) — the cadence bug class is structurally impossible.

Other review notes addressed: the upsert unit test now asserts the compiled `ON CONFLICT DO UPDATE`
statement (not just that `execute` was called); the live-DB test asserts both indexes and the
`ON DELETE RESTRICT` semantics. Known limitation kept: the upsert's `bank_id` is `None` and absent from the
natural key (DepositTranche carries no `bank_id`), so the `bank_id` column is reserved/nullable until a
later phase threads bank identity through.
