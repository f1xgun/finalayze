"""Single-source financial constants (Layer 0).

NDFL rate, ASV insurance cap, deposit demand rate, progressive-band
thresholds and the deposit non-taxable-floor base. Pure data + stdlib only;
zero project imports (true L0). Consumed downward by L2 (cbr deposit-rate)
and L5 (all three sleeves) -- see docs/architecture/DEPENDENCY_LAYERS.md.

Statutory values (the 2.4M RUB band threshold, the 15% upper rate, the 1M RUB
floor base) are named constants so the operator can adjust them without code
surgery if the law changes (research Assumption A2).
"""

from __future__ import annotations

from decimal import Decimal

# Russian personal income tax (NDFL). SINGLE source of truth -- dedupes
# bond_simulated_broker.py (_NDFL_TAX_RATE) + sandbox_tracker.py (_NDFL_RATE).
NDFL_RATE = Decimal("0.13")  # 13% base band
NDFL_RATE_HIGH = Decimal("0.15")  # 15% above the progressive threshold
NDFL_PROGRESSIVE_THRESHOLD = Decimal(2_400_000)  # 2.4M RUB taxable income/yr (13->15% switch, D-10)

# ASV deposit-insurance cap per bank (D-07/D-09). Accrued interest counts toward it.
ASV_CAP_PER_BANK = Decimal(1_400_000)

# "Do vostrebovaniya" (demand) rate a broken tranche resets to (D-03).
DEPOSIT_DEMAND_RATE = Decimal("0.0001")  # ~0.01% annual

# Deposit non-taxable-floor base (D-10):
# floor = DEPOSIT_FLOOR_BASE x max-monthly-key-rate-in-year.
DEPOSIT_FLOOR_BASE = Decimal(1_000_000)  # 1M RUB; floor ~= 1M x 0.21 ~= 210k RUB in 2024-25

# Deposit-ladder optimizer (Phase 88): a "forward" offered-rate snapshot older than
# this many days fails closed (no recommendation on stale offers). "backtest"-mode
# snapshots (historical-window evaluations) are exempt -- their as_of is the evaluation
# start, not a freshness claim.
MAX_OFFER_STALENESS_DAYS = 14

# ASV raised insurance tiers (Minfin, "from 18 Dec 2025"; effective-date/legal-force should
# be re-verified -- a SOFT reported metric only, never a hard cap). Boundary = strictly >3yr.
# Ordinary ruble deposit >3yr -> 2.0M; irrevocable savings cert 1-3yr -> 2.0M, >3yr -> 2.8M.
ASV_RAISED_TIER_2M = Decimal(2_000_000)
ASV_RAISED_TIER_2_8M = Decimal(2_800_000)
