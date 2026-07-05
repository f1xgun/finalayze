"""Wire the RU-securities tax-optimization engine to the operator's REAL portfolio.

DECISION-SUPPORT ONLY. This script reads ALREADY-SAVED readonly JSON snapshots
(accounts + per-account operations + per-account portfolio) from a data
directory and PRINTS an aggregated decision-support report. It NEVER places an
order, NEVER trades, NEVER uses a token, NEVER touches the network. Real money =
HARD STOP. All money math is ``decimal.Decimal`` -- never float.

It reuses the merged tax engine (``finalayze.tax.*``) and applies the REAL-DATA
rules:

- The account named "IIS" is an IIS account -> its lots are ``on_iis=True`` and
  are EXCLUDED from LDV and loss-harvest (IIS has its own regime); its coupon /
  dividend context is still reported separately.
- All other accounts are ordinary taxable brokerage.
- ``russian_issuer`` heuristic: a RUB-denominated MOEX instrument (MOEX-style
  FIGI / RU ticker) is treated as Russian; anything ambiguous/foreign is
  ``russian_issuer=False`` and FLAGGED for manual review (no LDV credit).
- Cash INPUT / INP_MULTI / OUTPUT / OUT_MULTI rows are cash movements, NOT lots.
  A securities transfer-in (INPUT carrying a ticker+quantity) is a cost-basis-
  UNKNOWN lot -> flagged, never a fabricated cost/date.
- gross dividend = DIVIDEND (net) + |DIVIDEND_TAX| (the withheld tax).
- coupons feed base A (netted with realized gains); dividends are a SEPARATE
  base (never netted, never harvested).
- FIFO is replayed PER INSTRUMENT (grouped by the stable ticker), never pooled
  across instruments -- a SELL of X can only consume lots of X (CR-1). A genuine
  per-instrument oversell (truncated buy history) is flagged HISTORY_TRUNCATED
  and that one instrument is excluded, not the whole account (CR-3).
- a bond redemption (BOND_REPAYMENT_FULL) is applied as a disposal at par of the
  still-open lot(s) of that bond, matched by ticker across the buy-vs-redemption
  figi drift, so the redeemed bond is no longer shown open (CR-2); a partial
  amortization (BOND_REPAYMENT) is surfaced as context only (no per-event
  quantity to fabricate).
- the inline ``commission`` on a BUY/SELL is the trade fee; the separate
  BROKER_FEE row DUPLICATES it and is NOT added again (double-count guard).
- Since today is well before any lot turns 3 full years, ``LDV now`` is expected
  empty -- the valuable output is the APPROACHING-LDV advisory: for every open,
  cost-basis-known, Russian, non-IIS lot, the 3-year cross date and the RUB that
  would become LDV-exempt if held to it (hypothetical finrez at current price).

Pure/deterministic given the data dir. Saves NOTHING sensitive to the repo.

Run:
    uv run --directory <WORKTREE> python scripts/research/tax_real_portfolio.py
    uv run --directory <WORKTREE> python scripts/research/tax_real_portfolio.py --data-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from finalayze.core.ndfl import ndfl_marginal
from finalayze.tax.baskets import realized_ytd_base_a
from finalayze.tax.harvest import HarvestCandidate
from finalayze.tax.ldv import LDV_BOUNDARY_BUFFER_DAYS, LDV_MIN_FULL_YEARS, ldv_eligible
from finalayze.tax.lots import (
    FifoMatchError,
    Operation,
    OperationType,
    RealizedResult,
    TaxLot,
    fifo_match,
    open_lots_after_match,
)
from finalayze.tax.report import (
    DegradationFlag,
    FlagReason,
    TaxReport,
    build_report,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# --- defaults ---
_DEFAULT_DATA_DIR = Path("/tmp/taxdata")  # noqa: S108 - readonly operator snapshot dir
_TODAY = date(2026, 7, 5)
_YEAR = 2026
_RUB = "RUB"
_IIS_NAME = "ИИС"  # "IIS" in Cyrillic (data only; not a code identifier)

# APPROACHING-LDV window: a lot crossing 3 full years within this many days is
# advised on. ~12 months.
_APPROACHING_DAYS = 400

# Raw sidecar op tokens present in the REAL data that the static fixture parser
# does not know about. Mapped / handled here.
_TRADE_BUY = "OPERATION_TYPE_BUY"
_TRADE_SELL = "OPERATION_TYPE_SELL"
_DIVIDEND = "OPERATION_TYPE_DIVIDEND"
_DIVIDEND_TAX = "OPERATION_TYPE_DIVIDEND_TAX"
_COUPON = "OPERATION_TYPE_COUPON"
_INPUT = "OPERATION_TYPE_INPUT"
_INP_MULTI = "OPERATION_TYPE_INP_MULTI"
_OUTPUT = "OPERATION_TYPE_OUTPUT"
_OUT_MULTI = "OPERATION_TYPE_OUT_MULTI"
_BOND_REPAYMENT = "OPERATION_TYPE_BOND_REPAYMENT"
_BOND_REPAYMENT_FULL = "OPERATION_TYPE_BOND_REPAYMENT_FULL"
_TAX = "OPERATION_TYPE_TAX"
_BENEFIT_TAX = "OPERATION_TYPE_BENEFIT_TAX"
_BROKER_FEE = "OPERATION_TYPE_BROKER_FEE"
_TRACK_MFEE = "OPERATION_TYPE_TRACK_MFEE"
_TRACK_PFEE = "OPERATION_TYPE_TRACK_PFEE"

# fee rows that DUPLICATE an inline BUY/SELL commission or are pure account
# service fees -- reported as context, NEVER folded into a lot's cost basis.
_FEE_TYPES = frozenset({_BROKER_FEE, _TRACK_MFEE, _TRACK_PFEE})
# cash-movement rows that are NOT securities lots.
_CASH_TYPES = frozenset({_INPUT, _INP_MULTI, _OUTPUT, _OUT_MULTI})
# withholding / reconciliation-only tax rows (not a base contribution here).
_RECON_TAX_TYPES = frozenset({_TAX, _BENEFIT_TAX})


class RealDataError(Exception):
    """Raised when a real-data snapshot cannot be parsed into the engine shape."""


def _dec(value: Any) -> Decimal | None:
    """Parse a numeric field to Decimal via str (no float error). None passthrough."""
    if value is None:
        return None
    return Decimal(str(value))


def _parse_date(raw: str) -> date:
    text = str(raw).replace("Z", "+00:00")
    return datetime.fromisoformat(text).date()


# Known-foreign FIGIs that trade on MOEX in RUB but are NOT Russian issuers under
# post-2025 FZ 58-FZ (foreign, non-EAEU) -> LDV must NOT be credited. The clearest
# real-data case is the ex-Yandex shell now "Nebius Group N.V." (Netherlands).
# This is a small hand-list; anything not on it stays subject to the RUB+MOEX
# heuristic and is additionally flagged whenever its current price is unavailable.
_KNOWN_FOREIGN_FIGIS = frozenset({"BBG006L8G4H1"})  # Nebius Group N.V. (ex-YNDX)

# Forced-redomicile foreign holding structures that now trade on MOEX in RUB as a
# RUSSIAN-registered MKPAO but whose ISSUER nationality is contested for LDV (art.
# 219.1 requires a Russian issuer; a redomiciled foreign structure is a grey area
# under FZ 58-FZ). We do NOT auto-deny LDV -- the RUB+MOEX heuristic still treats
# them as Russian -- but if such a lot were ever LDV-CREDITED we raise a needs-
# review FLAG so the operator confirms the issuer nationality before relying on it.
# INFO-2: currently these are sold-out / underwater so never mis-credited.
_REDOMICILE_REVIEW_FIGIS: dict[str, str] = {
    "BBG00JXPFBN0": "FIVE/X5 (X5 Retail Group forced redomicile)",
    "BBG000RMWQD4": "ENPG/EN+ (En+ Group forced redomicile)",
}
# Match redomicile structures by their TICKER too (figi drifts between the ops
# feed and the current registry; the ticker is the stable operator-facing key).
_REDOMICILE_REVIEW_TICKERS: frozenset[str] = frozenset({"FIVE", "X5", "ENPG"})


def _is_russian_issuer(figi: str, ticker: str, currency: str) -> bool:
    """Heuristic: a RUB-denominated MOEX instrument is a Russian issuer.

    MOEX FIGIs carry a ``BBG``/``TCS`` prefix; RU tickers are alphanumerics /
    ``RU000``/``SU`` bond codes / T-Bank fund tickers (``...@``). A non-RUB, a
    known-foreign FIGI, or an otherwise-ambiguous instrument returns False so the
    caller FLAGS it for manual review rather than crediting LDV on a possibly-
    foreign issuer.
    """
    if currency.upper() != _RUB:
        return False
    if figi.upper() in _KNOWN_FOREIGN_FIGIS:
        return False
    fg = figi.upper()
    if fg.startswith(("BBG", "TCS")):
        return True
    tk = ticker.upper()
    return bool(tk) and (tk.startswith(("RU000", "SU")) or tk.replace("@", "").isalnum())


# ----------------------------------------------------------------------------
# parsed-account container
# ----------------------------------------------------------------------------


@dataclass
class ParsedAccount:
    """Everything reconstructed from one account's ops + portfolio snapshot."""

    account_id: str
    name: str
    opened: date
    on_iis: bool
    buys: list[Operation] = field(default_factory=list)
    sells: list[Operation] = field(default_factory=list)
    coupons: list[Operation] = field(default_factory=list)
    dividends_net: list[Operation] = field(default_factory=list)  # DIVIDEND (net)
    dividend_tax_abs: Decimal = Decimal(0)  # sum |DIVIDEND_TAX| (all years)
    dividend_tax_by_year: dict[int, Decimal] = field(default_factory=dict)  # |tax| per year
    fees_total: Decimal = Decimal(0)  # broker/track fees (context only)
    withheld_tax_total: Decimal = Decimal(0)  # TAX/BENEFIT_TAX (reconciliation)
    bond_repayment_full: list[Operation] = field(default_factory=list)
    bond_amortization_total: Decimal = Decimal(0)  # partial BOND_REPAYMENT cash
    securities_input_flagged: int = 0  # transfer-in lots (cost-basis unknown)
    price_by_ticker: dict[str, Decimal] = field(default_factory=dict)  # from pf
    portfolio_positions: list[dict[str, Any]] = field(default_factory=list)  # raw pf rows
    russian_by_key: dict[str, bool] = field(default_factory=dict)  # figi|ticker -> ru
    unknown_op_types: dict[str, int] = field(default_factory=dict)


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return json.loads(text)


def _portfolio_prices(pf: Any) -> dict[str, Decimal]:
    """Map ticker -> current price from a portfolio snapshot (best effort)."""
    prices: dict[str, Decimal] = {}
    if not isinstance(pf, dict):
        return prices
    for pos in pf.get("positions", []) or []:
        tk = pos.get("ticker")
        cur = _dec(pos.get("currentPrice"))
        if tk and cur is not None:
            prices[str(tk)] = cur
    return prices


def _portfolio_position_index(pf: Any) -> dict[str, dict[str, Any]]:
    """Map ticker -> raw portfolio position (for hypothetical-finrez math)."""
    idx: dict[str, dict[str, Any]] = {}
    if not isinstance(pf, dict):
        return idx
    for pos in pf.get("positions", []) or []:
        tk = pos.get("ticker")
        if tk:
            idx[str(tk)] = pos
    return idx


def _make_trade(row: dict[str, Any], op_type: OperationType) -> Operation:
    figi = str(row.get("figi") or "")
    ticker = str(row.get("ticker") or "")
    currency = str(row.get("currency") or _RUB).upper()
    qty = _dec(row.get("quantity"))
    price = _dec(row.get("price"))
    # inline commission is the trade fee (negative in the raw feed) -> magnitude
    comm = _dec(row.get("commission"))
    commission = abs(comm) if comm is not None else Decimal(0)
    payment = _dec(row.get("payment")) or Decimal(0)
    return Operation(
        op_type=op_type,
        op_date=_parse_date(row["date"]),
        figi=figi,
        ticker=ticker,
        payment=payment,
        currency=currency,
        quantity=qty,
        price_per_unit=price,
        commission=commission,
        cost_basis_known=True,
    )


def parse_account(  # noqa: PLR0912 - one exhaustive op-type switch
    account_id: str,
    name: str,
    opened: date,
    ops_rows: Iterable[dict[str, Any]],
    pf: Any,
) -> ParsedAccount:
    """Reconstruct Operations/TaxLots inputs from one account's raw op rows.

    Applies every REAL-DATA rule (IIS exclusion via ``on_iis``, russian_issuer
    heuristic, securities-transfer-in cost-basis-unknown flag, bond redemption as
    a disposal, gross-dividend grossing-up, coupons->base A, fee de-duplication).
    Deterministic: no network, no token, no fabricated numbers.
    """
    on_iis = name.strip() == _IIS_NAME
    pf_positions: list[dict[str, Any]] = []
    if isinstance(pf, dict):
        pf_positions = [p for p in (pf.get("positions") or []) if isinstance(p, dict)]
    acc = ParsedAccount(
        account_id=account_id,
        name=name,
        opened=opened,
        on_iis=on_iis,
        price_by_ticker=_portfolio_prices(pf),
        portfolio_positions=pf_positions,
    )

    for row in ops_rows:
        raw = str(row.get("operationType") or "")
        figi = str(row.get("figi") or "")
        ticker = str(row.get("ticker") or "")
        currency = str(row.get("currency") or _RUB).upper()
        if figi or ticker:
            key = f"{figi}|{ticker}"
            acc.russian_by_key[key] = _is_russian_issuer(figi, ticker, currency)

        if raw == _TRADE_BUY:
            acc.buys.append(_make_trade(row, OperationType.BUY))
        elif raw == _TRADE_SELL:
            acc.sells.append(_make_trade(row, OperationType.SELL))
        elif raw == _COUPON:
            acc.coupons.append(
                Operation(
                    op_type=OperationType.COUPON,
                    op_date=_parse_date(row["date"]),
                    figi=figi,
                    ticker=ticker,
                    payment=_dec(row.get("payment")) or Decimal(0),
                    currency=currency,
                    payment_is_net_estimate=True,
                )
            )
        elif raw == _DIVIDEND:
            acc.dividends_net.append(
                Operation(
                    op_type=OperationType.DIVIDEND,
                    op_date=_parse_date(row["date"]),
                    figi=figi,
                    ticker=ticker,
                    payment=_dec(row.get("payment")) or Decimal(0),
                    currency=currency,
                    payment_is_net_estimate=True,
                )
            )
        elif raw == _DIVIDEND_TAX:
            pay = abs(_dec(row.get("payment")) or Decimal(0))
            acc.dividend_tax_abs += pay
            yr = _parse_date(row["date"]).year
            acc.dividend_tax_by_year[yr] = acc.dividend_tax_by_year.get(yr, Decimal(0)) + pay
        elif raw == _BOND_REPAYMENT_FULL:
            # full redemption at par -> a disposal of the remaining open lot(s).
            acc.bond_repayment_full.append(
                Operation(
                    op_type=OperationType.SELL,  # treated as a disposal
                    op_date=_parse_date(row["date"]),
                    figi=figi,
                    ticker=ticker,
                    payment=_dec(row.get("payment")) or Decimal(0),
                    currency=currency,
                )
            )
        elif raw == _BOND_REPAYMENT:
            # partial amortization: principal returned, no per-event quantity to
            # fabricate -> surface as context, do not synthesize a lot/qty.
            acc.bond_amortization_total += abs(_dec(row.get("payment")) or Decimal(0))
        elif raw in _FEE_TYPES:
            acc.fees_total += abs(_dec(row.get("payment")) or Decimal(0))
        elif raw in _RECON_TAX_TYPES:
            acc.withheld_tax_total += abs(_dec(row.get("payment")) or Decimal(0))
        elif raw in _CASH_TYPES:
            # a securities transfer-in carries a ticker+quantity -> cost-basis
            # UNKNOWN lot (flag, never fabricate). A cash movement carries neither.
            if ticker and _dec(row.get("quantity")):
                acc.securities_input_flagged += 1
        else:
            acc.unknown_op_types[raw] = acc.unknown_op_types.get(raw, 0) + 1

    return acc


# ----------------------------------------------------------------------------
# LDV approaching-advisory
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ApproachingLdv:
    """One open lot crossing the 3-year LDV threshold within the advisory window."""

    ticker: str
    figi: str
    account: str
    acquire_date: date
    cross_date: date
    quantity: Decimal
    cost_basis_known: bool
    russian_issuer: bool
    est_finrez: Decimal | None  # positive capital result at current price (None if unknown)
    est_saving: Decimal | None


def _cross_date(acquire: date) -> date:
    """Buffered earliest-LDV-eligibility date: 3-year anniversary + buffer days.

    INFO-1: the engine's ``full_years_held`` only counts a year once the holding
    reaches ``anniversary + LDV_BOUNDARY_BUFFER_DAYS`` (a T+2 / acquire-vs-credit
    safety buffer). To label the SAME date the engine's ``ldv_eligible`` would flip
    on, the advisory cross date must include that buffer -- otherwise we would
    advise a date on which the engine still returns ineligible.
    """
    try:
        anniv = acquire.replace(year=acquire.year + LDV_MIN_FULL_YEARS)
    except ValueError:
        anniv = acquire.replace(year=acquire.year + LDV_MIN_FULL_YEARS, day=28)
    return anniv + timedelta(days=LDV_BOUNDARY_BUFFER_DAYS)


def _lot_key_russian(lot: TaxLot, russian_by_key: dict[str, bool]) -> bool:
    return russian_by_key.get(f"{lot.figi}|{lot.ticker}", lot.currency.upper() == _RUB)


def _tag_lots(
    lots: list[TaxLot],
    *,
    on_iis: bool,
    russian_by_key: dict[str, bool],
) -> list[TaxLot]:
    """Re-stamp open lots with on_iis + russian_issuer per the REAL-DATA rules."""
    tagged: list[TaxLot] = []
    for lot in lots:
        russian = russian_by_key.get(f"{lot.figi}|{lot.ticker}", lot.currency.upper() == _RUB)
        tagged.append(
            TaxLot(
                figi=lot.figi,
                ticker=lot.ticker,
                acquire_date=lot.acquire_date,
                quantity=lot.quantity,
                price_per_unit=lot.price_per_unit,
                commission_buy=lot.commission_buy,
                currency=lot.currency,
                russian_issuer=russian,
                on_iis=on_iis,
                cost_basis_known=lot.cost_basis_known,
            )
        )
    return tagged


def approaching_ldv(
    acc: ParsedAccount,
    open_lots: list[TaxLot],
    today: date,
) -> list[ApproachingLdv]:
    """Lots crossing 3 full years within the advisory window (skip IIS)."""
    if acc.on_iis:
        return []
    out: list[ApproachingLdv] = []
    for lot in open_lots:
        if lot.acquire_date < date(2014, 1, 1):
            continue
        cross = _cross_date(lot.acquire_date)
        # already eligible now is handled elsewhere; here: not yet eligible but soon
        if cross <= today:
            continue
        if cross - today > timedelta(days=_APPROACHING_DAYS):
            continue
        est_finrez, est_saving = _hypothetical_finrez(acc, lot)
        out.append(
            ApproachingLdv(
                ticker=lot.ticker,
                figi=lot.figi,
                account=acc.name,
                acquire_date=lot.acquire_date,
                cross_date=cross,
                quantity=lot.quantity,
                cost_basis_known=lot.cost_basis_known,
                russian_issuer=lot.russian_issuer,
                est_finrez=est_finrez,
                est_saving=est_saving,
            )
        )
    return out


def _hypothetical_finrez(acc: ParsedAccount, lot: TaxLot) -> tuple[Decimal | None, Decimal | None]:
    """Positive capital result if the lot were disposed at the current price.

    Uses the portfolio ``currentPrice`` for the ticker. Returns (None, None) if
    the cost basis is unknown or no current price is available (never fabricate).
    A saving is valued at the 13% base rate (LDV would exempt this finrez); the
    marginal band is applied at the portfolio level, so per-lot we use the base
    rate as a conservative floor estimate.
    """
    if not lot.cost_basis_known:
        return None, None
    price = acc.price_by_ticker.get(lot.ticker)
    if price is None:
        return None, None
    proceeds = price * lot.quantity
    finrez = proceeds - lot.cost_basis
    if finrez <= 0:
        return finrez, Decimal(0)  # a loss/zero -> no LDV saving to advise
    saving = finrez * Decimal("0.13")
    return finrez, saving


# ----------------------------------------------------------------------------
# per-account run
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconRow:
    """One line of the per-account open-lots vs portfolio reconciliation."""

    ticker: str
    engine_qty: Decimal | None  # None = FIFO oversell (history truncated) for this ticker
    portfolio_qty: Decimal | None  # None = not present in the current portfolio
    matches: bool


@dataclass
class AccountResult:
    """Aggregated engine outputs for one account."""

    account_id: str
    name: str
    opened: str
    on_iis: bool
    n_buys: int
    n_sells: int
    open_lots: list[TaxLot]
    realized: list[RealizedResult]
    realized_ytd_2026: Decimal
    realized_all_time: Decimal
    realized_2025: Decimal
    coupon_ytd: Decimal
    dividend_net_ytd: Decimal
    dividend_gross_ytd: Decimal
    dividend_tax_withheld: Decimal
    fees_total: Decimal
    withheld_tax_total: Decimal
    bond_amortization_total: Decimal
    base_a_ytd: Decimal
    harvestable_unrealized_loss: Decimal
    approaching: list[ApproachingLdv]
    ldv_now: list[TaxLot]
    report: TaxReport
    flags: list[DegradationFlag]
    unknown_op_types: dict[str, int]
    recon: list[ReconRow]
    recon_matches: int
    recon_mismatches: int
    truncated_tickers: list[str]
    bond_redemptions_applied: list[str]
    bond_redemptions_unmatched: list[str]
    fifo_error: str | None = None


def _realized_in_year(realized: Iterable[RealizedResult], year: int) -> Decimal:
    total = Decimal(0)
    for r in realized:
        if r.dispose_date.year == year:
            total += r.realized
    return total


def _coupons_in_year(coupons: Iterable[Operation], year: int) -> list[Operation]:
    return [c for c in coupons if c.op_date.year == year]


def _dividends_in_year(divs: Iterable[Operation], year: int) -> list[Operation]:
    return [d for d in divs if d.op_date.year == year]


def _unrealized_loss_total(acc: ParsedAccount, open_lots: list[TaxLot]) -> Decimal:
    """Sum of SIGNED unrealized losses (<= 0) across open lots at current price."""
    total = Decimal(0)
    for lot in open_lots:
        if not lot.cost_basis_known:
            continue
        price = acc.price_by_ticker.get(lot.ticker)
        if price is None:
            continue
        finrez = price * lot.quantity - lot.cost_basis
        if finrez < 0:
            total += finrez
    return total


def _instrument_key(figi: str, ticker: str) -> str:
    """Stable per-instrument grouping key.

    The ticker is the operator-facing identity that stays constant across the raw
    ``figi`` drift between a BUY and a later redemption (e.g. the Selektel bond
    bought under ``TCS10A1089J4`` but redeemed under ``TCS00A1089J4`` -- both carry
    ticker ``RU000A1089J4``). We therefore group by ticker; only if the ticker is
    empty do we fall back to the figi so a ticker-less row still gets its own group
    (never pooled with a different instrument).
    """
    return ticker or figi


@dataclass(frozen=True)
class _PerFigiFifo:
    """Aggregated per-instrument FIFO result across an account."""

    realized: list[RealizedResult]
    open_lots: list[TaxLot]
    truncated_tickers: list[str]  # instruments whose FIFO oversold (excluded)
    redemptions_applied: list[str]  # bond tickers whose full redemption closed a lot
    redemptions_unmatched: list[str]  # full redemptions with no open lot to close


def _redemption_sell(bond: Operation, qty: Decimal) -> Operation:
    """A bond FULL redemption re-expressed as a SELL at par for the open quantity.

    The cash row carries no quantity, so the disposed quantity is the still-open
    lot quantity of that instrument (a full redemption closes the whole position).
    Par proceeds = the redemption payment; per-unit price = payment / qty (used by
    the engine's proceeds math). No commission on a redemption.
    """
    price = (bond.payment / qty) if qty > 0 else Decimal(0)
    return Operation(
        op_type=OperationType.SELL,
        op_date=bond.op_date,
        figi=bond.figi,
        ticker=bond.ticker,
        payment=bond.payment,
        currency=bond.currency,
        quantity=qty,
        price_per_unit=price,
        commission=Decimal(0),
        cost_basis_known=True,
    )


def _per_figi_fifo(acc: ParsedAccount) -> _PerFigiFifo:
    """Run strict FIFO PER INSTRUMENT (never pooled across figis) and aggregate.

    CR-1: pooling buys+sells across all instruments let a SELL of X consume the
    oldest lot of a DIFFERENT instrument Y. We group by the stable ticker key and
    call the engine per instrument, then aggregate realized results + open lots.

    CR-2: each bond FULL redemption is applied as a SELL at par into that bond's
    own SELL stream (matched by ticker, tolerating the BUY-vs-redemption figi
    drift). A redemption with no matching open lot is flagged, not fatal.

    CR-3: a genuine per-instrument oversell (truncated buy history) raises
    ``FifoMatchError`` for that one instrument only -- we flag it HISTORY_TRUNCATED,
    EXCLUDE it from open lots / LDV / harvest, and continue the rest of the account.
    """
    buys_by_key: dict[str, list[Operation]] = {}
    sells_by_key: dict[str, list[Operation]] = {}
    key_ticker: dict[str, str] = {}
    for op in acc.buys:
        key = _instrument_key(op.figi, op.ticker)
        buys_by_key.setdefault(key, []).append(op)
        key_ticker.setdefault(key, op.ticker or op.figi)
    for op in acc.sells:
        key = _instrument_key(op.figi, op.ticker)
        sells_by_key.setdefault(key, []).append(op)
        key_ticker.setdefault(key, op.ticker or op.figi)

    # index the FULL bond redemptions by the same instrument key.
    redemptions_by_key: dict[str, list[Operation]] = {}
    for bond in acc.bond_repayment_full:
        key = _instrument_key(bond.figi, bond.ticker)
        redemptions_by_key.setdefault(key, []).append(bond)
        key_ticker.setdefault(key, bond.ticker or bond.figi)

    realized: list[RealizedResult] = []
    open_lots: list[TaxLot] = []
    truncated: list[str] = []
    redeemed_applied: list[str] = []
    redeemed_unmatched: list[str] = []

    all_keys = set(buys_by_key) | set(sells_by_key) | set(redemptions_by_key)
    for key in sorted(all_keys):
        tk = key_ticker.get(key, key)
        buys = buys_by_key.get(key, [])
        sells = list(sells_by_key.get(key, []))
        reds = redemptions_by_key.get(key, [])

        # first reconstruct the open lots from trades ONLY, so a full redemption
        # can be sized against the still-open quantity of THIS instrument.
        try:
            pre_open = open_lots_after_match(buys, sells)
        except FifoMatchError:
            truncated.append(tk)
            continue
        pre_qty = sum((lot.quantity for lot in pre_open), Decimal(0))

        # CR-2: apply each full redemption as a SELL at par of the open quantity.
        for bond in reds:
            if pre_qty > 0:
                sells.append(_redemption_sell(bond, pre_qty))
                redeemed_applied.append(tk)
                pre_qty = Decimal(0)  # a full redemption closes the whole position
            else:
                redeemed_unmatched.append(tk)

        try:
            realized.extend(fifo_match(buys, sells))
            open_lots.extend(open_lots_after_match(buys, sells))
        except FifoMatchError:
            truncated.append(tk)

    return _PerFigiFifo(
        realized=realized,
        open_lots=open_lots,
        truncated_tickers=sorted(set(truncated)),
        redemptions_applied=sorted(set(redeemed_applied)),
        redemptions_unmatched=sorted(set(redeemed_unmatched)),
    )


def _reconcile_open_lots(
    acc: ParsedAccount,
    open_lots: list[TaxLot],
    truncated_tickers: list[str],
) -> tuple[list[ReconRow], int, int]:
    """Reconcile per-ticker engine open-lot quantities against the portfolio.

    The portfolio positions carry a ticker + quantity (figi is null in the feed),
    so we key on ticker. Cash-like ruble positions (ticker ``RUB...``) are not
    securities lots and are skipped. A truncated ticker (excluded above) is
    reported as a mismatch with ``engine_qty=None`` so it is never silently
    dropped. Returns (rows, matches, mismatches).
    """
    engine_qty: dict[str, Decimal] = {}
    for lot in open_lots:
        engine_qty[lot.ticker] = engine_qty.get(lot.ticker, Decimal(0)) + lot.quantity

    pf_qty: dict[str, Decimal] = {}
    for pos in acc.portfolio_positions:
        tk = str(pos.get("ticker") or "")
        if not tk or tk.upper().startswith("RUB"):
            continue  # ruble cash position, not a securities lot
        q = _dec(pos.get("quantity"))
        if q is not None:
            pf_qty[tk] = q

    rows: list[ReconRow] = []
    for tk in sorted(set(engine_qty) | set(pf_qty) | set(truncated_tickers)):
        if tk in truncated_tickers:
            rows.append(
                ReconRow(ticker=tk, engine_qty=None, portfolio_qty=pf_qty.get(tk), matches=False)
            )
            continue
        eq = engine_qty.get(tk)
        pq = pf_qty.get(tk)
        is_match = eq is not None and pq is not None and eq == pq
        rows.append(ReconRow(ticker=tk, engine_qty=eq, portfolio_qty=pq, matches=is_match))
    n_match = sum(1 for r in rows if r.matches)
    n_mismatch = len(rows) - n_match
    return rows, n_match, n_mismatch


def run_account(acc: ParsedAccount, today: date, year: int) -> AccountResult:
    """Run the engine for one account and aggregate the outputs."""
    fifo_error: str | None = None

    # CR-1/CR-2/CR-3: FIFO is run PER INSTRUMENT (never pooled), bond FULL
    # redemptions are applied as par disposals, and a per-instrument oversell is
    # flagged + excluded (not fatal to the whole account).
    fifo = _per_figi_fifo(acc)
    realized = fifo.realized
    open_lots = _tag_lots(fifo.open_lots, on_iis=acc.on_iis, russian_by_key=acc.russian_by_key)

    recon, recon_matches, recon_mismatches = _reconcile_open_lots(
        acc, open_lots, fifo.truncated_tickers
    )

    realized_ytd = [r for r in realized if r.dispose_date.year == year]
    coupons_ytd = _coupons_in_year(acc.coupons, year)
    dividends_ytd = _dividends_in_year(acc.dividends_net, year)

    base_a_ytd = realized_ytd_base_a(realized_ytd, coupons_ytd)
    dividend_net_ytd = sum((d.payment for d in dividends_ytd), Decimal(0))
    # gross dividend YTD = net + withheld tax for the SAME year window
    div_tax_ytd = _dividend_tax_ytd(acc, year)
    dividend_gross_ytd = dividend_net_ytd + div_tax_ytd

    approaching = approaching_ldv(acc, open_lots, today)
    ldv_now = [lot for lot in open_lots if ldv_eligible(lot, today)]

    harvest_candidates = _harvest_candidates(acc, open_lots)
    harvestable_loss = sum((c.unrealized_loss for c in harvest_candidates), Decimal(0))

    # gross-dividend rows for the report (base D uses the GROSS estimate, which is
    # the honest higher figure vs the net-only report path).
    gross_div_ops = _gross_dividend_ops(dividends_ytd, dividend_gross_ytd, dividend_net_ytd)

    report = build_report(
        today=today,
        year=year,
        open_lots=open_lots,
        realized_ytd=realized_ytd,
        coupons_ytd=coupons_ytd,
        dividends_ytd=gross_div_ops,
        forward_income=[],  # forward income projection out of scope for this run
        harvest_candidates=[] if acc.on_iis else harvest_candidates,
        history_truncated=False,
        ldv_hypothetical=None,
    )

    flags = list(report.flags)
    _extend_realdata_flags(flags, acc, open_lots, fifo_error)
    _flag_truncated_tickers(flags, acc, fifo.truncated_tickers)
    _flag_bond_redemptions(flags, acc, fifo)
    _flag_redomicile_ldv_credited(flags, acc, open_lots, today)
    _flag_unpriceable_approaching(flags, acc, approaching)

    return AccountResult(
        account_id=acc.account_id,
        name=acc.name,
        opened=acc.opened.isoformat(),
        on_iis=acc.on_iis,
        n_buys=len(acc.buys),
        n_sells=len(acc.sells),
        open_lots=open_lots,
        realized=realized,
        realized_ytd_2026=_realized_in_year(realized, year),
        realized_all_time=sum((r.realized for r in realized), Decimal(0)),
        realized_2025=_realized_in_year(realized, year - 1),
        coupon_ytd=sum((c.payment for c in coupons_ytd), Decimal(0)),
        dividend_net_ytd=dividend_net_ytd,
        dividend_gross_ytd=dividend_gross_ytd,
        dividend_tax_withheld=div_tax_ytd,
        fees_total=acc.fees_total,
        withheld_tax_total=acc.withheld_tax_total,
        bond_amortization_total=acc.bond_amortization_total,
        base_a_ytd=base_a_ytd,
        harvestable_unrealized_loss=harvestable_loss,
        approaching=approaching,
        ldv_now=ldv_now,
        report=report,
        flags=flags,
        unknown_op_types=acc.unknown_op_types,
        recon=recon,
        recon_matches=recon_matches,
        recon_mismatches=recon_mismatches,
        truncated_tickers=fifo.truncated_tickers,
        bond_redemptions_applied=fifo.redemptions_applied,
        bond_redemptions_unmatched=fifo.redemptions_unmatched,
        fifo_error=fifo_error,
    )


def _dividend_tax_ytd(acc: ParsedAccount, year: int) -> Decimal:
    """|DIVIDEND_TAX| withheld in ``year`` (DIVIDEND_TAX rows are dated with their pay).

    Used to gross-up the net dividend base for ``year`` back to its pre-withholding
    figure (gross = net + withheld). Restricted to the same tax-year window as the
    net dividends so the base-D gross estimate is year-consistent.
    """
    return acc.dividend_tax_by_year.get(year, Decimal(0))


def _gross_dividend_ops(
    net_ops: list[Operation],
    gross_total: Decimal,
    net_total: Decimal,
) -> list[Operation]:
    """Scale each net-dividend row up to the gross base for the report.

    The report sums DIVIDEND payments as the taxable base. We gross each row up by
    the account gross/net ratio so the reported base reflects the pre-withholding
    dividend (the honest higher figure), keeping per-row structure for base
    isolation. If net_total is zero, the rows pass through unchanged.
    """
    if net_total <= 0 or gross_total <= net_total:
        return net_ops
    ratio = gross_total / net_total
    return [
        Operation(
            op_type=op.op_type,
            op_date=op.op_date,
            figi=op.figi,
            ticker=op.ticker,
            payment=op.payment * ratio,
            currency=op.currency,
            payment_is_net_estimate=False,
        )
        for op in net_ops
    ]


def _harvest_candidates(acc: ParsedAccount, open_lots: list[TaxLot]) -> list[HarvestCandidate]:
    """Open lots at an unrealized loss (current price < cost) -> harvest candidates.

    Skips IIS lots (own regime) and cost-basis-unknown lots (no basis to compare).
    ``unrealized_loss`` is the SIGNED loss (<= 0) the engine expects.
    """
    if acc.on_iis:
        return []
    cands: list[HarvestCandidate] = []
    for lot in open_lots:
        if not lot.cost_basis_known:
            continue
        price = acc.price_by_ticker.get(lot.ticker)
        if price is None:
            continue
        finrez = price * lot.quantity - lot.cost_basis
        if finrez < 0:
            cands.append(HarvestCandidate(lot=lot, unrealized_loss=finrez))
    return cands


def _extend_realdata_flags(
    flags: list[DegradationFlag],
    acc: ParsedAccount,
    open_lots: list[TaxLot],
    fifo_error: str | None,
) -> None:
    """Append REAL-DATA honest-degradation flags not produced by build_report."""
    if fifo_error is not None:
        flags.append(
            DegradationFlag(
                reason=FlagReason.HISTORY_TRUNCATED,
                detail=(
                    f"{acc.name}: FIFO replay failed ({fifo_error}) -- open-lot / "
                    f"realized reconstruction is INCOMPLETE for this account; the "
                    f"operations history is likely truncated or an oversell appears"
                ),
            )
        )
    if acc.securities_input_flagged:
        flags.append(
            DegradationFlag(
                reason=FlagReason.COST_BASIS_UNKNOWN,
                detail=(
                    f"{acc.name}: {acc.securities_input_flagged} securities-transfer-in "
                    f"lot(s) with no acquisition price/date -- cost basis UNKNOWN, LDV "
                    f"clock not credited (flagged, never fabricated)"
                ),
            )
        )
    # non-Russian open lot -> LDV not credited, needs operator review
    flags.extend(
        DegradationFlag(
            reason=FlagReason.FX_NOT_COMPUTED,
            detail=(
                f"{acc.name}/{lot.ticker}: not classified as a Russian issuer "
                f"(post-2025 FZ 58-FZ) -- LDV NOT credited; verify issuer "
                f"nationality manually"
            ),
        )
        for lot in open_lots
        if not lot.russian_issuer
    )
    if acc.bond_amortization_total > 0:
        flags.append(
            DegradationFlag(
                reason=FlagReason.HISTORY_TRUNCATED,
                detail=(
                    f"{acc.name}: ~{acc.bond_amortization_total} RUB of partial bond "
                    f"amortization (BOND_REPAYMENT) returned as principal -- treated as "
                    f"context, not matched to a lot quantity (no per-event qty in feed)"
                ),
            )
        )


def _flag_truncated_tickers(
    flags: list[DegradationFlag],
    acc: ParsedAccount,
    truncated_tickers: list[str],
) -> None:
    """CR-3: per-instrument oversell -> HISTORY_TRUNCATED, excluded from LDV/harvest.

    Per-instrument FIFO surfaces genuine oversells (a SELL exceeding the known open
    lots, from a truncated buy history) that pooled FIFO masked by borrowing another
    instrument's lot. Each such ticker is flagged and EXCLUDED (its open lots /
    LDV / harvest are not reconstructed); the rest of the account continues.
    """
    flags.extend(
        DegradationFlag(
            reason=FlagReason.HISTORY_TRUNCATED,
            detail=(
                f"{acc.name}/{tk}: SELL(s) exceed the known open lots (oversell) -- "
                f"buy history is truncated for this instrument; EXCLUDED from open "
                f"lots / LDV / harvest (never silently borrowed another lot)"
            ),
        )
        for tk in truncated_tickers
    )


def _flag_bond_redemptions(
    flags: list[DegradationFlag],
    acc: ParsedAccount,
    fifo: _PerFigiFifo,
) -> None:
    """CR-2: surface which bond FULL redemptions closed a lot vs found no open lot."""
    flags.extend(
        DegradationFlag(
            reason=FlagReason.HISTORY_TRUNCATED,
            detail=(
                f"{acc.name}/{tk}: bond FULL redemption applied as a par disposal -- "
                f"the open lot is now CLOSED (no longer shown open); matched by "
                f"ticker across the buy-vs-redemption figi drift"
            ),
        )
        for tk in fifo.redemptions_applied
    )
    flags.extend(
        DegradationFlag(
            reason=FlagReason.HISTORY_TRUNCATED,
            detail=(
                f"{acc.name}/{tk}: bond FULL redemption has NO matching open lot "
                f"(buy history truncated or already disposed) -- flagged, not applied"
            ),
        )
        for tk in fifo.redemptions_unmatched
    )


def _flag_redomicile_ldv_credited(
    flags: list[DegradationFlag],
    acc: ParsedAccount,
    open_lots: list[TaxLot],
    today: date,
) -> None:
    """INFO-2: needs-review flag for a redomiciled foreign structure IF LDV-credited.

    A forced-redomicile structure (FIVE/X5, ENPG/EN+) trades on MOEX in RUB and the
    RUB heuristic treats it as Russian, so it CAN pass ``ldv_eligible``. Its issuer
    nationality is contested for art. 219.1, so if such a lot would ever be credited
    LDV we raise a needs-review flag (never crash, never auto-deny). Currently these
    are sold-out / underwater, so this fires only defensively.
    """
    if acc.on_iis:
        return
    for lot in open_lots:
        is_redomicile = (
            lot.figi.upper() in _REDOMICILE_REVIEW_FIGIS
            or lot.ticker.upper() in _REDOMICILE_REVIEW_TICKERS
        )
        if is_redomicile and ldv_eligible(lot, today):
            label = _REDOMICILE_REVIEW_FIGIS.get(lot.figi.upper(), lot.ticker)
            flags.append(
                DegradationFlag(
                    reason=FlagReason.FX_NOT_COMPUTED,
                    detail=(
                        f"{acc.name}/{lot.ticker}: LDV would be credited to a forced-"
                        f"redomicile structure ({label}) -- issuer nationality is "
                        f"CONTESTED under art. 219.1 / FZ 58-FZ; VERIFY manually before "
                        f"relying on the exemption (not auto-denied)"
                    ),
                )
            )


def _flag_unpriceable_approaching(
    flags: list[DegradationFlag],
    acc: ParsedAccount,
    approaching: list[ApproachingLdv],
) -> None:
    """Flag approaching-LDV lots whose current price is unavailable (no valuation).

    A lot whose ops-feed ticker is not present in the current portfolio (a MOEX
    ticker rename, a foreign-restructured shell, or a fully-sold-then-rebought
    position) cannot be valued -> the LDV exemption cannot be sized. We surface
    the tickers honestly instead of fabricating a finrez.
    """
    missing = sorted({a.ticker for a in approaching if a.est_finrez is None})
    if missing:
        flags.append(
            DegradationFlag(
                reason=FlagReason.COST_BASIS_UNKNOWN,
                detail=(
                    f"{acc.name}: {len(missing)} approaching-LDV lot(s) have no current "
                    f"portfolio price (ticker rename / foreign restructuring / sold-out): "
                    f"{missing} -- LDV finrez NOT sized; verify ticker + issuer manually"
                ),
            )
        )


# ----------------------------------------------------------------------------
# orchestration + printing
# ----------------------------------------------------------------------------


def load_accounts(data_dir: Path) -> list[ParsedAccount]:
    """Parse accounts.json + each <id>_ops.json + <id>_pf.json (readonly)."""
    accounts_path = data_dir / "accounts.json"
    if not accounts_path.exists():
        msg = f"accounts.json not found in {data_dir}"
        raise RealDataError(msg)
    accounts_raw = _load_json(accounts_path)
    if not isinstance(accounts_raw, list):
        msg = "accounts.json must be a list of account objects"
        raise RealDataError(msg)

    parsed: list[ParsedAccount] = []
    for a in accounts_raw:
        acc_id = str(a["id"])
        name = str(a.get("name") or acc_id)
        opened = _parse_date(a["openedDate"]) if a.get("openedDate") else _TODAY
        ops_path = data_dir / f"{acc_id}_ops.json"
        pf_path = data_dir / f"{acc_id}_pf.json"
        ops_raw = _load_json(ops_path) if ops_path.exists() else []
        rows = ops_raw.get("operations", ops_raw) if isinstance(ops_raw, dict) else (ops_raw or [])
        pf = _load_json(pf_path) if pf_path.exists() else None
        parsed.append(parse_account(acc_id, name, opened, rows, pf))
    return parsed


def _fmt(x: Decimal | None) -> str:
    if x is None:
        return "n/a"
    return f"{x.quantize(Decimal('0.01'))}"


def _harvest_offset(r: AccountResult) -> tuple[Decimal, Decimal]:
    """(capped offset, est saving @13%) for an account (0 for IIS / no positive base A).

    The harvestable loss is capped by the account's OWN positive base-A result (RU
    nets per account); the offset comes off the top of base A. All bases here are
    below 2.4M so the marginal band is 13%.
    """
    if r.on_iis:
        return Decimal(0), Decimal(0)
    positive_base_a = max(Decimal(0), r.base_a_ytd)
    total_loss = abs(r.harvestable_unrealized_loss)
    offset = min(total_loss, positive_base_a)
    saving = offset * Decimal("0.13")
    return offset, saving


def print_report(results: list[AccountResult]) -> None:  # noqa: PLR0915 - one linear report
    """Print the aggregated decision-support report (aggregates only)."""
    print("=" * 92)
    print("REAL-PORTFOLIO TAX DECISION-SUPPORT REPORT  --  DECISION-SUPPORT ONLY / NO ORDERS")
    print("=" * 92)
    print(f"today={_TODAY.isoformat()}  year={_YEAR}  accounts={len(results)}")
    print()
    total_open = sum(len(r.open_lots) for r in results)
    total_recon_ok = sum(r.recon_matches for r in results)
    total_recon_bad = sum(r.recon_mismatches for r in results)
    print(f"TOTAL open lots across all accounts: {total_open}")
    print(f"TOTAL open-lots-vs-portfolio: {total_recon_ok} match / {total_recon_bad} mismatch")
    print()
    for r in results:
        tag = "IIS (excluded from LDV/harvest)" if r.on_iis else "taxable brokerage"
        print("-" * 92)
        print(f"ACCOUNT {r.account_id}  {r.name!r}  opened={r.opened}  [{tag}]")
        print(
            f"  trades: {r.n_buys} BUY / {r.n_sells} SELL   open lots: {len(r.open_lots)}"
            f"   fifo_error={r.fifo_error or 'none'}"
        )
        # (a) per-account open-lots vs portfolio reconciliation
        print(
            f"  RECON open-lots vs portfolio: {r.recon_matches} match / "
            f"{r.recon_mismatches} mismatch"
        )
        for row in r.recon:
            if row.matches:
                continue
            eq = "oversell/excluded" if row.engine_qty is None else str(row.engine_qty)
            pq = "absent" if row.portfolio_qty is None else str(row.portfolio_qty)
            print(f"    MISMATCH {row.ticker:16s} engine={eq}  portfolio={pq}")
        if r.bond_redemptions_applied:
            print(f"  bond FULL redemptions applied (lot CLOSED): {r.bond_redemptions_applied}")
        if r.bond_redemptions_unmatched:
            print(f"  bond FULL redemptions UNMATCHED (flagged): {r.bond_redemptions_unmatched}")
        if r.truncated_tickers:
            print(f"  HISTORY_TRUNCATED (oversell, excluded): {r.truncated_tickers}")
        print(
            f"  realized YTD {_YEAR} (base A capital): {_fmt(r.realized_ytd_2026)} RUB   "
            f"(2025 context: {_fmt(r.realized_2025)} RUB)"
        )
        print(
            f"  coupons YTD (base A): {_fmt(r.coupon_ytd)} RUB   "
            f"=> base A YTD (netted): {_fmt(r.base_a_ytd)} RUB"
        )
        print(
            f"  dividends YTD: net {_fmt(r.dividend_net_ytd)} + withheld "
            f"{_fmt(r.dividend_tax_withheld)} = gross ~{_fmt(r.dividend_gross_ytd)} RUB (base D)"
        )
        print(
            f"  fees (context): {_fmt(r.fees_total)} RUB   withheld TAX rows: "
            f"{_fmt(r.withheld_tax_total)} RUB   bond amort: {_fmt(r.bond_amortization_total)} RUB"
        )
        # (c) harvest per account: loss, capped offset, est saving @13%
        offset, hsaving = _harvest_offset(r)
        print(
            f"  harvestable unrealized loss: {_fmt(r.harvestable_unrealized_loss)} RUB   "
            f"positive base A: {_fmt(max(Decimal(0), r.base_a_ytd))} RUB   "
            f"capped offset: {_fmt(offset)} RUB   est saving@13%: {_fmt(hsaving)} RUB"
        )
        print(f"  LDV now: {len(r.ldv_now)}   approaching-LDV: {len(r.approaching)}")
        # (b) approaching-LDV list (buffered eligibility date)
        for a in r.approaching:
            print(
                f"    APPROACHING-LDV {a.ticker} qty={a.quantity} acq={a.acquire_date} "
                f"eligible>={a.cross_date} (3y+buffer) est_finrez={_fmt(a.est_finrez)} "
                f"est_saving@13%={_fmt(a.est_saving)} cost_known={a.cost_basis_known}"
            )
        # (e) needs-review / degradation flags for this account (deduped)
        seen: set[str] = set()
        for fl in r.flags:
            line = f"    FLAG[{fl.reason.value}] {fl.detail}"
            if line in seen:
                continue
            seen.add(line)
            print(line)
    print("-" * 92)
    # portfolio-wide rollups (a few examples only; no raw per-op rows)
    print("PORTFOLIO ROLLUP:")
    all_appr = [(r.name, a) for r in results for a in r.approaching]
    print(f"  approaching-LDV lots (all taxable accounts): {len(all_appr)}")
    sber = [(nm, a) for nm, a in all_appr if a.ticker == "SBER"]
    for nm, a in sber:
        print(
            f"    SBER present: acct={nm!r} qty={a.quantity} acq={a.acquire_date} "
            f"eligible>={a.cross_date} est_finrez={_fmt(a.est_finrez)} "
            f"est_saving@13%={_fmt(a.est_saving)}"
        )
    tot_offset = sum((_harvest_offset(r)[0] for r in results), Decimal(0))
    tot_hsaving = sum((_harvest_offset(r)[1] for r in results), Decimal(0))
    print(
        f"  harvest capped offset (all accounts): {_fmt(tot_offset)} RUB   "
        f"est saving@13%: {_fmt(tot_hsaving)} RUB"
    )
    div_gross = sum((r.dividend_gross_ytd for r in results), Decimal(0))
    div_tax, _ = _ndfl_on(div_gross)
    coupon_a = sum((r.coupon_ytd for r in results), Decimal(0))
    print(
        f"  dividend base D gross (all accounts, YTD): {_fmt(div_gross)} RUB "
        f"-> NDFL ~{_fmt(div_tax)} RUB"
    )
    print(f"  coupon base A (all accounts, YTD): {_fmt(coupon_a)} RUB")
    print("-" * 92)
    print("(Full scope banner + per-item action descriptions available in the engine report.)")


def _ndfl_on(amount: Decimal) -> tuple[Decimal, Decimal]:
    """NDFL on ``amount`` from a zero YTD start (reuses the engine's marginal band)."""
    return ndfl_marginal(max(Decimal(0), amount), Decimal(0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-portfolio tax decision-support (readonly).")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    accounts = load_accounts(args.data_dir)
    results = [run_account(acc, _TODAY, _YEAR) for acc in accounts]
    print_report(results)

    # explicit no-order invariant across every action item of every account
    for r in results:
        assert all(not a.places_order for a in r.report.action_items)


if __name__ == "__main__":
    main()
