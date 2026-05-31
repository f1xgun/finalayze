"""One-shot operator script: enumerate the full MOEX universe -> committed JSON snapshot.

Reuses the TinkoffFetcher gRPC/cert plumbing (Plan 01 fetch_all_* siblings). Live gRPC is
GENERATION-TIME ONLY; the runtime loader (markets/instruments.py, Plan 03) reads the committed
file offline -- no live gRPC at runtime (D-03). The generator REFUSES to write unless every
trading-critical symbol derived from config/segments.py is present (D-04 fail-closed safety
obligation, UNIV-08) and every traded OFZ is YTM-able (UNIV-06 / Pitfall 1).

The required-symbol set is DERIVED from DEFAULT_SEGMENTS (never hardcoded) so it tracks the
segment definitions; TCSG->T is reconciled defensively (Pitfall 2). The token is read from the
environment by TinkoffFetcher only and is NEVER logged or serialized (T-65-04 / T-65-07).

Usage (operator, with .env + certs symlinked -- see project_worktree_moex_retrain_recipe):
    # Export the token into this shell first (do NOT `source .env` -- breaks pydantic), e.g.
    # via awk on the FINALAYZE_TINKOFF_TOKEN line of .env, then:
    uv run python scripts/generate_moex_universe.py --dry-run   # validate + print counts, no write
    uv run python scripts/generate_moex_universe.py             # writes the committed JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

# Ensure project root is importable (config/ lives at the repo root -- MEMORY convention).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog
from config.segments import DEFAULT_SEGMENTS

from finalayze.markets.instruments import InstrumentRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

_log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

# Historical rebrand reconciliation (VERIFIED live; TCS Group -> figi TCS80A107UL4 -> "T").
# No segment emits "TCSG" today (ru_finance already trades "T"); the alias keeps a future
# re-add covered (Pitfall 2).
_ALIAS: dict[str, str] = {"TCSG": "T"}

_PERCENT_SCALE = Decimal(100)

# Committed snapshot location (package-relative committed data asset, like presets/*.yaml).
_DEFAULT_OUT_PATH = (
    Path(_PROJECT_ROOT) / "src" / "finalayze" / "markets" / "data" / "moex_universe.json"
)

_TINKOFF_TOKEN_ENV = "FINALAYZE_TINKOFF_TOKEN"  # noqa: S105 -- env var NAME, not a secret value

# Floating OFZ-PK (SU29...) spread-over-RUONIA values preserved from the hand-list
# (A3: do NOT derive a fixed rate for floaters). Source: DEFAULT_MOEX_OFZ_INSTRUMENTS.
_OFZ_PK_HANDLIST_RATE: dict[str, Decimal] = {
    "SU29007RMFS0": Decimal("1.30"),
    "SU29008RMFS8": Decimal("1.40"),
    "SU29009RMFS6": Decimal("1.50"),
    "SU29010RMFS4": Decimal("1.60"),
}


class _CouponLike(Protocol):
    amount_per_bond: Decimal


class _FetcherLike(Protocol):
    def fetch_all_shares(self) -> list[dict[str, Any]]: ...
    def fetch_all_etfs(self) -> list[dict[str, Any]]: ...
    def fetch_all_bonds(self) -> list[dict[str, Any]]: ...
    def fetch_all_futures(self) -> list[dict[str, Any]]: ...
    def fetch_all_currencies(self) -> list[dict[str, Any]]: ...


# ── Required-symbol derivation + validation (DERIVE, never hardcode) ───────────


def required_symbols() -> set[str]:
    """Derive the trading-critical symbol set from enabled MOEX segments (TCSG->T)."""
    req: set[str] = set()
    for seg in DEFAULT_SEGMENTS:
        if seg.market != "moex" or not seg.enabled:
            continue
        req |= {_ALIAS.get(s, s) for s in seg.symbols}
    return req  # includes OFZ names from ru_ofz_pd / ru_ofz_pk


def traded_ofz_symbols() -> set[str]:
    """The OFZ set needing coupon enrichment = union(ru_ofz_pd, ru_ofz_pk) symbols."""
    ofz: set[str] = set()
    for seg in DEFAULT_SEGMENTS:
        if seg.market != "moex" or not seg.enabled:
            continue
        if seg.instrument_type == "bond":
            ofz |= {_ALIAS.get(s, s) for s in seg.symbols}
    return ofz


def validate(snapshot_symbols: set[str]) -> None:
    """Fail-closed: refuse to write when ANY required symbol is absent (UNIV-08 / D-04)."""
    missing = required_symbols() - snapshot_symbols
    if missing:
        raise SystemExit(
            f"REFUSING to write snapshot — missing required symbols: {sorted(missing)}"
        )


# ── Coupon-rate derivation (Pitfall 1 / UNIV-06) ───────────────────────────────


def derive_coupon_rate(
    pay_one_bond: Decimal,
    coupon_quantity_per_year: int,
    nominal: Decimal,
) -> Decimal:
    """coupon_rate = pay_one_bond * coupon_quantity_per_year / nominal * 100 (annual %)."""
    return pay_one_bond * Decimal(coupon_quantity_per_year) / nominal * _PERCENT_SCALE


# ── Row normalization ───────────────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """Make a single field JSON-serializable: Decimal->str, date/datetime->ISO."""
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "isoformat"):
        result: str = value.isoformat()
        return result
    return value


def _base_row(row: dict[str, Any], instrument_type: str) -> dict[str, Any]:
    """Map a fetcher dict to the snapshot row shape (identity metadata only -- D-05)."""
    currency = row.get("currency")
    return {
        "symbol": row.get("ticker"),
        "market_id": "moex",
        "name": row.get("name"),
        "instrument_type": instrument_type,
        "figi": row.get("figi"),
        "lot_size": row.get("lot") or 1,
        "currency": currency.upper() if isinstance(currency, str) else currency,  # Pitfall 3
        "isin": row.get("isin"),
        "class_code": row.get("class_code"),
        "asset_uid": row.get("asset_uid"),
        "basic_asset": row.get("basic_asset"),
        "expiration_date": _jsonable(row.get("expiration_date")),
    }


def _bond_row(
    row: dict[str, Any],
    coupon_lookup: Callable[[str], list[_CouponLike]],
    traded_ofz: set[str],
) -> dict[str, Any]:
    """A bond snapshot row, coupon-enriched when it is in the traded-OFZ set (UNIV-06)."""
    out = _base_row(row, "bond")
    symbol = row.get("ticker")
    nominal = row.get("nominal")
    face_value = nominal if isinstance(nominal, Decimal) else None
    coupon_qty = row.get("coupon_quantity_per_year")
    floating = bool(row.get("floating_coupon_flag"))

    coupon_rate: Decimal | None = None
    if symbol in traded_ofz:
        if floating and symbol in _OFZ_PK_HANDLIST_RATE:
            # OFZ-PK: preserve hand-list spread over RUONIA (A3), do NOT derive a fixed rate.
            coupon_rate = _OFZ_PK_HANDLIST_RATE[symbol]
        elif face_value is not None and coupon_qty:
            figi = row.get("figi")
            coupons = coupon_lookup(figi) if figi else []
            if coupons:
                pay_one_bond = coupons[0].amount_per_bond
                coupon_rate = derive_coupon_rate(pay_one_bond, int(coupon_qty), face_value)

    out["face_value"] = _jsonable(face_value)
    out["coupon_rate"] = _jsonable(coupon_rate)
    out["coupon_frequency"] = coupon_qty
    out["maturity_date"] = _jsonable(row.get("maturity_date"))
    out["floating_coupon"] = floating
    return out


def build_rows(
    fetcher: _FetcherLike,
    coupon_lookup: Callable[[str], list[_CouponLike]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Enumerate all 5 classes, enrich traded-OFZ coupons, return (rows, per-class counts)."""
    traded_ofz = traded_ofz_symbols()
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []

    shares = fetcher.fetch_all_shares()
    counts["shares"] = len(shares)
    rows.extend(_base_row(r, "stock") for r in shares)

    etfs = fetcher.fetch_all_etfs()
    counts["etfs"] = len(etfs)
    rows.extend(_base_row(r, "etf") for r in etfs)

    bonds = fetcher.fetch_all_bonds()
    counts["bonds"] = len(bonds)
    rows.extend(_bond_row(r, coupon_lookup, traded_ofz) for r in bonds)

    futures = fetcher.fetch_all_futures()
    counts["futures"] = len(futures)
    rows.extend(_base_row(r, "future") for r in futures)

    currencies = fetcher.fetch_all_currencies()
    counts["currencies"] = len(currencies)
    rows.extend(_base_row(r, "currency") for r in currencies)

    return rows, counts


def _assert_ofz_yieldable(rows: list[dict[str, Any]]) -> None:
    """Every traded OFZ row must carry the four YTM fields non-None (UNIV-06)."""
    traded_ofz = traded_ofz_symbols()
    seen = {r["symbol"] for r in rows if r["symbol"] in traded_ofz}
    missing = traded_ofz - seen
    if missing:
        raise SystemExit(
            f"REFUSING to write snapshot — traded OFZ absent from universe: {sorted(missing)}"
        )
    for row in rows:
        if row["symbol"] not in traded_ofz:
            continue
        for field in ("coupon_rate", "coupon_frequency", "face_value", "maturity_date"):
            if row.get(field) is None:
                raise SystemExit(
                    f"REFUSING to write snapshot — traded OFZ {row['symbol']} "
                    f"missing {field} (not YTM-able)"
                )


def build_and_write(
    fetcher: _FetcherLike,
    coupon_lookup: Callable[[str], list[_CouponLike]],
    out_path: Path,
    *,
    dry_run: bool,
) -> None:
    """Enumerate -> enrich -> validate (fail-closed) -> write the committed JSON snapshot."""
    rows, counts = build_rows(fetcher, coupon_lookup)
    snapshot_symbols = {row["symbol"] for row in rows}

    _log.info("moex_universe_enumerated", sdk_universe_counts=counts, total=len(rows))

    validate(snapshot_symbols)  # UNIV-08 / D-04 -- raises SystemExit on any missing symbol
    _assert_ofz_yieldable(rows)  # UNIV-06 -- traded OFZ must be YTM-able

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sdk_universe_counts": counts,
        "instruments": rows,
    }

    if dry_run:
        _log.info("moex_universe_dry_run", would_write=str(out_path), counts=counts)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _log.info("moex_universe_written", path=str(out_path), total=len(rows), counts=counts)


# ── Live driver (operator entrypoint) ─────────────────────────────────────────


def _live_coupon_lookup(fetcher: Any) -> Callable[[str], list[_CouponLike]]:
    """Build a figi-keyed coupon lookup using the SDK directly.

    fetch_bond_coupons() resolves symbol->FIGI via the registry, which is empty at
    generation time -- so call services.instruments.get_bond_coupons(figi=...) directly
    (Pitfall 1 / <interfaces> caveat). Window: a full coupon year from today.
    """
    from finalayze.core.schemas import CouponPayment  # noqa: PLC0415

    def lookup(figi: str) -> list[_CouponLike]:
        now = datetime.now(UTC)
        # One year forward captures at least one upcoming coupon for the rate derivation.
        end = now.replace(year=now.year + 1)

        async def _async() -> list[CouponPayment]:
            services = await fetcher._get_services_async()
            resp = await services.instruments.get_bond_coupons(figi=figi, from_=now, to=end)
            out: list[CouponPayment] = []
            for c in resp.events:
                amount = fetcher._money_to_decimal(c.pay_one_bond)
                coupon_date = (
                    c.coupon_date.date() if hasattr(c.coupon_date, "date") else c.coupon_date
                )
                out.append(
                    CouponPayment(
                        bond_figi=figi,
                        coupon_date=coupon_date,
                        record_date=coupon_date,
                        amount_per_bond=amount,
                        coupon_number=getattr(c, "coupon_number", 0),
                    )
                )
            return out

        result: list[_CouponLike] = fetcher._run_async(_async())
        return result

    return lookup


def main(argv: list[str] | None = None) -> None:
    """Operator entrypoint: construct TinkoffFetcher, enumerate, validate, write."""
    parser = argparse.ArgumentParser(description="Generate the committed MOEX universe snapshot.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate + print counts WITHOUT writing the snapshot file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT_PATH,
        help="Output path for the committed snapshot JSON.",
    )
    args = parser.parse_args(argv)

    token = os.environ.get(_TINKOFF_TOKEN_ENV)
    if not token:
        raise SystemExit(
            f"{_TINKOFF_TOKEN_ENV} is not set. Export it into this shell before running "
            "(do NOT `source .env`). See project_worktree_moex_retrain_recipe."
        )

    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415

    registry = InstrumentRegistry()
    fetcher = TinkoffFetcher(token, registry, sandbox=False)
    try:
        build_and_write(
            fetcher=fetcher,
            coupon_lookup=_live_coupon_lookup(fetcher),
            out_path=args.out,
            dry_run=args.dry_run,
        )
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
