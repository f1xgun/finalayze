"""Batch-fetch MOEX dividend data from T-Invest API and write moex_dividends.yaml.

Fetches dividend events for all unique equity symbols across ru_* segments,
maps them to {ex_date, amount, status} format, applies manual overrides for
known cancelled/reduced dividends, and writes the result as YAML.

Requires FINALAYZE_TINKOFF_TOKEN env var (T-Invest API token).

Usage:
    uv run python scripts/fetch_moex_dividends.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# gRPC env vars MUST be set before importing grpc
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

# Set up gRPC SSL roots if available
_GRPC_ROOTS = Path(_PROJECT_ROOT) / "certs" / "grpc_roots.pem"
if _GRPC_ROOTS.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_GRPC_ROOTS))

from decimal import Decimal

from config.segments import DEFAULT_SEGMENTS
from dotenv import load_dotenv
from t_tech.invest import AsyncClient
from t_tech.invest.schemas import InstrumentIdType

# ── Constants ─────────────────────────────────────────────────────────────

_TBANK_GRPC_TARGET = "invest-public-api.tbank.ru:443"
_NANO_DIVISOR = Decimal(1_000_000_000)

_FETCH_FROM = datetime(2020, 1, 1, tzinfo=UTC)
_FETCH_TO = datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)

_OUTPUT_PATH = (
    Path(_PROJECT_ROOT) / "src" / "finalayze" / "strategies" / "presets" / "moex_dividends.yaml"
)

# Rate limiting: 0.1s delay between symbol fetches (< 600 req/min)
_DELAY_BETWEEN_SYMBOLS = 0.1

# Bond segment IDs to exclude (they are not equities)
_BOND_SEGMENT_IDS = {"ru_ofz_pd", "ru_ofz_pk"}

# ── Known manual overrides ────────────────────────────────────────────────
# These are dividends where the API data does not reflect the actual outcome.
# Format: (symbol, ex_date_str, override_fields)
# - GAZP 2022: Board recommended 52.53 RUB, shareholders rejected in June 2022.
#   Later a smaller dividend of 51.03 RUB was paid (separate record_date in Oct 2022).

_MANUAL_OVERRIDES: list[dict[str, object]] = [
    {
        "symbol": "GAZP",
        "ex_date": "2022-06-30",
        "amount": 52.53,
        "status": "cancelled",
        "comment": "Board recommended 52.53 RUB, shareholders rejected June 2022",
    },
]


def _quotation_to_float(q: object) -> float:
    """Convert Tinkoff Quotation(units, nano) to float."""
    units = getattr(q, "units", 0)
    nano = getattr(q, "nano", 0)
    return float(Decimal(units) + Decimal(nano) / _NANO_DIVISOR)


def _next_business_day(dt: datetime) -> datetime:
    """Advance a datetime to the next business day (skip weekends)."""
    from datetime import timedelta  # noqa: PLC0415

    one_day = timedelta(days=1)
    nxt = dt + one_day
    while nxt.weekday() >= 5:  # noqa: PLR2004  # Saturday=5, Sunday=6
        nxt += one_day
    return nxt


def _collect_ru_equity_symbols() -> list[str]:
    """Collect unique equity symbols from all ru_* segments (excluding bonds)."""
    symbols: set[str] = set()
    for seg in DEFAULT_SEGMENTS:
        if seg.market == "moex" and seg.segment_id not in _BOND_SEGMENT_IDS:
            symbols.update(seg.symbols)
    return sorted(symbols)


async def _fetch_dividends_for_symbol(
    services: object,
    figi: str,
    from_date: datetime,
    to_date: datetime,
) -> list[dict[str, object]]:
    """Fetch dividends for a single FIGI via T-Invest API."""
    response = await services.instruments.get_dividends(  # type: ignore[attr-defined]
        figi=figi,
        from_=from_date,
        to=to_date,
    )
    results: list[dict[str, object]] = []
    for d in response.dividends:
        amount = _quotation_to_float(d.dividend_net)
        # Tinkoff returns last_buy_date; actual ex-div is next trading day
        ex_date = _next_business_day(d.last_buy_date)
        ex_date_str = ex_date.strftime("%Y-%m-%d")

        # T-Invest API does not provide a dividend_type or status field
        # that distinguishes cancelled dividends. All fetched dividends
        # are marked as "paid" by default. Known cancelled/reduced events
        # are applied as manual overrides in post-processing.
        results.append(
            {
                "ex_date": ex_date_str,
                "amount": round(amount, 4),
                "status": "paid",
            }
        )
    return results


async def _resolve_figi(services: object, ticker: str) -> str | None:
    """Resolve a MOEX ticker to FIGI via T-Invest share lookup."""
    for class_code in ("TQBR", "TQTF", "TQPI"):
        try:
            resp = await services.instruments.share_by(  # type: ignore[attr-defined]
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                class_code=class_code,
                id=ticker,
            )
            return resp.instrument.figi  # type: ignore[attr-defined]
        except Exception:
            continue
    return None


async def _main(token: str) -> dict[str, list[dict[str, object]]]:
    """Main async entry point: fetch dividends for all ru_* equity symbols."""
    symbols = _collect_ru_equity_symbols()
    print(f"Collected {len(symbols)} unique MOEX equity symbols from ru_* segments")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Fetching dividends from {_FETCH_FROM.date()} to {_FETCH_TO.date()}...")
    print()

    all_dividends: dict[str, list[dict[str, object]]] = {}
    failed_symbols: list[str] = []

    client = AsyncClient(token, target=_TBANK_GRPC_TARGET)
    async with client as services:
        for i, symbol in enumerate(symbols, 1):
            print(f"  [{i}/{len(symbols)}] Fetching {symbol}...", end=" ")

            # Resolve ticker to FIGI
            figi = await _resolve_figi(services, symbol)
            if figi is None:
                print("SKIP (FIGI not found)")
                failed_symbols.append(symbol)
                await asyncio.sleep(_DELAY_BETWEEN_SYMBOLS)
                continue

            try:
                divs = await _fetch_dividends_for_symbol(
                    services,
                    figi,
                    _FETCH_FROM,
                    _FETCH_TO,
                )
                if divs:
                    all_dividends[symbol] = divs
                    print(f"OK ({len(divs)} events)")
                else:
                    print("OK (0 events)")
            except Exception as exc:
                print(f"ERROR: {exc}")
                failed_symbols.append(symbol)

            await asyncio.sleep(_DELAY_BETWEEN_SYMBOLS)

    if failed_symbols:
        print(f"\nFailed symbols: {', '.join(failed_symbols)}")

    return all_dividends


def _apply_manual_overrides(
    all_dividends: dict[str, list[dict[str, object]]],
) -> None:
    """Apply known manual overrides (cancelled/reduced dividends)."""
    for override in _MANUAL_OVERRIDES:
        symbol = str(override["symbol"])
        ex_date = str(override["ex_date"])
        amount = override["amount"]
        status = str(override["status"])
        comment = override.get("comment", "")

        entries = all_dividends.get(symbol, [])

        # Check if this ex_date already exists
        found = False
        for entry in entries:
            if entry["ex_date"] == ex_date:
                entry["status"] = status
                found = True
                print(f"  Override applied: {symbol} {ex_date} -> status={status}")
                break

        if not found:
            # Add the missing entry
            new_entry: dict[str, object] = {
                "ex_date": ex_date,
                "amount": amount,
                "status": status,
            }
            all_dividends.setdefault(symbol, []).append(new_entry)
            print(f"  Override added: {symbol} {ex_date} amount={amount} status={status}")

        if comment:
            print(f"    Note: {comment}")

    print()


def _sort_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort dividend entries by ex_date."""
    return sorted(entries, key=lambda e: str(e["ex_date"]))


def _write_yaml(
    all_dividends: dict[str, list[dict[str, object]]],
    output_path: Path,
) -> None:
    """Write dividend data to YAML file with header comments."""
    fetch_date = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    lines: list[str] = [
        "# Historical MOEX dividend last-buy-dates for backtesting",
        f"# Source: Tinkoff Invest API (get_dividends), fetched {fetch_date}",
        "# ex_date = last buy date (Tinkoff convention); actual ex-div is next trading day",
        "# status: paid = dividend paid, cancelled = board recommended but shareholders rejected,",
        "#         reduced = paid at lower amount than initially recommended",
    ]

    for symbol in sorted(all_dividends.keys()):
        entries = _sort_entries(all_dividends[symbol])
        lines.append(f"{symbol}:")
        for entry in entries:
            ex_date = entry["ex_date"]
            amount = entry["amount"]
            status = entry["status"]
            lines.append(f'  - {{ex_date: "{ex_date}", amount: {amount}, status: {status}}}')

    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")
    print(f"Written to {output_path}")


def main() -> None:
    """Entry point."""
    load_dotenv()

    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")
    if not token:
        print(
            "ERROR: FINALAYZE_TINKOFF_TOKEN not set. "
            "This script requires a T-Invest API token. "
            "See T-Invest developer portal.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 60)
    print("MOEX Dividend Calendar Fetcher")
    print("=" * 60)
    print()

    # Fetch dividends from API
    all_dividends = asyncio.run(_main(token))

    # Apply manual overrides (cancelled/reduced dividends)
    print("Applying manual overrides...")
    _apply_manual_overrides(all_dividends)

    # Write YAML
    _write_yaml(all_dividends, _OUTPUT_PATH)

    # Print summary
    total_events = sum(len(v) for v in all_dividends.values())
    total_symbols = len(all_dividends)
    cancelled_count = sum(
        1 for entries in all_dividends.values() for e in entries if e.get("status") == "cancelled"
    )
    print()
    print("=" * 60)
    print(f"Summary: {total_events} events across {total_symbols} symbols")
    print(f"  Cancelled: {cancelled_count}")
    print(f"  Paid: {total_events - cancelled_count}")
    print("=" * 60)

    # Warn about T-Invest API limitation
    print()
    print(
        "WARNING: T-Invest API does not distinguish cancelled dividends. "
        "All API-fetched entries are marked as 'paid'. Known cancelled events "
        "are applied via manual overrides. Review the YAML for accuracy."
    )
    print("  Known override: GAZP 2022-06-30 (52.53 RUB, cancelled)")


if __name__ == "__main__":
    main()
