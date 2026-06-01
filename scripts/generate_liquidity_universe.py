"""One-shot operator script: rank MOEX shares by RUB turnover -> committed liquid-universe JSON.

Mirrors ``scripts/generate_moex_universe.py`` end to end (Phase-66 Plan-04, LIQ-02/06/09-12):
enumerate the committed 268-share MOEX universe, compute each share's trailing-60d median RUB
turnover via the Layer-2 primitive ``finalayze.markets.liquidity.median_rub_turnover`` (NO
duplicated turnover math), assign each share to a curated sector, take Top-N per sector, validate
FAIL-CLOSED (refuse to write if any curated sector enumerates to 0 names), and write the committed
``src/finalayze/markets/data/moex_liquidity_universe.json`` in EXACTLY the schema the Plan-01
fail-closed loader (``liquidity._load_liquidity_snapshot``) expects:

    {"generated_at": <iso>, "as_of": <iso>,
     "sectors": {<curated-sector>: [<symbol>, ... ranked desc]}}

Live gRPC is GENERATION-TIME ONLY; the runtime selector (``liquidity.select_segment_symbols`` /
``eligible_universe_as_of``) reads the committed file offline -- no live gRPC at runtime (D-03).
The generator REFUSES to write unless every curated sector resolves to >= 1 ranked name (D-04 /
T-66-13 fail-closed safety obligation): each ``fetch_candles`` failure / an empty class MUST
surface, NOT silently write a partial/stale universe.

Sector tags (D-08 / A3): the snapshot has NO sector field, so sectors are assigned from a curated
``_TICKER_SECTOR`` seed (manual seed per A3). The seed is reviewable and deterministic.
``--dry-run`` reports any enumerated share NOT in the seed so the operator can extend the seed
(or supply ``--sectors-file``) before the real generation -- a one-off T-Invest sector pull at
generation may supersede the manual seed later (A3 refinement). Sector VALUES are validated against
``config.segments.SECTOR_TO_SEGMENT`` (the single D-08 source) so a typo'd sector cannot be written.

The token is read from the environment ONLY (``FINALAYZE_TINKOFF_TOKEN``) and is NEVER logged or
serialized (T-66-12 / Phase-65 T-65-04/07 precedent); error logs carry ``error_type`` only.

Usage (operator, with .env + certs symlinked -- see project_worktree_moex_retrain_recipe):
    # Export the token into THIS shell first (do NOT `source .env` -- breaks pydantic), then:
    export GRPC_DNS_RESOLVER=native
    uv run python scripts/generate_liquidity_universe.py --dry-run   # print distribution, no write
    uv run python scripts/generate_liquidity_universe.py             # writes the committed JSON

Quarterly regeneration runbook: re-run the two commands above each quarter; review the printed
per-sector turnover distribution, confirm the chosen N / RUB floor still fit, then regenerate and
commit the refreshed JSON. Record the chosen numbers + the backtest-iteration verdict in
docs/operations/liquidity_universe_runbook.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

# Ensure project root is importable (config/ lives at the repo root -- MEMORY convention).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os

import structlog
from config.segments import SECTOR_TO_SEGMENT

from finalayze.backtest.config import MOEX_2022_BREAK
from finalayze.markets.instruments import build_default_registry
from finalayze.markets.liquidity import (
    _apply_safety_filters,
    median_rub_turnover,
    top_n_per_sector,
)

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

_log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

_TINKOFF_TOKEN_ENV = "FINALAYZE_TINKOFF_TOKEN"  # noqa: S105 -- env var NAME, not a secret value

# Committed snapshot location (package-relative committed data asset, like presets/*.yaml and
# markets/data/moex_universe.json). This is the SAME path the Plan-01 fail-closed loader reads
# (markets/liquidity._LIQ_SNAPSHOT).
_DEFAULT_OUT_PATH = (
    Path(_PROJECT_ROOT) / "src" / "finalayze" / "markets" / "data" / "moex_liquidity_universe.json"
)

# D-02 trailing window for the turnover ranking -- single-sourced from the Layer-2 primitive's
# default so the generator and the runtime selector use the IDENTICAL window (no magic number).
_TURNOVER_WINDOW = 60

# Candle lookback: fetch enough calendar days to clear the 60-TRADING-day window AFTER the
# MOEX_2022_BREAK exclusion + weekends/holidays. ~252 trading days/year -> 365 calendar days
# comfortably yields >= 60 clean trading bars for any continuously-listed name.
_CANDLE_LOOKBACK = timedelta(days=365)

# Curated ticker -> sector seed (D-08 / A3 manual seed). The snapshot has NO sector field, so each
# share is assigned here. Overlapping blue-chips are assigned to their PRIMARY economic sector
# (SBER->banks, LKOH->oil_gas, GMKN->metals_mining); "diversified" (ru_blue_chips) is reserved for
# true conglomerates with no single dominant sector. Sectors are validated against
# SECTOR_TO_SEGMENT below. Extend this (or pass --sectors-file) for shares --dry-run reports as
# unmapped before the real generation. A future one-off T-Invest sector pull may supersede this
# manual seed (A3 refinement) without changing the committed schema.
_TICKER_SECTOR: dict[str, str] = {
    # oil_gas (ru_energy)
    "ROSN": "oil_gas",
    "LKOH": "oil_gas",
    "TATN": "oil_gas",
    "TATNP": "oil_gas",
    "NVTK": "oil_gas",
    "SIBN": "oil_gas",
    "TRNFP": "oil_gas",
    "GAZP": "oil_gas",
    "SNGS": "oil_gas",
    "SNGSP": "oil_gas",
    "BANE": "oil_gas",
    "BANEP": "oil_gas",
    "RNFT": "oil_gas",
    # banks (ru_finance)
    "SBER": "banks",
    "SBERP": "banks",
    "T": "banks",
    "VTBR": "banks",
    "CBOM": "banks",
    "BSPB": "banks",
    "MOEX": "banks",
    "AFKS": "banks",
    "RENI": "banks",
    "SVCB": "banks",
    "SPBE": "banks",
    # metals_mining (ru_metals)
    "GMKN": "metals_mining",
    "CHMF": "metals_mining",
    "NLMK": "metals_mining",
    "MAGN": "metals_mining",
    "PLZL": "metals_mining",
    "RUAL": "metals_mining",
    "MTLR": "metals_mining",
    "MTLRP": "metals_mining",
    "ALRS": "metals_mining",
    "RASP": "metals_mining",
    "SELG": "metals_mining",
    "UGLD": "metals_mining",
    "ENPG": "metals_mining",
    "VSMO": "metals_mining",
    # utilities (ru_utilities)
    "HYDR": "utilities",
    "FEES": "utilities",
    "MSNG": "utilities",
    "UPRO": "utilities",
    "IRAO": "utilities",
    "OGKB": "utilities",
    "TGKA": "utilities",
    "MRKC": "utilities",
    "MRKP": "utilities",
    "ELFV": "utilities",
    # telecom (ru_telecom)
    "MTSS": "telecom",
    "RTKM": "telecom",
    "RTKMP": "telecom",
    "MGTSP": "telecom",
    # consumer (ru_consumer)
    "MGNT": "consumer",
    "X5": "consumer",
    "LENT": "consumer",
    "FIVE": "consumer",
    "BELU": "consumer",
    "ABRD": "consumer",
    "GCHE": "consumer",
    "AGRO": "consumer",
    # transport (ru_transport)
    "AFLT": "transport",
    "FLOT": "transport",
    "NMTP": "transport",
    "FESH": "transport",
    "GLTR": "transport",
    # chemicals (ru_chemicals)
    "PHOR": "chemicals",
    "AKRN": "chemicals",
    "KAZT": "chemicals",
    "NKNC": "chemicals",
    "NKNCP": "chemicals",
    "KZOS": "chemicals",
    # tech (ru_tech)
    "YDEX": "tech",
    "OZON": "tech",
    "VKCO": "tech",
    "HEAD": "tech",
    "POSI": "tech",
    "ASTR": "tech",
    "DIAS": "tech",
    "SOFL": "tech",
    "CIAN": "tech",
    "WUSH": "tech",
    # real_estate (ru_construction)
    "PIKK": "real_estate",
    "SMLT": "real_estate",
    "LSRG": "real_estate",
    "ETLN": "real_estate",
    # diversified (ru_blue_chips) -- true conglomerates only
    "SFIN": "diversified",
}


class _FetcherLike(Protocol):
    def fetch_candles(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = ...
    ) -> list[Candle]: ...
    def close(self) -> None: ...


# ── Sector assignment + validation (DERIVE/seed, never silently drop) ──────────


def _valid_sectors() -> frozenset[str]:
    """Curated valid-sector set = SECTOR_TO_SEGMENT keys (the single D-08 source, V5/IN-05)."""
    return frozenset(SECTOR_TO_SEGMENT.keys())


def _load_sector_overrides(path: Path | None) -> dict[str, str]:
    """Optional operator-supplied ticker->sector overrides (JSON), merged over the seed.

    Lets the operator extend/correct sector tags WITHOUT editing the script (A3). The file is a
    flat ``{"<ticker>": "<sector>"}`` map; every value is validated against the curated set below.
    """
    if path is None:
        return {}
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, TypeError) as exc:
        raise SystemExit(f"--sectors-file unreadable at {path}: {type(exc).__name__}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"--sectors-file must be a JSON object of ticker->sector, got {type(raw)}")
    return {str(k): str(v) for k, v in raw.items()}


def build_sector_map(overrides: dict[str, str]) -> dict[str, str]:
    """Merge the curated seed with operator overrides; validate every sector value (D-08/IN-05)."""
    merged: dict[str, str] = {**_TICKER_SECTOR, **overrides}
    valid = _valid_sectors()
    bad = {sym: sec for sym, sec in merged.items() if sec not in valid}
    if bad:
        raise SystemExit(
            f"REFUSING to write snapshot -- unknown sector(s) in the ticker->sector map: "
            f"{sorted(bad.items())}. Valid sectors (config.segments.SECTOR_TO_SEGMENT keys): "
            f"{sorted(valid)}."
        )
    return merged


# ── Turnover scoring (reuses the Layer-2 primitive -- NO duplicated math) ──────


def _exclude_break_windows(candles: list[Candle]) -> list[Candle]:
    """Drop candles whose timestamp falls inside any MOEX_2022_BREAK window (UTC-aware compare).

    MOEX_2022_BREAK is a tuple of ("YYYY-MM-DD", "YYYY-MM-DD") string ranges
    (backtest/config.py); the 2022 market closure printed no clean prices, so those bars must NOT
    enter the turnover median. Parsed once to UTC-aware datetimes vs Candle.timestamp.
    """
    windows = [
        (
            datetime.fromisoformat(start).replace(tzinfo=UTC),
            datetime.fromisoformat(end).replace(tzinfo=UTC),
        )
        for start, end in MOEX_2022_BREAK
    ]
    return [c for c in candles if not any(start <= c.timestamp <= end for start, end in windows)]


def score_turnover(
    fetcher: _FetcherLike,
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
) -> tuple[dict[str, Decimal], list[str]]:
    """Per-symbol median RUB turnover over the trailing window (reuses median_rub_turnover).

    Returns ``(scores, short_history)`` where ``scores`` maps a symbol to its median RUB turnover
    (only names with >= window CLEAN bars) and ``short_history`` lists names excluded for too few
    bars. A single share's gRPC failure is isolated (logged with error_type only, then skipped) so
    one transient/delisted FIGI cannot abort the whole 268-share enumeration -- the fail-closed
    per-sector check downstream still catches a wholesale empty (auth/cert/DNS) failure.
    """
    scores: dict[str, Decimal] = {}
    short_history: list[str] = []
    for symbol in symbols:
        try:
            candles = fetcher.fetch_candles(symbol, start, end, "1d")
        except Exception as exc:  # operator one-shot: isolate a single share's failure
            _log.warning("turnover_fetch_failed", symbol=symbol, error_type=type(exc).__name__)
            continue
        clean = _exclude_break_windows(candles)
        liq = median_rub_turnover(clean, window=_TURNOVER_WINDOW)
        if liq is None:
            short_history.append(symbol)
            continue
        scores[symbol] = liq
    return scores, short_history


def _print_distribution(
    scores: dict[str, Decimal],
    sector_map: dict[str, str],
    *,
    short_history: list[str],
    unmapped: list[str],
) -> None:
    """Print the per-sector turnover distribution so the operator can pick N / a RUB floor.

    Pure-stdout summary (NOT logged) -- the operator reads this at the Task-2 checkpoint to decide
    N (Top-N per sector), the RUB turnover floor, and the D-11 tolerance %s.
    """
    by_sector: dict[str, list[tuple[str, Decimal]]] = {}
    for sym, sec in sector_map.items():
        if sym in scores:
            by_sector.setdefault(sec, []).append((sym, scores[sym]))

    print("\n=== MOEX share turnover distribution (median daily RUB, trailing 60d) ===")
    for sector in sorted(by_sector):
        ranked = sorted(by_sector[sector], key=lambda kv: kv[1], reverse=True)
        vals = [float(v) for _, v in ranked]
        med = statistics.median(vals) if vals else 0.0
        print(f"\n[{sector}]  scored={len(ranked)}  median={med:,.0f} RUB")
        for sym, turnover in ranked:
            print(f"    {sym:<8} {float(turnover):>18,.0f}")

    if unmapped:
        print(
            f"\n!! {len(unmapped)} enumerated share(s) NOT in the curated sector map "
            f"(extend _TICKER_SECTOR or --sectors-file before generating): {sorted(unmapped)}"
        )
    if short_history:
        print(
            f"\n.. {len(short_history)} share(s) excluded for < {_TURNOVER_WINDOW} clean bars: "
            f"{sorted(short_history)}"
        )
    print(
        "\nDecide and report back: (a) N (Top-N per sector), (b) the RUB turnover floor (if any), "
        "(c) the D-11 tolerance %s for PF / MaxDD / WF-Sharpe.\n"
    )


# ── Fail-closed validation (refuse-to-write) ───────────────────────────────────


def _assert_sectors_non_empty(sectors: dict[str, list[str]]) -> None:
    """Refuse to write if any curated sector resolved to 0 ranked names (T-66-13 / D-04).

    A sector enumerating to 0 names almost always means a wholesale gRPC/auth/cert/DNS failure
    (every fetch_candles raised, so no share scored) -- surface it instead of committing a
    partial/empty universe. Mirrors generate_moex_universe._assert_classes_non_empty.
    """
    empty = sorted(sector for sector, syms in sectors.items() if not syms)
    if empty:
        raise SystemExit(
            "REFUSING to write snapshot -- curated sector(s) resolved to 0 ranked names: "
            f"{empty}. A T-Bank gRPC enumeration likely returned empty (fetch_candles raised for "
            "every share); check the token, the gRPC cert (certs/grpc_roots.pem), and DNS "
            "(GRPC_DNS_RESOLVER=native) before re-running. Never commit a partial universe."
        )


# ── Build + write ──────────────────────────────────────────────────────────────


def build_and_write(
    fetcher: _FetcherLike,
    out_path: Path,
    *,
    top_n: int,
    min_turnover_rub: Decimal | None,
    sector_overrides: dict[str, str] | None = None,
    dry_run: bool,
) -> dict[str, list[str]]:
    """Enumerate -> score turnover -> Top-N per sector -> validate fail-closed -> write JSON.

    Returns the ``sectors`` map written (or that WOULD be written on --dry-run). ``top_n`` and
    ``min_turnover_rub`` are operator-chosen (from the --dry-run distribution at the Task-2
    checkpoint); ``min_turnover_rub`` drops any name below the floor regardless of rank.
    """
    sector_map = build_sector_map(sector_overrides or {})

    registry = build_default_registry()
    shares = registry.list_by_type("moex", "stock")
    symbols = [inst.symbol for inst in shares]
    _log.info("liquidity_universe_enumerated", share_count=len(symbols))

    now = datetime.now(UTC)
    scores, short_history = score_turnover(fetcher, symbols, start=now - _CANDLE_LOOKBACK, end=now)
    unmapped = sorted(sym for sym in scores if sym not in sector_map)

    # Apply the optional RUB sanity floor BEFORE ranking (drop below-floor names regardless rank).
    if min_turnover_rub is not None:
        scores = {sym: v for sym, v in scores.items() if v >= min_turnover_rub}

    if dry_run:
        _print_distribution(scores, sector_map, short_history=short_history, unmapped=unmapped)
        # On dry-run, still surface the proposed Top-N per sector (uses the passed top_n default).
        proposed = top_n_per_sector(scores, sector_map, top_n)
        _log.info(
            "liquidity_universe_dry_run",
            would_write=str(out_path),
            proposed_top_n=top_n,
            min_turnover_rub=str(min_turnover_rub) if min_turnover_rub is not None else None,
            per_sector_counts={s: len(v) for s, v in proposed.items()},
        )
        return proposed

    sectors = top_n_per_sector(scores, sector_map, top_n)
    # Wholesale-failure guard runs on the PRE-safety-filter result: a sector that scored 0 names
    # means a gRPC/auth/cert/DNS failure (every fetch_candles raised) -- refuse (T-66-13 / D-04).
    _assert_sectors_non_empty(sectors)
    # Defense-in-depth: write a snapshot that is ALREADY clean of toxic/sanctioned names and
    # preferred-share duplicates (the SAME universal filter the runtime selector re-applies, so
    # the committed file and the live selection agree). Single source: liquidity._apply_safety_
    # filters. NOTE: dedup is per-sector here -- a preferred share is dropped only when its common
    # is in the SAME sector's ranked list (e.g. SBERP dropped vs SBER in banks).
    filtered = {sector: _apply_safety_filters(syms) for sector, syms in sectors.items()}
    # A sector emptied ONLY by the safety filter (every liquid name is toxic/sanctioned, e.g.
    # utilities whose sole liquid name IRAO is sanctioned) is a legitimate "no tradeable non-toxic
    # name" -- DROP it (warn), do NOT refuse. (Distinct from the wholesale-failure case above.)
    dropped = sorted(sector for sector, syms in filtered.items() if not syms)
    sectors = {sector: syms for sector, syms in filtered.items() if syms}
    if dropped:
        _log.warning("liquidity_universe_sector_dropped_all_toxic", sectors=dropped)
    if not sectors:
        raise SystemExit(
            "REFUSING to write snapshot -- every sector emptied after safety filtering "
            "(no tradeable non-toxic name in any sector)."
        )

    payload = {
        "generated_at": now.isoformat(),
        "as_of": now.isoformat(),
        "sectors": sectors,
        # Provenance (not consumed by the loader; aids the quarterly runbook review).
        "params": {
            "top_n": top_n,
            "min_turnover_rub": str(min_turnover_rub) if min_turnover_rub is not None else None,
            "window": _TURNOVER_WINDOW,
            "share_count": len(symbols),
            "scored_count": len(scores),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _log.info(
        "liquidity_universe_written",
        path=str(out_path),
        sectors=len(sectors),
        per_sector_counts={s: len(v) for s, v in sectors.items()},
    )
    return sectors


# ── Live driver (operator entrypoint) ──────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Operator entrypoint: construct TinkoffFetcher, score turnover, validate, write."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the committed MOEX liquid-universe snapshot "
            "(Top-N per sector by RUB turnover)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-sector turnover distribution + proposed counts WITHOUT writing.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N highest-turnover names per sector (operator-chosen from the --dry-run dist).",
    )
    parser.add_argument(
        "--min-turnover-rub",
        type=Decimal,
        default=None,
        help="Optional RUB turnover floor: drop any name below this median regardless of rank.",
    )
    parser.add_argument(
        "--sectors-file",
        type=Path,
        default=None,
        help="Optional JSON ticker->sector overrides merged over the curated seed.",
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

    overrides = _load_sector_overrides(args.sectors_file)

    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415

    registry = build_default_registry()
    fetcher = TinkoffFetcher(token, registry, sandbox=False)
    try:
        build_and_write(
            fetcher=fetcher,
            out_path=args.out,
            top_n=args.top_n,
            min_turnover_rub=args.min_turnover_rub,
            sector_overrides=overrides,
            dry_run=args.dry_run,
        )
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
