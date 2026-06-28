"""Fetch the MOEX share candle panel for the active-equity-sleeve experiment.

ONE-OFF data-eng step (active-equity-sleeve R&D). Pulls daily OHLCV for the fixed
liquid MOEX universe via Tinkoff gRPC (the ONLY sanctioned MOEX source — never
yfinance) and writes a committed snapshot so the experiment cert reproduces
offline/deterministically (the allocation-gate snapshot pattern).

Token: read from FINALAYZE_TINKOFF_TOKEN (env) or the repo .env. Run with network
access enabled:

    uv run python scripts/research/fetch_equity_tilt_panel.py \
        --start 2022-01-01 --end 2026-06-29
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import build_default_registry

_LOG = structlog.get_logger(__name__)

# Fixed liquid MOEX universe — mirrors config/segments.py _BOOTSTRAP_SEGMENT_SYMBOLS
# (the reviewable, pinned universe for this diagnostic; drift is intentional-only).
UNIVERSE: list[str] = [
    # ru_energy
    "ROSN",
    "TATN",
    "NVTK",
    "SIBN",
    "TATNP",
    "TRNFP",
    # ru_tech
    "YDEX",
    "OZON",
    "VKCO",
    "HEAD",
    "POSI",
    "ASTR",
    "DIAS",
    "SOFL",
    # ru_finance
    "SBER",
    "T",
    "CBOM",
    "BSPB",
    "MOEX",
    "VTBR",
    "AFKS",
    "RENI",
    # ru_metals
    "GMKN",
    "CHMF",
    "NLMK",
    "MAGN",
    "PLZL",
    "RUAL",
    "MTLR",
    # ru_consumer
    "MGNT",
    "X5",
    "LENT",
    # ru_telecom
    "MTSS",
    "RTKM",
    # ru_utilities
    "HYDR",
    "FEES",
    "MSNG",
    "UPRO",
    # ru_construction
    "PIKK",
    "SMLT",
    # ru_chemicals
    "PHOR",
    "AKRN",
    # ru_transport
    "AFLT",
    "FLOT",
    "NMTP",
]

_DEFAULT_OUT = Path("results/research/equity_tilt/panel_snapshot.json")


def _load_token() -> str:
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "").strip()
    if token:
        return token
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("FINALAYZE_TINKOFF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    msg = "FINALAYZE_TINKOFF_TOKEN not set (env or .env)"
    raise SystemExit(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-06-29")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

    registry = build_default_registry()
    fetcher = TinkoffFetcher(token=_load_token(), registry=registry, sandbox=False)

    panel: dict[str, list[list[str]]] = {}
    dropped: dict[str, str] = {}
    for sym in UNIVERSE:
        try:
            candles = fetcher.fetch_candles(sym, start, end, "1d")
        except Exception as exc:
            dropped[sym] = f"{type(exc).__name__}: {exc}"
            _LOG.warning("equity_tilt_fetch_skip", symbol=sym, error=str(exc))
            continue
        rows = [[c.timestamp.date().isoformat(), str(c.close), str(c.volume)] for c in candles]
        if rows:
            panel[sym] = rows
        else:
            dropped[sym] = "no candles in window"
        _LOG.info("equity_tilt_fetched", symbol=sym, bars=len(rows))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "meta": {
            "start": args.start,
            "end": args.end,
            "universe_requested": UNIVERSE,
            "fetched": sorted(panel),
            "dropped": dropped,
            "source": "tinkoff_grpc_daily",
        },
        "panel": panel,
    }
    args.out.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(panel)}/{len(UNIVERSE)} symbols, dropped={list(dropped)}")


if __name__ == "__main__":
    main()
