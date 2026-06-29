"""Fetch Tinkoff candles for IMOEX constituents missing from the base panel (step 3).

The low-vol-blend tilt is judged against a REAL cap-weight baseline built from the
committed IMOEX index-weight snapshot. That needs candles for every historical
IMOEX constituent — including big names absent from the #299 base panel
(GAZP/LKOH/SNGS/...). This one-off extends the base candle panel with the missing,
still-listed constituents via Tinkoff gRPC (the only sanctioned MOEX source).
De-listed/renamed tickers (POLY/QIWI/old YNDX/TCSG/...) simply fail to fetch and
are recorded as dropped — both the baseline and the tilt exclude them equally, so
the relative verdict stays sound; per-date index-weight COVERAGE is reported so the
baseline's faithfulness is auditable.

    uv run python scripts/research/fetch_lowvol_candles.py
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import build_default_registry

_LOG = structlog.get_logger(__name__)

_BASE_PANEL = Path("results/research/equity_tilt/panel_snapshot.json")
_WEIGHTS = Path("results/research/lowvol/index_weights_snapshot.json")
_OUT = Path("results/research/lowvol/candle_panel.json")


def _load_token() -> str:
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "").strip()
    if token:
        return token
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("FINALAYZE_TINKOFF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    msg = "FINALAYZE_TINKOFF_TOKEN not set"
    raise SystemExit(msg)


def main() -> None:
    base = json.loads(_BASE_PANEL.read_text(encoding="utf-8"))["panel"]
    weights = json.loads(_WEIGHTS.read_text(encoding="utf-8"))["weights"]
    union = {t for per in weights.values() for t in per}
    missing = sorted(union - set(base))
    _LOG.info("lowvol_missing_constituents", n=len(missing), names=missing)

    registry = build_default_registry()
    fetcher = TinkoffFetcher(token=_load_token(), registry=registry, sandbox=False)
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2026, 6, 29, tzinfo=UTC)

    panel: dict[str, list[list[str]]] = dict(base)  # start from the base 45 names
    fetched: list[str] = []
    dropped: dict[str, str] = {}
    for sym in missing:
        try:
            candles = fetcher.fetch_candles(sym, start, end, "1d")
        except Exception as exc:
            dropped[sym] = f"{type(exc).__name__}"
            _LOG.warning("lowvol_fetch_skip", symbol=sym, error=str(exc))
            continue
        rows = [[c.timestamp.date().isoformat(), str(c.close), str(c.volume)] for c in candles]
        if rows:
            panel[sym] = rows
            fetched.append(sym)
        else:
            dropped[sym] = "no candles"

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "base_panel": str(_BASE_PANEL),
                    "added": sorted(fetched),
                    "dropped": dropped,
                    "total_symbols": len(panel),
                    "source": "tinkoff_grpc_daily + #299 base",
                },
                "panel": panel,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {_OUT}: {len(panel)} symbols (+{len(fetched)} added, "
        f"{len(dropped)} dropped: {sorted(dropped)})"
    )


if __name__ == "__main__":
    main()
