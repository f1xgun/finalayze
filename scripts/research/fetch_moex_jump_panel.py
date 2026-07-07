"""Fetch the MOEX jump-response panel — Tinkoff gRPC READONLY (the only sanctioned MOEX source).

The crypto reactive-news cert (PR #318) measured, on the fastest 24/7 market, that a reactive trade
into an already-started 1-minute shock is uncapturable net. This ports the SAME measurement to the
real RUB trading universe — liquid MOEX stocks, benchmark OFZ, and USD/RUB — at 1-minute resolution.

Token-gated, READONLY (no orders — real money is a hard stop). Requires FINALAYZE_TINKOFF_TOKEN
(env or .env) + certs/ (symlinked from the main repo) + GRPC_DNS_RESOLVER=native. 1-minute candles
are capped at ONE day per gRPC request, so we paginate weekday-by-weekday.

MOEX is not 24/7, so the series has SESSION BREAKS (main->evening) and overnight/weekend gaps. We
split each instrument's bars into contiguous runs (``split_runs_on_gaps``) so shock detection and
the forward path never straddle a non-tradeable gap. We also record daily open/close per instrument
so the study can decompose the daily move into the un-tradeable OVERNIGHT GAP vs the intraday
continuous move a reactor could actually chase — the MOEX version of "priced before you can act".

Only the shocks + their forward paths + daily open/close are written (not the ~millions of raw
bars): the snapshot stays small and ``run_moex_jump_study.py`` consumes it deterministically.

    GRPC_DNS_RESOLVER=native uv run python scripts/research/fetch_moex_jump_panel.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from finalayze.backtest.jump_response_lab import bar_returns, forward_path, split_runs_on_gaps
from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher
from finalayze.markets.instruments import build_default_registry

os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

_DIR = Path("results/research/moex_jump")
_OUT = _DIR / "moex_jump_panel.json"

# Universe (probed live for 1-min availability): liquid stocks + benchmark OFZ + USD/RUB.
# ETFs are dropped — their registry FIGIs return 0 candles here, and an index ETF has no
# idiosyncratic news shock anyway (it tracks the basket). Reported honestly in the writeup.
_STOCKS = ("SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "PLZL", "MGNT", "MTSS")
_OFZ = ("SU26238RMFS4", "SU26230RMFS1", "SU26240RMFS0", "SU26243RMFS4", "SU26221RMFS0")
_FX = ("USD000UTSTOM",)
_CLASS = {
    **dict.fromkeys(_STOCKS, "stock"),
    **dict.fromkeys(_OFZ, "ofz"),
    **dict.fromkeys(_FX, "fx"),
}

_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2026, 1, 1, tzinfo=UTC)  # exclusive — all of 2024 + 2025
_VOL_WINDOW = 60
_Z_FLOOR = 5.0
_HORIZON = 120
_MAX_GAP_S = 300  # >5min between bars = session break / overnight → new run
_PATH_SCALE = 1_000_000  # forward paths stored as int 0.01-bp units (1e-6 frac)
# store the forward path only at the horizons the study reads (entry latencies + exit horizons +
# decay checkpoints); half-life is judged on this grid. Keeps the snapshot small.
_PATH_HORIZONS = (0, 1, 2, 5, 15, 30, 60, 120)
_SLEEP_S = 0.12
_RETRIES = 2
_SATURDAY = 5  # weekday() >= 5 is the weekend


def _load_token() -> str:
    t = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "").strip()
    if t:
        return t
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("FINALAYZE_TINKOFF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FINALAYZE_TINKOFF_TOKEN not set")


def _weekdays(start: datetime, end: datetime) -> list[datetime]:
    days, d = [], start
    while d < end:
        if d.weekday() < _SATURDAY:  # Mon-Fri (holidays simply return no candles)
            days.append(d)
        d += timedelta(days=1)
    return days


def _fetch_day(fetcher: TinkoffFetcher, sym: str, day: datetime) -> list[object]:
    for attempt in range(_RETRIES + 1):
        try:
            return fetcher.fetch_candles(sym, day, day + timedelta(days=1), "1m")
        except Exception:
            if attempt == _RETRIES:
                return []
            time.sleep(0.5)
    return []


def _detect_run(levels: list[Decimal]) -> list[tuple[int, int, float]]:
    """(index, sign, z) of >=_Z_FLOOR-sigma shocks in one contiguous run. numpy pre-scan +
    authoritative Decimal returns for the stored z. Look-ahead-free (trailing std excludes bar)."""
    if len(levels) < _VOL_WINDOW + _HORIZON + 2:
        return []
    returns = bar_returns(levels)  # authoritative Decimal
    rf = np.array([float(x) for x in returns], dtype=np.float64)
    sd = pd.Series(rf).rolling(_VOL_WINDOW).std(ddof=1).shift(1).to_numpy()
    out: list[tuple[int, int, float]] = []
    for i in range(_VOL_WINDOW + 1, len(levels) - _HORIZON):
        s = sd[i]
        if not np.isfinite(s) or s <= 0:
            continue
        z = abs(float(returns[i])) / float(s)
        if z >= _Z_FLOOR:
            out.append((i, 1 if returns[i] > 0 else -1, round(z, 3)))
    return out


def _extract(sym: str, stamps: list[int], closes: list[str]) -> list[dict[str, object]]:
    levels_all = [Decimal(c) for c in closes]
    # split at session/overnight gaps, tracking the stamp offset so each run keeps its own levels
    runs: list[list[Decimal]] = split_runs_on_gaps(stamps, levels_all, _MAX_GAP_S)
    cls = _CLASS[sym]
    shocks: list[dict[str, object]] = []
    for run in runs:
        for i, sign, z in _detect_run(run):
            path = forward_path(run, i, _HORIZON)  # Decimal, len _HORIZON+1
            shocks.append(
                {
                    "sym": sym,
                    "cls": cls,
                    "sign": sign,
                    "z": z,
                    "path": [round(float(path[h]) * _PATH_SCALE) for h in _PATH_HORIZONS],
                }
            )
    return shocks


def main() -> None:
    token = _load_token()
    registry = build_default_registry()
    fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)
    days = _weekdays(_START, _END)
    print(f"universe {len(_CLASS)} instruments x {len(days)} weekdays", flush=True)

    all_shocks: list[dict[str, object]] = []
    daily: dict[str, list[list[str]]] = {}  # sym -> [[date, open, close], ...]
    coverage: dict[str, object] = {}
    try:
        for sym, cls in _CLASS.items():
            stamps: list[int] = []
            closes: list[str] = []
            rows: list[list[str]] = []
            for day in days:
                candles = _fetch_day(fetcher, sym, day)
                if not candles:
                    continue
                rows.append([day.date().isoformat(), str(candles[0].open), str(candles[-1].close)])
                for c in candles:
                    stamps.append(int(c.timestamp.timestamp()))
                    closes.append(str(c.close))
                time.sleep(_SLEEP_S)
            shocks = _extract(sym, stamps, closes) if stamps else []
            all_shocks.extend(shocks)
            daily[sym] = rows
            coverage[sym] = {
                "cls": cls,
                "bars": len(closes),
                "days": len(rows),
                "shocks": len(shocks),
            }
            print(
                f"  {sym:14} {cls:5} {len(closes):>7} bars {len(rows):>4} days "
                f"{len(shocks):>4} shocks",
                flush=True,
            )
    finally:
        fetcher.close()

    panel = {
        "meta": {
            "fetched": datetime.now(tz=UTC).date().isoformat(),
            "source": "tinkoff gRPC get_candles interval=1m (MOEX, production readonly)",
            "window": {
                "start": _START.date().isoformat(),
                "end_exclusive": _END.date().isoformat(),
            },
            "params": {
                "vol_window_min": _VOL_WINDOW,
                "z_floor": _Z_FLOOR,
                "horizon_min": _HORIZON,
                "path_scale": _PATH_SCALE,
                "max_gap_s": _MAX_GAP_S,
                "path_horizons": list(_PATH_HORIZONS),
            },
            "coverage": coverage,
            "disclaimer": (
                "MOEX data via Tinkoff gRPC readonly (the only sanctioned MOEX source). No orders, "
                "real-money execution is a hard stop. Shocks are large 1-minute moves within a "
                "continuous session; the forward path is what a reactor could still capture AFTER "
                "the move fired. Latency axis is bars (=minutes for liquid names; approximate when "
                "1-min bars are sparse, e.g. thin OFZ / FX)."
            ),
        },
        "shocks": all_shocks,
        "daily": daily,
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(panel, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {_OUT} ({_OUT.stat().st_size // 1024} KB, {len(all_shocks)} shocks)", flush=True)


if __name__ == "__main__":
    main()
