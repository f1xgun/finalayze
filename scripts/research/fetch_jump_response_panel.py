"""Fetch the jump-response panel — token-free, public, READ-ONLY. Run ONCE, commit the snapshot.

Pulls Binance 1-minute klines for BTC/ETH over a multi-year window (no API keys, no orders — real
money is a hard stop) and extracts every large instantaneous shock and its forward path. A "shock"
is a 1-minute return exceeding ``_Z_FLOOR`` times the trailing realised vol (:func:`detect_jumps`,
look-ahead-free). For each shock we store the cumulative return path over the next ``_HORIZON``
minutes — the material a *reactive* trader could still capture after the move has fired.

Detection uses a fast numpy rolling-std pre-scan for candidacy, then the AUTHORITATIVE Decimal
``jump_response_lab`` recomputes the exact z and forward path for each candidate (a few hundred).
The two agree away from the z-gate boundary, and shocks are many-sigma — candidacy never hinges on
float rounding. Only the shocks + their paths are written (not the ~1M raw bars): the snapshot stays
small and ``run_jump_response_study.py`` consumes it deterministically offline.

    uv run python scripts/research/fetch_jump_response_panel.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from finalayze.backtest.jump_response_lab import bar_returns, forward_path

_DIR = Path("results/research/jump_response")
_OUT = _DIR / "jump_panel.json"

_SYMBOLS = ("BTCUSDT", "ETHUSDT")
_START = "2024-01-01"
_END = "2026-01-01"  # exclusive — all of 2024 + 2025 (complete years)
_INTERVAL = "1m"
_VOL_WINDOW = 60  # trailing minutes for the shock denominator
_Z_FLOOR = Decimal(5)  # store every >=5-sigma shock; the study filters up to 6/8
_HORIZON = 120  # forward minutes retained per shock
_PATH_SCALE = (
    1_000_000  # store forward paths as int 0.01-bp units (1e-6 frac) — keeps snapshot small
)
_HTTP_TIMEOUT = 20.0
_MS_PER_MIN = 60_000
_PAGE = 1000
_SLEEP_S = 0.15


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "finalayze-research/1.0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310 (public GET)
        return resp.read().decode("utf-8", "replace")


def _klines_1m(symbol: str) -> tuple[list[int], list[str]]:
    """Return (open_time_ms, close_price_str) for every 1m bar in [_START, _END)."""
    start_ms = int(datetime.fromisoformat(_START).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(_END).replace(tzinfo=UTC).timestamp() * 1000)
    times: list[int] = []
    closes: list[str] = []
    while start_ms < end_ms:
        url = (
            f"https://api.binance.com/api/v3/klines?symbol={symbol}"
            f"&interval={_INTERVAL}&startTime={start_ms}&endTime={end_ms}&limit={_PAGE}"
        )
        batch = json.loads(_get(url))
        if not batch:
            break
        for row in batch:
            times.append(int(row[0]))
            closes.append(str(row[4]))
        start_ms = int(batch[-1][0]) + _MS_PER_MIN
        if len(batch) < _PAGE:
            break
        time.sleep(_SLEEP_S)
    # de-dup on open-time (paging-boundary safety), keep first occurrence, sorted
    seen: dict[int, str] = {}
    for t, c in zip(times, closes, strict=True):
        seen.setdefault(t, c)
    ordered = sorted(seen)
    return ordered, [seen[t] for t in ordered]


def _candidate_jumps(closes_f: np.ndarray) -> list[int]:
    """Fast float pre-scan: indices whose |1m return| >= _Z_FLOOR * trailing(_VOL_WINDOW) std.

    Trailing std ends at i-1 (``.shift(1)``) → excludes the bar's own return (look-ahead-free), and
    the first ``_VOL_WINDOW`` bars are masked so index 0 never leaks into a window (matches lab).
    """
    rets = np.zeros_like(closes_f)
    rets[1:] = closes_f[1:] / closes_f[:-1] - 1.0
    sd = pd.Series(rets).rolling(_VOL_WINDOW).std(ddof=1).shift(1).to_numpy()
    z = np.abs(rets) / sd
    z[: _VOL_WINDOW + 1] = np.nan  # need a full prior window of real returns (i - window >= 1)
    hits = np.nonzero(np.isfinite(z) & (z >= float(_Z_FLOOR)))[0]
    return [int(i) for i in hits]


def _extract(symbol: str, times: list[int], closes: list[str]) -> list[dict[str, object]]:
    levels = [Decimal(c) for c in closes]
    closes_f = np.array([float(c) for c in closes], dtype=np.float64)
    returns = bar_returns(levels)  # authoritative Decimal returns
    # authoritative trailing std for the exact stored z, computed once (float, for reporting only)
    sd_f = pd.Series(closes_f)
    sd_series = (sd_f.pct_change().rolling(_VOL_WINDOW).std(ddof=1).shift(1)).to_numpy()
    n = len(levels)
    jumps: list[dict[str, object]] = []
    for i in _candidate_jumps(closes_f):
        if i + _HORIZON >= n:  # not enough forward data
            continue
        sd = sd_series[i]
        if not np.isfinite(sd) or sd <= 0:
            continue
        z = abs(float(returns[i])) / sd
        path = forward_path(levels, i, _HORIZON)  # Decimal, len _HORIZON+1
        jumps.append(
            {
                "coin": symbol,
                "ts": datetime.fromtimestamp(times[i] / 1000, tz=UTC).isoformat(),
                "sign": 1 if returns[i] > 0 else -1,
                "z": round(float(z), 3),
                "ret": str(returns[i]),
                "path": [round(float(x) * _PATH_SCALE) for x in path],
            }
        )
    return jumps


def main() -> None:
    all_jumps: list[dict[str, object]] = []
    coverage: dict[str, object] = {}
    for sym in _SYMBOLS:
        print(f"fetching 1m klines {sym} {_START}..{_END} ...", flush=True)
        times, closes = _klines_1m(sym)
        print(f"  {len(closes)} bars", flush=True)
        jumps = _extract(sym, times, closes)
        print(f"  {len(jumps)} shocks (z>={_Z_FLOOR})", flush=True)
        all_jumps.extend(jumps)
        coverage[sym] = {
            "bars": len(closes),
            "first": datetime.fromtimestamp(times[0] / 1000, tz=UTC).isoformat() if times else None,
            "last": datetime.fromtimestamp(times[-1] / 1000, tz=UTC).isoformat() if times else None,
            "shocks": len(jumps),
        }

    panel = {
        "meta": {
            "fetched": datetime.now(tz=UTC).date().isoformat(),
            "source": "binance /api/v3/klines interval=1m (USDT-quoted, public read-only)",
            "window": {"start": _START, "end_exclusive": _END},
            "params": {
                "vol_window_min": _VOL_WINDOW,
                "z_floor": str(_Z_FLOOR),
                "horizon_min": _HORIZON,
                "path_scale": _PATH_SCALE,
            },
            "coverage": coverage,
            "disclaimer": (
                "Public read-only market data. No orders, no API keys. Shocks are large 1-minute "
                "moves (news-scale); the forward path is what a reactor could still capture "
                "AFTER the move fired. Real-money execution is a hard stop."
            ),
        },
        "jumps": all_jumps,
    }
    _DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(panel, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {_OUT} ({_OUT.stat().st_size // 1024} KB, {len(all_jumps)} shocks)", flush=True)


if __name__ == "__main__":
    main()
