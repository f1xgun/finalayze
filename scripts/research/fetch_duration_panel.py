"""Fetch the duration-regime panel (fixed vs floating OFZ vs equity), token-free.

Answers the operator's question: bonds currently yield 15-16% while the best 3-month deposit is
~14% — does locking a FIXED-coupon bond beat the falling deposit? The live SAA holds the FLOATER
(RUFLBITR / ОФЗ-ПК, tracks the key rate ~1:1 -> falls with the deposit in easing); it does NOT hold
FIXED-coupon ОФЗ-ПД (RGBITR), which LOCKS the yield and gains price as rates fall. This panel lets
the cert compare deposit vs floater vs fixed-coupon OFZ per rate regime.

    uv run python scripts/research/fetch_duration_panel.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.loader import load_mcftr_series

_LOG = structlog.get_logger(__name__)
_OUT = Path("results/research/duration_regimes/panel_snapshot.json")
_SECIDS = {
    "ofz_fixed_rgbitr": "RGBITR",  # ОФЗ-ПД fixed-coupon TR index (duration / yield-lock)
    "ofz_floater_ruflbitr": "RUFLBITR",  # ОФЗ-ПК floater TR index (tracks the key)
    "equity_mcftrr": "MCFTRR",
}
_START = datetime(2022, 1, 1, tzinfo=UTC)
_END = datetime(2026, 6, 11, tzinfo=UTC)


def main() -> None:
    legs: dict[str, list[list[str]]] = {}
    for key, secid in _SECIDS.items():
        rows = load_mcftr_series(secid, _START, _END)
        legs[key] = [[d.isoformat(), str(c)] for d, c in rows]
        _LOG.info("duration_fetched", key=key, secid=secid, bars=len(rows))
    if not legs["ofz_fixed_rgbitr"] or not legs["ofz_floater_ruflbitr"]:
        msg = "empty OFZ fetch"
        raise SystemExit(msg)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(
        json.dumps(
            {
                "meta": {
                    "source": "moex_iss_rest_public (token-free)",
                    "secids": _SECIDS,
                    "note": "index legs carry the MSK T-1 convention; cert shifts +1 to true date.",
                },
                "legs": legs,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print("wrote", _OUT)
    for key, bars in legs.items():
        span = f"{bars[0][0]}..{bars[-1][0]}" if bars else "EMPTY"
        print(f"  {key}: {len(bars)} bars [{span}]")


if __name__ == "__main__":
    main()
