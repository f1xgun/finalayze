"""Fetch the iter-2 inflation-linker panel (INFLTR) + equity, token-free.

INFLTR = "Ингосстрах Индекс Инфляционных ОФЗ" (inflation-linked OFZ total-return index) — the
last distinct RUB fixed-income factor (real-rate / inflation) not yet run through the gate.
Token-free ISS index endpoint. NOTE: the series is short/discontinued (~2022-2024); the fetch
records its real span so the gate honestly reports the limited window.

    uv run python scripts/research/fetch_iter2_linker.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.loader import load_mcftr_series

_LOG = structlog.get_logger(__name__)
_OUT = Path("results/research/iter2/linker_panel.json")
_SECIDS = {"infltr_linker": "INFLTR", "equity_mcftrr": "MCFTRR"}
_START = datetime(2022, 1, 1, tzinfo=UTC)
_END = datetime(2026, 6, 11, tzinfo=UTC)


def main() -> None:
    legs: dict[str, list[list[str]]] = {}
    for key, secid in _SECIDS.items():
        rows = load_mcftr_series(secid, _START, _END)
        legs[key] = [[d.isoformat(), str(c)] for d, c in rows]
        _LOG.info("iter2_fetched", key=key, secid=secid, bars=len(rows))
    if not legs["infltr_linker"]:
        msg = "empty INFLTR fetch"
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
