"""Fetch the ZO (replacement-bond) sleeve panel, token-free (beyond-MOEX-edge, Phase B).

All legs from the PUBLIC MOEX ISS REST API — NO Tinkoff token, NO cert:

- ZO      = ``RURPLRUBTR`` (MOEX Replacement-Bond Index, RUB-quoted total return) —
            replacement bonds: USD/FX-linked eurobond successors SETTLED IN RUB on MOEX,
            bypassing Euroclear. Via the INDEX endpoint (``load_mcftr_series`` secid swap).
- CNY-bond = ``RUCNYTR`` (CNY bond total-return index) — the yuan-denominated comparator.
- equity  = ``MCFTRR`` net total-return index.
- FX proxies = ``CNYRUB_TOM`` (durable post-2024 hard-FX proxy) and ``USD000UTSTOM`` (exchange
            USDRUB — CLOSE goes to 0 / halts after the Jun-2024 NCC sanctions, so it only
            covers ~2023-2024H1; kept as a USD cross-check over its live window).

KNOWN STRUCTURAL LIMIT (honest, pre-registered): the ZO / CNY-bond indices both start
2023-01-03 — they POSTDATE the 2022 crash they would hedge (replacement bonds were created
*because* the 2022 freeze trapped eurobonds). So the acute-2022 tail benefit is
UN-BACKTESTABLE; Phase B measures the in-window FX-linkage (USDRUB/CNYRUB beta) and the
diversification vs the all-ruble stack, and reports the tail benefit as a forward-structural
ARGUMENT, never as measured.

    uv run python scripts/research/fetch_zo_panel.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.fetchers.moex_iss import MoexISSFetcher
from finalayze.data.loader import load_mcftr_series

_LOG = structlog.get_logger(__name__)

_OUT = Path("results/research/zo/panel_snapshot.json")
# Index-engine legs (RUB total-return indices) — fetched via the INDEX history path.
_INDEX_SECIDS = {
    "zo_rurplrubtr": "RURPLRUBTR",
    "cnybond_rucnytr": "RUCNYTR",
    "equity_mcftrr": "MCFTRR",
}
# Currency-engine legs (FX proxies) — fetched via the currency/selt CETS path.
_CURRENCY_SECIDS = {"fx_cnyrub": "CNYRUB_TOM", "fx_usdrub": "USD000UTSTOM"}
_START = datetime(2023, 1, 1, tzinfo=UTC)
_END = datetime(2026, 6, 11, tzinfo=UTC)  # exclusive -> last usable bar 2026-06-10


def main() -> None:
    legs: dict[str, list[list[str]]] = {}
    with MoexISSFetcher() as fetcher:
        for key, secid in _CURRENCY_SECIDS.items():
            rows = fetcher.fetch_currency_close_history(secid, _START, _END)
            legs[key] = [[d.isoformat(), str(c)] for d, c in rows]
            _LOG.info("zo_fx_fetched", key=key, secid=secid, bars=len(rows))
    for key, secid in _INDEX_SECIDS.items():
        rows = load_mcftr_series(secid, _START, _END)
        legs[key] = [[d.isoformat(), str(c)] for d, c in rows]
        _LOG.info("zo_index_fetched", key=key, secid=secid, bars=len(rows))

    if not legs["zo_rurplrubtr"] or not legs["equity_mcftrr"]:
        msg = "empty ZO or equity fetch"
        raise SystemExit(msg)

    snapshot = {
        "meta": {
            "source": "moex_iss_rest_public (token-free)",
            "index_secids": _INDEX_SECIDS,
            "currency_secids": _CURRENCY_SECIDS,
            "note": (
                "ZO (RURPLRUBTR) + RUCNYTR start 2023-01-03 — postdate the 2022 crash; the "
                "acute tail is UN-backtestable. Index legs carry the MCFTRR/_parse_history_row "
                "MSK->UTC "
                "T-1 date convention; the cert shifts them +1 to the true date to align with the "
                "true-dated currency FX legs. USD000UTSTOM halts (CLOSE 0) after Jun-2024."
            ),
        },
        "legs": legs,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    print("wrote", _OUT)
    for key in {**_INDEX_SECIDS, **_CURRENCY_SECIDS}:
        bars = legs[key]
        span = f"{bars[0][0]}..{bars[-1][0]}" if bars else "EMPTY"
        print(f"  {key}: {len(bars)} bars [{span}]")


if __name__ == "__main__":
    main()
