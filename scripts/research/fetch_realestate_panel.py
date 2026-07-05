"""Fetch the real-estate sleeve panel, token-free (beyond-MOEX-edge, Phase C).

All legs from the PUBLIC MOEX ISS REST API — NO Tinkoff token, NO cert:

- real_estate = ``MREDC`` (MOEX/DomClick Moscow residential price index, RUB/sq.m) via the
                INDEX endpoint (``load_mcftr_series`` secid swap). PRICE ONLY — it captures
                residential sale-price appreciation, NOT rental income; the cert adds a
                labelled net-rental overlay to represent real estate's income component.
- equity      = ``MCFTRR`` net total-return index — the leg real estate is carved from.

TWO STRUCTURAL LIMITS (honest, pre-registered — the cert reports both prominently):

  1. SMOOTHING ARTIFACT. MREDC updates ~WEEKLY (≈52 bars/yr), not daily, and is a
     transaction/appraisal index — so its measured volatility and drawdown are
     STRUCTURALLY UNDERSTATED vs a real traded asset. The investable form (a rental
     ZPIF) carries real market volatility + illiquidity + wide bid/ask + 1-3%/yr fees
     that MREDC hides. Any low-MaxDD reading here is partly an artifact, never a free lunch.

  2. POLICY-DRIVEN APPRECIATION. The ~+8.5%/yr residential price rise over 2022-2026 was
     largely driven by subsidised mortgages (lgotnaya ipoteka), a policy now wound down —
     so the historical appreciation is NOT a forward-looking expectation.

    uv run python scripts/research/fetch_realestate_panel.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.loader import load_mcftr_series

_LOG = structlog.get_logger(__name__)

_OUT = Path("results/research/realestate/panel_snapshot.json")
# Index-engine legs (RUB indices) — fetched via the INDEX history path. Both carry the
# MCFTRR/_parse_history_row MSK-midnight -> UTC T-1 date convention; the cert shifts both
# +1 day to the true ISS trade date.
_INDEX_SECIDS = {
    "real_estate_mredc": "MREDC",
    "equity_mcftrr": "MCFTRR",
}
_START = datetime(2022, 1, 1, tzinfo=UTC)
_END = datetime(2026, 6, 11, tzinfo=UTC)  # exclusive -> last usable bar 2026-06-10


def main() -> None:
    legs: dict[str, list[list[str]]] = {}
    for key, secid in _INDEX_SECIDS.items():
        rows = load_mcftr_series(secid, _START, _END)
        legs[key] = [[d.isoformat(), str(c)] for d, c in rows]
        _LOG.info("realestate_index_fetched", key=key, secid=secid, bars=len(rows))

    if not legs["real_estate_mredc"] or not legs["equity_mcftrr"]:
        msg = "empty MREDC or equity fetch"
        raise SystemExit(msg)

    snapshot = {
        "meta": {
            "source": "moex_iss_rest_public (token-free)",
            "index_secids": _INDEX_SECIDS,
            "note": (
                "MREDC = DomClick Moscow residential price index, PRICE ONLY (no rent) and "
                "~WEEKLY (smoothed) — measured volatility/drawdown are structurally understated "
                "vs a traded asset; the investable rental-ZPIF wrapper carries illiquidity + "
                "1-3%/yr fees MREDC hides. The ~+8.5%/yr appreciation was largely subsidised-"
                "mortgage (lgotnaya ipoteka) driven — NOT a forward expectation. Index legs "
                "carry the MCFTRR MSK->UTC T-1 date convention; the cert shifts them +1 day."
            ),
        },
        "legs": legs,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    print("wrote", _OUT)
    for key in _INDEX_SECIDS:
        bars = legs[key]
        span = f"{bars[0][0]}..{bars[-1][0]}" if bars else "EMPTY"
        print(f"  {key}: {len(bars)} bars [{span}]")


if __name__ == "__main__":
    main()
