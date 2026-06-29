"""Fetch the gold-sleeve panel (GLDRUB spot + MCFTRR net equity), token-free (Phase A).

Both legs come from the PUBLIC MOEX ISS REST API — NO Tinkoff token, NO cert (the
"MOEX data = Tinkoff gRPC only" invariant governs INSTRUMENT candles, not the public
ISS index/currency statistics):

- gold = ``GLDRUB_TOM`` daily CLOSE on the currency/selt CETS board
  (:meth:`MoexISSFetcher.fetch_currency_close_history`) — spot gold in RUB, which kept
  trading through the 27-day 2022 equity halt;
- equity = ``MCFTRR`` net total-return index (:func:`load_mcftr_series`).

The deposit leg is NOT fetched: it is accrued at runtime from the committed CBR
key-rate archive (look-ahead-safe). Writes a committed snapshot so the cert
reproduces deterministically offline.

    uv run python scripts/research/fetch_gold_panel.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from finalayze.data.fetchers.moex_iss import MoexISSFetcher
from finalayze.data.loader import load_mcftr_series

_LOG = structlog.get_logger(__name__)

_OUT = Path("results/research/gold/panel_snapshot.json")
_GOLD_SECID = "GLDRUB_TOM"
_EQUITY_SECID = "MCFTRR"  # net (after-tax) MOEX total-return index
# MCFTRR begins 2022-01; GLDRUB has deep history, but the binding window starts with
# equity. End clamps to the allocation-gate binding end (look-ahead clamp).
_START = datetime(2022, 1, 1, tzinfo=UTC)
_END = datetime(2026, 6, 11, tzinfo=UTC)  # exclusive -> last usable bar 2026-06-10


def main() -> None:
    with MoexISSFetcher() as fetcher:
        gold = fetcher.fetch_currency_close_history(_GOLD_SECID, _START, _END)
    equity = load_mcftr_series(_EQUITY_SECID, _START, _END)
    _LOG.info("gold_panel_fetched", gold_bars=len(gold), equity_bars=len(equity))
    if not gold or not equity:
        msg = f"empty fetch: gold={len(gold)} equity={len(equity)}"
        raise SystemExit(msg)

    snapshot = {
        "meta": {
            "source": "moex_iss_rest_public (token-free)",
            "gold_secid": _GOLD_SECID,
            "gold_board": "CETS (currency/selt)",
            "equity_secid": _EQUITY_SECID,
            "note": (
                "deposit leg accrued at runtime from the committed CBR key-rate archive; "
                "OFZ floater intentionally excluded (no pre-2023 data, settled in Phase 76)"
            ),
        },
        "window": {
            "gold_start": gold[0][0].isoformat(),
            "gold_end": gold[-1][0].isoformat(),
            "equity_start": equity[0][0].isoformat(),
            "equity_end": equity[-1][0].isoformat(),
        },
        "legs": {
            "gold_gldrub": [[d.isoformat(), str(c)] for d, c in gold],
            "equity_mcftrr": [[d.isoformat(), str(c)] for d, c in equity],
        },
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    print(
        f"wrote {_OUT}: gold {len(gold)} bars [{gold[0][0]}..{gold[-1][0]}], "
        f"equity {len(equity)} bars [{equity[0][0]}..{equity[-1][0]}]"
    )


if __name__ == "__main__":
    main()
