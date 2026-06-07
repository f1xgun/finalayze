"""Committed dividend-schedule loader/repackager (ACCT-01 / D-13 / D-14).

Reads the committed ``moex_dividends.yaml`` snapshot (the source-of-record, built
at snapshot time from ``TinkoffFetcher.fetch_dividends`` / MOEX ISS -- never fetched
at backtest runtime, D-13) and repackages it into an ``{(symbol, ex_date): gross}``
date index. Per-bar accrual is then O(held positions) -- a held name's dividend is a
single dict lookup keyed by ``(symbol, bar_date)`` (avoids the per-name nested-loop
trap, Pitfall 322 / D-14).

Honesty rules:
- ``status != "paid"`` events (cancelled/reduced -- e.g. GAZP 2022-06-30) are SKIPPED
  at load time. T-Invest does NOT flag cancelled dividends, so the committed ``status``
  field is the source of truth (mandatory honesty, ACCT-01).
- A corrupt/unparseable/malformed file raises ``ConfigurationError`` (fail-closed, no
  fallback to a stale list, Pattern 4). A held symbol simply absent from the snapshot
  legitimately yields no entries -- that is not an error (a name with no dividend
  history accrues nothing).

The index stores GROSS amounts; netting (NDFL band/floor) happens at accrual time via
the L0 helper so the cross-sleeve cumulative tax is applied in one place (Pattern 1).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog
import yaml

from finalayze.core.exceptions import ConfigurationError

_LOGGER = structlog.get_logger(__name__)

# Committed snapshot path. ``dividend_schedule.py`` lives at ``src/finalayze/backtest/``;
# the YAML at ``src/finalayze/strategies/presets/`` -- ``.parent.parent`` -> ``src/finalayze``.
_DIVIDENDS_PATH = (
    Path(__file__).resolve().parent.parent / "strategies" / "presets" / "moex_dividends.yaml"
)

_PAID_STATUS = "paid"


def load_dividend_schedule(path: Path | None = None) -> dict[tuple[str, date], Decimal]:
    """Build a ``{(symbol, ex_date): gross_amount}`` index from the committed YAML.

    Skips ``status != "paid"`` (cancelled/reduced) events -- honoring the committed
    status field is mandatory honesty (T-Invest does not flag cancelled dividends;
    GAZP 2022-06-30 is absent). A corrupt/unparseable/malformed file raises
    ``ConfigurationError`` (fail-closed). A held symbol absent from the snapshot
    legitimately yields no entries (not an error).

    Args:
        path: Optional override for the snapshot location (defaults to the committed
            ``moex_dividends.yaml``).

    Returns:
        Mapping of ``(symbol, ex_date)`` to the GROSS dividend amount per share
        (a ``Decimal``). Netting happens at accrual time via the L0 NDFL helper.

    Raises:
        ConfigurationError: if the snapshot file is missing, unparseable, or not a
            top-level mapping.
    """
    target = path or _DIVIDENDS_PATH
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"dividend snapshot missing/corrupt at {target}: {exc}"
        raise ConfigurationError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"dividend snapshot malformed (expected a mapping) at {target}"
        raise ConfigurationError(msg)

    index: dict[tuple[str, date], Decimal] = {}
    for symbol, events in raw.items():
        for ev in events:
            if ev.get("status", _PAID_STATUS) != _PAID_STATUS:
                # Skip cancelled/reduced -- mandatory honesty (GAZP 2022-06-30).
                continue
            ex = date.fromisoformat(ev["ex_date"])
            # Decimal(str(amount)): the YAML amount is a float; round-tripping via str
            # avoids float-binary contamination of the kopeck-exact golden test.
            amount = Decimal(str(ev["amount"]))
            key = (symbol, ex)
            index[key] = index.get(key, Decimal(0)) + amount

    _LOGGER.debug("dividend_schedule_loaded", path=str(target), entries=len(index))
    return index
