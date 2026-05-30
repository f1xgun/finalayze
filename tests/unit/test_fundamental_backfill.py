"""RED TDD suite for the one-shot fundamental backfill driver (BACKFILL-H-02/03).

Wave 0: RED. ``scripts.backfill_fundamentals`` does not exist yet (Plan 04), so
collection fails on ``ModuleNotFoundError`` until then.

Asserts the contract: no-lookahead (as_of never the fiscal-quarter end),
idempotent (as_of, symbol) upsert, and non-blocking short-history flagging.
All fixture-driven; no live network (T-63.1-01).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Production symbols created in Plan 04 — import expected to FAIL at Wave 0.
from scripts.backfill_fundamentals import (  # noqa: E402
    SHORT_HISTORY_SYMBOLS,
    build_snapshots,
)

from finalayze.core.schemas import FundamentalSnapshot

_FIXTURES = Path(__file__).parent / "fixtures"

# --- Named constants (ruff PLR2004) ------------------------------------------
_SBER = "SBER"
_LKOH = "LKOH"
_OZON = "OZON"
# Fiscal-quarter-end dates that as_of must NEVER equal (the look-ahead trap).
_FISCAL_QUARTER_ENDS = frozenset(
    {
        date(2025, 3, 31),
        date(2025, 6, 30),
        date(2025, 9, 30),
        date(2025, 12, 31),
    }
)


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _make_smartlab_fetcher(symbol: str, fixture: str) -> MagicMock:
    """A SmartlabFundamentalsFetcher stand-in whose parse_html returns fixture snapshots."""
    from finalayze.data.fetchers.smartlab_fundamentals import (  # noqa: PLC0415
        SmartlabFundamentalsFetcher,
    )

    # spec the mock against the real class so ``assert_robots_allowed`` is a
    # recognised attribute (MagicMock otherwise rejects ``assert*`` names).
    fetcher = MagicMock(spec=SmartlabFundamentalsFetcher)
    fetcher.fetch_html.return_value = _read(fixture)
    fetcher.assert_robots_allowed.return_value = None
    # build_snapshots is expected to call parse_html(content, symbol); delegate to the
    # real parser via the production import once Plan 02 lands (RED until then).
    real = SmartlabFundamentalsFetcher()
    fetcher.parse_html.side_effect = lambda content, sym=symbol: real.parse_html(content, sym)
    return fetcher


def _make_iss_fetcher() -> MagicMock:
    iss = MagicMock()
    iss.fetch_dividends.return_value = []
    iss.fetch_issuesize.return_value = None
    iss.reconstruct_market_cap.return_value = None
    return iss


class TestNoLookahead:
    def test_no_lookahead(self) -> None:
        """Every built snapshot's as_of > its fiscal-quarter end (no fiscal stamping)."""
        smartlab = _make_smartlab_fetcher(_LKOH, "smartlab_lkoh_msfo_q.html")
        snapshots = build_snapshots(_LKOH, smartlab, _make_iss_fetcher())

        assert snapshots, "expected at least one snapshot"
        for snap in snapshots:
            assert isinstance(snap, FundamentalSnapshot)
            assert snap.as_of.date() not in _FISCAL_QUARTER_ENDS


class TestIdempotentUpsert:
    def test_upsert_idempotent(self) -> None:
        """Calling the driver twice yields the same (as_of, symbol) keys."""
        smartlab = _make_smartlab_fetcher(_SBER, "smartlab_sber_msfo_q.html")
        iss = _make_iss_fetcher()

        first = build_snapshots(_SBER, smartlab, iss)
        second = build_snapshots(_SBER, smartlab, iss)

        keys_first = sorted((s.as_of, s.symbol) for s in first)
        keys_second = sorted((s.as_of, s.symbol) for s in second)
        assert keys_first == keys_second
        assert all(s.symbol == _SBER for s in first)


class TestShortHistoryFlag:
    def test_short_history_membership(self) -> None:
        """Growth-tech names are flagged short-history (D-03)."""
        assert _OZON in SHORT_HISTORY_SYMBOLS
        for sym in ("VKCO", "CIAN", "YDEX"):
            assert sym in SHORT_HISTORY_SYMBOLS

    def test_short_history_flag_does_not_abort(self) -> None:
        """A short-history symbol with sparse data yields snapshots, does NOT raise."""
        smartlab = _make_smartlab_fetcher(_OZON, "smartlab_lkoh_msfo_q.html")
        iss = _make_iss_fetcher()
        # Must not raise even though OZON is short-history.
        snapshots = build_snapshots(_OZON, smartlab, iss)
        assert isinstance(snapshots, list)


@pytest.mark.parametrize("symbol", ["SBER", "LKOH"])
def test_blue_chips_not_flagged_short(symbol: str) -> None:
    """Blue chips are NOT in the short-history set."""
    assert symbol not in SHORT_HISTORY_SYMBOLS
