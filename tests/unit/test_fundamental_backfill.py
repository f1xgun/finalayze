"""RED TDD suite for the one-shot fundamental backfill driver (BACKFILL-H-02/03).

Wave 0: RED. ``scripts.backfill_fundamentals`` does not exist yet (Plan 04), so
collection fails on ``ModuleNotFoundError`` until then.

Asserts the contract: no-lookahead (as_of never the fiscal-quarter end),
idempotent (as_of, symbol) upsert, and non-blocking short-history flagging.
All fixture-driven; no live network (T-63.1-01).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

# Production symbols created in Plan 04 — import expected to FAIL at Wave 0.
from scripts.backfill_fundamentals import (  # noqa: E402
    SHORT_HISTORY_SYMBOLS,
    build_snapshots,
    run_backfill,
)

from finalayze.core.schemas import FundamentalSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterable

_FIXTURES = Path(__file__).parent / "fixtures"

# --- Named constants (ruff PLR2004) ------------------------------------------
_SBER = "SBER"
_LKOH = "LKOH"
_OZON = "OZON"
# Annual fiscal years older than the quarterly window — pure-addition depth (D-03).
_ANNUAL_DEPTH_YEARS = frozenset({2021, 2022, 2023})
# Sentinel pe_ratio values distinguishing an annual vs a quarterly write at a
# forced collision key (the quarterly value must be the last writer / winner).
_ANNUAL_PE_SENTINEL = 11.11
_QUARTERLY_PE_SENTINEL = 22.22
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


def _make_annual_smartlab_fetcher(symbol: str, q_fixture: str, y_fixture: str) -> MagicMock:
    """A SmartlabFundamentalsFetcher stand-in wiring BOTH the quarterly and the
    Plan-01 annual seams (fetch_html_annual / parse_html_annual) to real fixtures.

    NOTE: ``MagicMock(spec=SmartlabFundamentalsFetcher)`` only exposes
    ``fetch_html_annual`` / ``parse_html_annual`` once Plan 01 added them to the
    class — and ``run_backfill`` only calls them once Plan 02 (this plan) wires the
    two-pass build, so the annual tests stay RED until the driver grows.
    """
    from finalayze.data.fetchers.smartlab_fundamentals import (  # noqa: PLC0415
        SmartlabFundamentalsFetcher,
    )

    fetcher = MagicMock(spec=SmartlabFundamentalsFetcher)
    fetcher.fetch_html.return_value = _read(q_fixture)
    fetcher.fetch_html_annual.return_value = _read(y_fixture)
    fetcher.assert_robots_allowed.return_value = None
    real = SmartlabFundamentalsFetcher()
    fetcher.parse_html.side_effect = lambda content, sym=symbol: real.parse_html(content, sym)
    fetcher.parse_html_annual.side_effect = lambda content, sym=symbol: real.parse_html_annual(
        content, sym
    )
    return fetcher


class _RecordingPersistence:
    """Captures ``persist_fundamental_snapshot_async`` writes in order.

    ``writes`` preserves the call sequence (so annual-before-quarterly ordering is
    inspectable); ``final`` keeps the last-writer-wins value per ``(as_of, symbol)``
    key — mirroring the real UNCONDITIONAL ``on_conflict_do_update`` upsert.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[datetime, str, object]] = []
        self.final: dict[tuple[datetime, str], object] = {}
        # The sync wrapper must never be called (63.1 async-only regression).
        self.persist_fundamental_snapshot = MagicMock()

    async def persist_fundamental_snapshot_async(self, snap: FundamentalSnapshot) -> None:
        key = (snap.as_of, snap.symbol)
        self.writes.append((snap.as_of, snap.symbol, snap.pe_ratio))
        self.final[key] = snap.pe_ratio  # last writer wins (mirrors the real upsert)


def _as_of_set(snaps: Iterable[FundamentalSnapshot]) -> set[datetime]:
    return {s.as_of for s in snaps}


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


class TestRunBackfillPersists:
    """Regression: run_backfill must persist via the ASYNC upsert under its own loop.

    The one-shot script has no background async loop, so the sync fire-and-forget
    ``persist_fundamental_snapshot`` raises "async_loop not available" and silently
    drops every row. The driver must drive ``persist_fundamental_snapshot_async``
    under ``asyncio.run`` instead. This test pins that contract — it FAILS on the
    pre-fix code (which called the sync wrapper, awaiting the async path 0 times).
    """

    def test_run_backfill_awaits_async_upsert(self) -> None:
        smartlab = _make_smartlab_fetcher(_SBER, "smartlab_sber_msfo_q.html")
        expected = build_snapshots(_SBER, smartlab, _make_iss_fetcher())
        assert expected, "fixture should yield snapshots"

        persistence = MagicMock()
        persistence.persist_fundamental_snapshot_async = AsyncMock(return_value=None)

        persisted = run_backfill(
            persistence,
            smartlab,
            _make_iss_fetcher(),
            symbols=(_SBER,),
            statement="MSFO",
            dry_run=False,
        )

        # Persisted via the async path exactly once per snapshot...
        assert persistence.persist_fundamental_snapshot_async.await_count == len(expected)
        assert persisted == len(expected)
        # ...and NOT via the loop-requiring sync wrapper (the pre-fix bug path).
        persistence.persist_fundamental_snapshot.assert_not_called()

    def test_dry_run_persists_nothing(self) -> None:
        smartlab = _make_smartlab_fetcher(_SBER, "smartlab_sber_msfo_q.html")
        persistence = MagicMock()
        persistence.persist_fundamental_snapshot_async = AsyncMock(return_value=None)

        persisted = run_backfill(
            persistence,
            smartlab,
            _make_iss_fetcher(),
            symbols=(_SBER,),
            statement="MSFO",
            dry_run=True,
        )

        assert persisted == 0
        persistence.persist_fundamental_snapshot_async.assert_not_awaited()


# --- Wave-2: two-pass annual-first/quarterly-second merge --------------------


def _force_collision_fetcher(symbol: str) -> tuple[MagicMock, datetime]:
    """Build an annual fetcher whose parse side-effects share ONE (as_of, symbol).

    Forces the collision deterministically: the quarterly list and the annual list
    each contain one snapshot at the SAME ``shared_as_of`` but with DIFFERENT
    ``pe_ratio`` sentinels (quarterly = winner). The annual list additionally
    carries deep-history years (2021/2022/2023) absent from the quarterly window,
    proving pure-addition depth (D-03).
    """
    fetcher = _make_annual_smartlab_fetcher(
        symbol, "smartlab_lkoh_msfo_q.html", "smartlab_lkoh_msfo_y.html"
    )
    shared_as_of = datetime(2024, 4, 30, tzinfo=UTC)

    def _snap(as_of: datetime, pe: float) -> FundamentalSnapshot:
        return FundamentalSnapshot(symbol=symbol, as_of=as_of, pe_ratio=pe, currency="RUB")

    quarterly = [
        _snap(datetime(2025, 8, 30, tzinfo=UTC), 5.0),
        _snap(shared_as_of, _QUARTERLY_PE_SENTINEL),  # collision key — quarterly wins
    ]
    annual = [
        _snap(datetime(year, 4, 30, tzinfo=UTC), float(year))
        for year in sorted(_ANNUAL_DEPTH_YEARS)  # 2021/2022/2023 — deep additions
    ]
    annual.append(_snap(shared_as_of, _ANNUAL_PE_SENTINEL))  # collision key — overwritten

    fetcher.parse_html.side_effect = lambda content, sym=symbol: list(quarterly)
    fetcher.parse_html_annual.side_effect = lambda content, sym=symbol: list(annual)
    return fetcher, shared_as_of


class TestQuarterlyWinsOnCollision:
    def test_quarterly_wins_on_collision(self) -> None:
        """On a shared (as_of, symbol), the QUARTERLY value is the last writer (D-02)."""
        smartlab, shared_as_of = _force_collision_fetcher(_LKOH)
        recording = _RecordingPersistence()

        run_backfill(
            recording,  # type: ignore[arg-type]
            smartlab,
            _make_iss_fetcher(),
            symbols=(_LKOH,),
            statement="MSFO",
            dry_run=False,
        )

        key = (shared_as_of, _LKOH)
        # Quarterly is authoritative on the collision key.
        assert recording.final[key] == _QUARTERLY_PE_SENTINEL
        # ...because the annual write to that key preceded the quarterly write.
        annual_idx = recording.writes.index((shared_as_of, _LKOH, _ANNUAL_PE_SENTINEL))
        quarterly_idx = recording.writes.index((shared_as_of, _LKOH, _QUARTERLY_PE_SENTINEL))
        assert annual_idx < quarterly_idx

    def test_idempotent_merged_rerun(self) -> None:
        """Re-running the merged backfill yields the same key set and per-key values."""
        results: list[dict[tuple[datetime, str], object]] = []
        for _ in range(2):
            smartlab, _shared = _force_collision_fetcher(_LKOH)
            recording = _RecordingPersistence()
            run_backfill(
                recording,  # type: ignore[arg-type]
                smartlab,
                _make_iss_fetcher(),
                symbols=(_LKOH,),
                statement="MSFO",
                dry_run=False,
            )
            results.append(dict(recording.final))

        assert results[0].keys() == results[1].keys()
        assert results[0] == results[1]


class TestAnnualDepth:
    def test_annual_depth_added(self) -> None:
        """Annual years older than the quarterly window are pure additions (D-03)."""
        smartlab, _shared = _force_collision_fetcher(_LKOH)
        recording = _RecordingPersistence()

        run_backfill(
            recording,  # type: ignore[arg-type]
            smartlab,
            _make_iss_fetcher(),
            symbols=(_LKOH,),
            statement="MSFO",
            dry_run=False,
        )

        persisted_years = {as_of.year for (as_of, _sym) in recording.final}
        # Deep annual years 2021/2022/2023 appear in the persisted key set...
        assert persisted_years >= _ANNUAL_DEPTH_YEARS
        # ...as keys with NO quarterly counterpart (none of the quarterly as_ofs is in 2021-2023).
        depth_keys = {
            (as_of, sym) for (as_of, sym) in recording.final if as_of.year in {2021, 2022}
        }
        assert depth_keys, "expected deep annual-only keys in the persisted set"


class TestAnnualRobotsGate:
    def test_robots_gate_before_annual_fetch(self) -> None:
        """The /f/y/ robots gate is called BEFORE fetch_html_annual (Y-04)."""
        smartlab = _make_annual_smartlab_fetcher(
            _LKOH, "smartlab_lkoh_msfo_q.html", "smartlab_lkoh_msfo_y.html"
        )
        recording = _RecordingPersistence()

        run_backfill(
            recording,  # type: ignore[arg-type]
            smartlab,
            _make_iss_fetcher(),
            symbols=(_LKOH,),
            statement="MSFO",
            dry_run=False,
        )

        names = [c[0] for c in smartlab.mock_calls]
        y_gate_idx = next(
            i
            for i, c in enumerate(smartlab.mock_calls)
            if c[0] == "assert_robots_allowed" and "/f/y/" in (c.args[0] if c.args else "")
        )
        annual_fetch_idx = names.index("fetch_html_annual")
        assert y_gate_idx < annual_fetch_idx

    def test_annual_persist_uses_async_only(self) -> None:
        """Merged run persists via the async upsert only; the sync wrapper is never called."""
        smartlab = _make_annual_smartlab_fetcher(
            _LKOH, "smartlab_lkoh_msfo_q.html", "smartlab_lkoh_msfo_y.html"
        )
        persistence = MagicMock()
        persistence.persist_fundamental_snapshot_async = AsyncMock(return_value=None)

        persisted = run_backfill(
            persistence,
            smartlab,
            _make_iss_fetcher(),
            symbols=(_LKOH,),
            statement="MSFO",
            dry_run=False,
        )

        assert persistence.persist_fundamental_snapshot_async.await_count == persisted
        assert persisted > 0
        persistence.persist_fundamental_snapshot.assert_not_called()
