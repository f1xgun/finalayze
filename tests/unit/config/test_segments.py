"""Tests for segment configuration validity."""

from __future__ import annotations

from config.segments import DEFAULT_SEGMENTS

_STALE_TICKERS = {"FIVE", "FIXP", "POLY", "YNDX", "HHRU"}


def test_no_stale_tickers_in_segments() -> None:
    """OPS-02: stale/delisted tickers must not appear in any segment."""
    for seg in DEFAULT_SEGMENTS:
        stale_found = set(seg.symbols) & _STALE_TICKERS
        assert not stale_found, (
            f"Segment {seg.segment_id} contains stale tickers: {stale_found}"
        )


def test_ru_tech_contains_head() -> None:
    """OPS-02: HHRU renamed to HEAD on MOEX."""
    ru_tech = next(s for s in DEFAULT_SEGMENTS if s.segment_id == "ru_tech")
    assert "HEAD" in ru_tech.symbols
    assert "HHRU" not in ru_tech.symbols


def test_ru_tech_contains_ydex() -> None:
    """OPS-02: YNDX renamed to YDEX on MOEX."""
    ru_tech = next(s for s in DEFAULT_SEGMENTS if s.segment_id == "ru_tech")
    assert "YDEX" in ru_tech.symbols
    assert "YNDX" not in ru_tech.symbols
