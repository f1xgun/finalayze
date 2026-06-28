"""MOEX data-integrity regression tests (audit 2026-06-28).

Two findings:
  * HIGH  -- scripts/training/data_loader.py fell back to yfinance for MOEX
            (ru_*) tickers when Tinkoff returned empty, violating the hard
            "MOEX = Tinkoff gRPC only" invariant.
  * MEDIUM -- ml/features/fundamental.py attributed the globally-latest snapshot's
            fundamentals to EVERY symbol (cross-symbol contamination), because it
            picked the target from segment-wide data without the scored symbol.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from finalayze.core.schemas import FundamentalSnapshot, MoexMarketData
from finalayze.ml.features.fundamental import compute_fundamental_features

# ── HIGH: yfinance must never serve MOEX tickers ────────────────────────────


def _patch_db_empty(monkeypatch) -> None:
    async def _empty(*_a: object, **_k: object) -> list:
        return []

    monkeypatch.setattr("scripts.training.data_loader.fetch_from_db", _empty)


def test_moex_segment_failclosed_never_calls_yfinance(monkeypatch) -> None:
    from scripts.training import data_loader as dl

    _patch_db_empty(monkeypatch)
    monkeypatch.setattr(dl, "fetch_tinkoff_candles", lambda _symbol: [])
    yf = MagicMock(name="YFinanceFetcher")
    monkeypatch.setattr(dl, "YFinanceFetcher", yf)

    result = dl.fetch_symbol_candles("SBER", "moex", MagicMock(), segment_id="ru_blue_chips")

    assert result == []
    yf.assert_not_called()  # the whole point: no yfinance for MOEX


def test_moex_by_market_id_without_segment_is_still_tinkoff_only(monkeypatch) -> None:
    from scripts.training import data_loader as dl

    _patch_db_empty(monkeypatch)
    monkeypatch.setattr(dl, "fetch_tinkoff_candles", lambda _symbol: [])
    yf = MagicMock(name="YFinanceFetcher")
    monkeypatch.setattr(dl, "YFinanceFetcher", yf)

    result = dl.fetch_symbol_candles("GAZP", "moex", MagicMock(), segment_id=None)

    assert result == []
    yf.assert_not_called()


def test_us_segment_still_uses_yfinance_fallback(monkeypatch) -> None:
    from scripts.training import data_loader as dl

    _patch_db_empty(monkeypatch)
    sentinel = [object()]
    instance = MagicMock()
    instance.fetch_candles.return_value = sentinel
    yf = MagicMock(name="YFinanceFetcher", return_value=instance)
    monkeypatch.setattr(dl, "YFinanceFetcher", yf)

    result = dl.fetch_symbol_candles("SPY", "us", MagicMock(), segment_id=None)

    assert result is sentinel
    yf.assert_called_once_with(market_id="us")


# ── MEDIUM: fundamentals must be attributed to the scored symbol ────────────


def _snap(symbol: str, pe: float, day: int) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        as_of=datetime(2025, 1, day, tzinfo=UTC),
        pe_ratio=pe,
    )


def _segment_wide() -> MoexMarketData:
    # BBB is the globally-latest snapshot; AAA is older. Distinct pe -> distinct yield.
    return MoexMarketData(
        fundamentals=(
            _snap("AAA", pe=10.0, day=1),  # earnings_yield 0.10
            _snap("BBB", pe=20.0, day=9),  # earnings_yield 0.05 (latest as_of)
        )
    )


def test_fundamentals_attributed_to_scored_symbol_not_globally_latest() -> None:
    as_of = datetime(2025, 2, 1, tzinfo=UTC)
    feats = compute_fundamental_features(_segment_wide(), as_of=as_of, symbol="AAA")
    # Must be AAA's 1/10 = 0.10, NOT BBB's (globally-latest) 1/20 = 0.05.
    assert feats["earnings_yield"] == 0.10


def test_unknown_symbol_returns_default() -> None:
    as_of = datetime(2025, 2, 1, tzinfo=UTC)
    feats = compute_fundamental_features(_segment_wide(), as_of=as_of, symbol="ZZZ")
    assert feats["earnings_yield"] == 0.0


def test_legacy_no_symbol_uses_globally_latest_backcompat() -> None:
    as_of = datetime(2025, 2, 1, tzinfo=UTC)
    feats = compute_fundamental_features(_segment_wide(), as_of=as_of)
    # Back-compat: globally-latest snapshot is BBB -> 1/20 = 0.05.
    assert feats["earnings_yield"] == 0.05
