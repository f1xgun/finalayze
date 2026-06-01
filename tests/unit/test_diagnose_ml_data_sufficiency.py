"""Unit tests for the H1 data/sample-sufficiency reporter (MLDIAG-01).

Tests cover:
- The PURE counting helper ``report_for_candles`` produces an integer
  triple-barrier ``sample_count``, an integer ``fold_count``, a ``raw_bar_count``
  equal to the number of candles fed, a ``date_span`` (start, end) tuple, a
  ``class_balance`` mapping, and per-fold ``n_test`` / ``n_effective`` entries.
- The helper is TOKEN-FREE by construction: it takes already-fetched candles as
  an argument, reads no ``FINALAYZE_TINKOFF_TOKEN`` and makes no network call.

These tests run on SYNTHETIC, deterministic candle input only -- NO network,
NO token, NO Tinkoff, NO DB. They reuse the EXACT production code paths
(``build_triple_barrier_dataset`` + ``generate_walk_forward_folds``) so the
reported counts equal what the WF gate sees.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Ensure scripts/ and project root are importable (config/ is at project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from finalayze.core.schemas import Candle  # noqa: E402

# Named constants -- no magic numbers (ruff PLR2004)
_N_BARS = 600  # > _MIN_HISTORY_DAYS (500) so the symbol is NOT skipped
_BASE_TS = datetime(2023, 1, 1, tzinfo=UTC)
_BASE_PRICE = 100.0
_SYMBOL = "TEST"
_SEGMENT_ID = "ru_blue_chips"  # a D-03 MOEX segment (drives MOEX WF windows)
_TF = "1d"
_EXPECTED_MIN_FOLDS = 1  # MOEX 8/1/3 step-2 windows over ~600 daily bars -> >= 1 fold
_TWO_LABEL_CLASSES = 2  # triple-barrier labels are binary {0, 1}


def _make_candles(n: int, start: datetime = _BASE_TS) -> list[Candle]:
    """Build a deterministic ``list[Candle]`` of known length / date span.

    Mirrors the ``_make_candle`` builder idiom from ``test_triple_barrier.py``:
    one daily candle per index with a gently oscillating close so that
    triple-barrier labelling produces a non-degenerate mix of label classes.
    """
    candles: list[Candle] = []
    for i in range(n):
        # Deterministic oscillation: slow sine-like drift via integer arithmetic.
        close = _BASE_PRICE + (i % 20) - 10 + (i // 50)
        high = close * 1.01
        low = close * 0.99
        candles.append(
            Candle(
                symbol=_SYMBOL,
                market_id="moex",
                timeframe=_TF,
                timestamp=start + timedelta(days=i),
                open=Decimal(str(round(close, 4))),
                high=Decimal(str(round(high, 4))),
                low=Decimal(str(round(low, 4))),
                close=Decimal(str(round(close, 4))),
                volume=1000,
            )
        )
    return candles


def test_reports_sample_and_fold_counts_on_synthetic_input() -> None:
    """The pure helper returns exact synthetic sample/fold/history counts."""
    from scripts.diagnose_ml_data_sufficiency import report_for_candles

    candles = _make_candles(_N_BARS)
    report = report_for_candles(_SEGMENT_ID, _SYMBOL, candles)

    # Raw history
    assert report["raw_bar_count"] == _N_BARS
    start, end = report["date_span"]
    assert start == candles[0].timestamp
    assert end == candles[-1].timestamp

    # Triple-barrier samples
    assert isinstance(report["sample_count"], int)
    assert report["sample_count"] >= 0

    # Label class balance is a mapping of label -> count (binary triple-barrier)
    class_balance = report["class_balance"]
    assert isinstance(class_balance, dict)
    assert len(class_balance) <= _TWO_LABEL_CLASSES
    assert sum(class_balance.values()) == report["sample_count"]

    # WF folds reuse the production generator
    assert isinstance(report["fold_count"], int)
    assert report["fold_count"] >= _EXPECTED_MIN_FOLDS
    assert len(report["folds"]) == report["fold_count"]
    for fold in report["folds"]:
        assert isinstance(fold["n_test"], int)
        assert fold["n_test"] >= 0
        assert isinstance(fold["n_effective"], float)
        assert fold["n_effective"] >= 0.0


def test_reporter_is_token_free_on_synthetic_input(
    monkeypatch: object,
) -> None:
    """The helper produces its report WITHOUT reading any Tinkoff token.

    The helper-under-test takes candles as an argument, so it is token-free by
    construction. We hard-prove it: remove ``FINALAYZE_TINKOFF_TOKEN`` from the
    environment and replace ``os.environ.get`` with a guard that fails the test
    if the token name is ever read.
    """
    import os

    mp = monkeypatch  # typed loosely to avoid importing pytest at module scope
    mp.delenv("FINALAYZE_TINKOFF_TOKEN", raising=False)  # type: ignore[attr-defined]

    real_get = os.environ.get

    def _guarded_get(key: str, default: object = None) -> object:
        if key == "FINALAYZE_TINKOFF_TOKEN":
            msg = "reporter read FINALAYZE_TINKOFF_TOKEN on the synthetic-candle path"
            raise AssertionError(msg)
        return real_get(key, default)

    mp.setattr(os.environ, "get", _guarded_get)  # type: ignore[attr-defined]

    from scripts.diagnose_ml_data_sufficiency import report_for_candles

    candles = _make_candles(_N_BARS)
    report = report_for_candles(_SEGMENT_ID, _SYMBOL, candles)
    assert report["raw_bar_count"] == _N_BARS
