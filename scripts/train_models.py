"""Train XGBoost + LightGBM + LSTM models per market segment.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --segment us_tech
    uv run python scripts/train_models.py --segment us_tech --output-dir models/
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure src/ is importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# torch must be imported before lightgbm to prevent OpenMP thread-pool conflicts
import torch  # noqa: F401
from sklearn.metrics import accuracy_score

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.features.technical import compute_features
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.lstm_model import LSTMModel
from finalayze.ml.models.xgboost_model import XGBoostModel

_WINDOW_SIZE = 60
_TRAIN_RATIO = 0.8
_LOOKBACK_DAYS = 730  # 2 years of history
_DEFAULT_OUTPUT_DIR = "models/"
_SEQUENCE_LENGTH = 20

# Map segment_id → representative symbols for yfinance fallback
_SEGMENT_SYMBOLS: dict[str, list[str]] = {
    "us_tech": ["AAPL", "MSFT", "GOOGL"],
    "us_healthcare": ["JNJ", "PFE", "UNH"],
    "us_finance": ["JPM", "BAC", "GS"],
    "us_broad": ["SPY", "QQQ", "IWM"],
    "ru_blue_chips": ["SBER.ME", "GAZP.ME", "LKOH.ME"],
    "ru_energy": ["NVTK.ME", "ROSN.ME"],
    "ru_tech": ["YNDX.ME", "OZON.ME"],
    "ru_finance": ["VTBR.ME", "MOEX.ME"],
}


def _fetch_candles(segment_id: str, symbols: list[str]) -> list[Candle]:
    """Fetch candles from yfinance for the given symbols."""
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_LOOKBACK_DAYS)
    market_id = segment_id.split("_", maxsplit=1)[0]
    fetcher = YFinanceFetcher(market_id=market_id)
    candles: list[Candle] = []
    for symbol in symbols:
        try:
            fetched = fetcher.fetch_candles(symbol, start, end)
            candles.extend(fetched)
        except Exception as exc:
            print(f"  [warn] Could not fetch {symbol}: {exc}")
    return candles


def _build_dataset(
    candles: list[Candle],
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) from windowed candles."""
    features_list: list[dict[str, float]] = []
    label_list: list[int] = []
    # Sort candles by timestamp
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)
    for i in range(len(sorted_candles) - _WINDOW_SIZE):
        window = sorted_candles[i : i + _WINDOW_SIZE]
        try:
            row_features = compute_features(window)
        except Exception:
            continue
        next_close = float(sorted_candles[i + _WINDOW_SIZE].close)
        cur_close = float(sorted_candles[i + _WINDOW_SIZE - 1].close)
        label = 1 if next_close > cur_close else 0
        features_list.append(row_features)
        label_list.append(label)
    return features_list, label_list


def train_one_segment(
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
) -> None:
    """Train and save models for a single segment."""
    print(f"\n[{segment_id}] Fetching candles for {symbols}...")
    candles = _fetch_candles(segment_id, symbols)

    if not candles:
        print(f"[{segment_id}] No candles — skipping.")
        return

    features_list, label_list = _build_dataset(candles)
    if len(features_list) < _WINDOW_SIZE:
        print(f"[{segment_id}] Only {len(features_list)} samples — need {_WINDOW_SIZE}+, skipping.")
        return

    split = int(len(features_list) * _TRAIN_RATIO)
    train_features = features_list[:split]
    test_features = features_list[split:]
    train_labels = label_list[:split]
    test_labels = label_list[split:]

    if len(train_features) < _SEQUENCE_LENGTH:
        print(f"[{segment_id}] Train split too small for LSTM — skipping.")
        return

    segment_dir = output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}

    # XGBoost
    xgb = XGBoostModel(segment_id=segment_id)
    xgb.fit(train_features, train_labels)
    xgb.save(segment_dir / "xgb.pkl")
    if test_features:
        pred_xgb = [round(xgb.predict_proba(f)) for f in test_features]
        results["XGB"] = float(accuracy_score(test_labels, pred_xgb))

    # LightGBM
    lgbm = LightGBMModel(segment_id=segment_id)
    lgbm.fit(train_features, train_labels)
    lgbm.save(segment_dir / "lgbm.pkl")
    if test_features:
        pred_lgbm = [round(lgbm.predict_proba(f)) for f in test_features]
        results["LGBM"] = float(accuracy_score(test_labels, pred_lgbm))

    # LSTM
    lstm = LSTMModel(segment_id=segment_id, sequence_length=_SEQUENCE_LENGTH)
    lstm.fit(train_features, train_labels)
    lstm.save(segment_dir / "lstm.pkl")
    if test_features:
        pred_lstm = [round(lstm.predict_proba(f)) for f in test_features]
        results["LSTM"] = float(accuracy_score(test_labels, pred_lstm))

    summary = " | ".join(f"{k}: {v:.2f}" for k, v in results.items())
    print(f"[{segment_id}] {summary}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LightGBM + LSTM models per segment"
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Segment ID to train (default: all segments)",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = _parse_args()
    output_dir = Path(args.output_dir)

    if args.segment:
        segments = {args.segment: _SEGMENT_SYMBOLS.get(args.segment, [])}
    else:
        segments = _SEGMENT_SYMBOLS

    for segment_id, symbols in segments.items():
        try:
            train_one_segment(
                segment_id=segment_id,
                symbols=symbols,
                output_dir=output_dir,
            )
        except FileNotFoundError as exc:
            print(f"[{segment_id}] FileNotFoundError — {exc}, skipping.")
        except Exception as exc:
            print(f"[{segment_id}] Unexpected error — {exc}, skipping.")


if __name__ == "__main__":
    main()
