"""Train XGBoost + LightGBM + LSTM models per market segment.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --segment us_tech
    uv run python scripts/train_models.py --segment us_tech --output-dir models/
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure src/ and project root are importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))  # for config.settings

# torch must be imported before lightgbm to prevent OpenMP thread-pool conflicts
import torch  # noqa: F401
from config.settings import Settings
from sklearn.metrics import accuracy_score
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from finalayze.core.models import CandleModel
from finalayze.core.schemas import Candle
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.ml.features.technical import compute_features  # noqa: F401
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.lstm_model import LSTMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_windows

_WINDOW_SIZE = DEFAULT_WINDOW_SIZE
_TRAIN_RATIO = 0.8
_LOOKBACK_DAYS = 730  # 2 years of history
_DEFAULT_OUTPUT_DIR = "models/"
_SEQUENCE_LENGTH = 20
_MIN_CANDLES = _WINDOW_SIZE + 1  # need at least WINDOW_SIZE + 1 for one sample

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


def _orm_to_candle(row: CandleModel) -> Candle:
    """Convert a CandleModel ORM row to a Candle schema object."""
    return Candle(
        symbol=row.symbol,
        market_id=row.market_id,
        timeframe=row.timeframe,
        timestamp=row.timestamp,
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
    )


async def _fetch_from_db(symbol: str, market_id: str, settings: Settings) -> list[Candle]:
    """Try to load candles from DB. Returns empty list on failure."""
    try:
        engine = create_async_engine(settings.database_url, echo=False)
        async with AsyncSession(engine) as session:
            result = await session.execute(
                select(CandleModel)
                .where(CandleModel.symbol == symbol, CandleModel.market_id == market_id)
                .order_by(CandleModel.timestamp)
            )
            rows = result.scalars().all()
            return [_orm_to_candle(row) for row in rows]
    except Exception:
        return []


def _fetch_symbol_candles(symbol: str, market_id: str, settings: Settings) -> list[Candle]:
    """Fetch candles for a single symbol: DB first, yfinance fallback."""
    candles = asyncio.run(_fetch_from_db(symbol, market_id, settings))
    if candles:
        return candles
    # Fallback to yfinance
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=_LOOKBACK_DAYS)
    fetcher = YFinanceFetcher(market_id=market_id)
    try:
        return fetcher.fetch_candles(symbol, start, end)
    except Exception as exc:
        print(f"  [warn] Could not fetch {symbol} from yfinance: {exc}")
        return []


def _fetch_candles(
    segment_id: str, symbols: list[str], settings: Settings | None = None
) -> list[Candle]:
    """Fetch candles for all symbols in a segment, processing each independently."""
    if settings is None:
        settings = Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]
    candles: list[Candle] = []
    for symbol in symbols:
        symbol_candles = _fetch_symbol_candles(symbol, market_id, settings)
        candles.extend(symbol_candles)
    return candles


def _build_windows(candles: list[Candle]) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) from a single contiguous candle series.

    Delegates to the shared ``build_windows`` utility in ``finalayze.ml.training``.
    """
    return build_windows(candles, _WINDOW_SIZE)


def _build_dataset(
    segment_id: str,
    symbols: list[str],
    settings: Settings | None = None,
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) by processing each symbol's candles independently."""
    if settings is None:
        settings = Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]
    features_out: list[dict[str, float]] = []
    labels_out: list[int] = []
    for symbol in symbols:
        candles = _fetch_symbol_candles(symbol, market_id, settings)
        if len(candles) < _MIN_CANDLES:
            continue
        x_sym, y_sym = _build_windows(candles)
        features_out.extend(x_sym)
        labels_out.extend(y_sym)
    return features_out, labels_out


def train_one_segment(
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
    settings: Settings | None = None,
) -> None:
    """Train and save models for a single segment."""
    if settings is None:
        settings = Settings()
    print(f"\n[{segment_id}] Fetching candles for {symbols}...")

    features_list, label_list = _build_dataset(segment_id, symbols, settings)
    if not features_list:
        print(f"[{segment_id}] No candles — skipping.")
        return

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
