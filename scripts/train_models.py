"""Train XGBoost + LightGBM + LSTM models per market segment.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --segment us_tech
    uv run python scripts/train_models.py --segment us_tech --output-dir models/
"""

from __future__ import annotations

import argparse
import asyncio
import os
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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
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
_LOOKBACK_DAYS = 1825  # 5 years of history for US segments
_MOEX_LOOKBACK_DAYS = 730  # 2 years for MOEX (post-sanctions structural break)
_DEFAULT_OUTPUT_DIR = "models/"
_SEQUENCE_LENGTH = 20
_MIN_CANDLES = _WINDOW_SIZE + 1  # need at least WINDOW_SIZE + 1 for one sample

# XGBoost max_depth: shallower for MOEX (smaller dataset, prevent overfit)
_US_MAX_DEPTH = 5
_MOEX_MAX_DEPTH = 3

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


def _is_moex_segment(segment_id: str) -> bool:
    """Return True if segment_id is a MOEX/Russian segment."""
    return segment_id.startswith("ru_")


def _get_lookback_days(segment_id: str) -> int:
    """Return lookback days: 2 years for MOEX, 5 years for US."""
    return _MOEX_LOOKBACK_DAYS if _is_moex_segment(segment_id) else _LOOKBACK_DAYS


def _get_xgboost_max_depth(segment_id: str) -> int:
    """Return XGBoost max_depth: 3 for MOEX, 5 for US."""
    return _MOEX_MAX_DEPTH if _is_moex_segment(segment_id) else _US_MAX_DEPTH


def _fetch_tinkoff_candles(symbol: str) -> list[Candle]:
    """Fetch candles from Tinkoff Invest API for MOEX symbols.

    Requires TINKOFF_TOKEN environment variable. Returns empty list on failure.
    """
    token = os.environ.get("TINKOFF_TOKEN")
    if not token:
        print(f"  [warn] TINKOFF_TOKEN not set, skipping Tinkoff fetch for {symbol}")
        return []

    try:
        from t_tech.invest import AsyncClient, CandleInterval  # noqa: PLC0415

        async def _fetch() -> list[Candle]:
            end = datetime.now(tz=UTC)
            start = end - timedelta(days=_MOEX_LOOKBACK_DAYS)
            candles_out: list[Candle] = []
            async with AsyncClient(token) as client:
                async for candle in client.get_all_candles(
                    figi=symbol,
                    from_=start,
                    to=end,
                    interval=CandleInterval.CANDLE_INTERVAL_DAY,
                ):
                    from decimal import Decimal  # noqa: PLC0415

                    candles_out.append(
                        Candle(
                            symbol=symbol,
                            market_id="moex",
                            timeframe="1d",
                            timestamp=candle.time,
                            open=Decimal(str(candle.open.units + candle.open.nano / 1e9)),
                            high=Decimal(str(candle.high.units + candle.high.nano / 1e9)),
                            low=Decimal(str(candle.low.units + candle.low.nano / 1e9)),
                            close=Decimal(str(candle.close.units + candle.close.nano / 1e9)),
                            volume=int(candle.volume),
                        )
                    )
            return candles_out

        return asyncio.run(_fetch())
    except Exception as exc:
        print(f"  [warn] Tinkoff fetch failed for {symbol}: {exc}")
        return []


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


def _fetch_symbol_candles(
    symbol: str,
    market_id: str,
    settings: Settings,
    segment_id: str | None = None,
) -> list[Candle]:
    """Fetch candles for a single symbol: DB first, then API fallback.

    For MOEX segments, tries Tinkoff API before yfinance. Uses segment-aware
    lookback (2 years for MOEX, 5 years for US).
    """
    candles = asyncio.run(_fetch_from_db(symbol, market_id, settings))
    if candles:
        return candles

    lookback = _get_lookback_days(segment_id) if segment_id else _LOOKBACK_DAYS

    # For MOEX segments, try Tinkoff first
    if segment_id and _is_moex_segment(segment_id):
        tinkoff_candles = _fetch_tinkoff_candles(symbol)
        if tinkoff_candles:
            return tinkoff_candles

    # Fallback to yfinance
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=lookback)
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
        symbol_candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        candles.extend(symbol_candles)
    return candles


def _build_windows(
    candles: list[Candle],
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) from a single contiguous candle series.

    Delegates to the shared ``build_windows`` utility in ``finalayze.ml.training``.
    Discards timestamps (used only for multi-symbol temporal ordering).
    """
    features, labels, _ts = build_windows(candles, _WINDOW_SIZE)
    return features, labels


def _build_dataset(
    segment_id: str,
    symbols: list[str],
    settings: Settings | None = None,
) -> tuple[list[dict[str, float]], list[int]]:
    """Build (features, labels) by processing each symbol's candles independently.

    Collects windows from all symbols and sorts by timestamp to maintain
    proper temporal ordering for train/test splits (no future leakage).
    """
    if settings is None:
        settings = Settings()
    market_id = segment_id.split("_", maxsplit=1)[0]
    rows: list[tuple[datetime, dict[str, float], int]] = []
    for symbol in symbols:
        candles = _fetch_symbol_candles(symbol, market_id, settings, segment_id=segment_id)
        if len(candles) < _MIN_CANDLES:
            continue
        x_sym, y_sym, ts_sym = build_windows(candles, _WINDOW_SIZE)
        for ts, feat, lbl in zip(ts_sym, x_sym, y_sym, strict=True):
            rows.append((ts, feat, lbl))
    rows.sort(key=lambda r: r[0])
    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
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
    gap_end = min(split + _WINDOW_SIZE, len(features_list))
    train_features = features_list[:split]
    test_features = features_list[gap_end:]
    train_labels = label_list[:split]
    test_labels = label_list[gap_end:]

    if len(train_features) < _SEQUENCE_LENGTH:
        print(f"[{segment_id}] Train split too small for LSTM — skipping.")
        return

    segment_dir = output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)

    results = _train_and_evaluate_models(
        segment_id, segment_dir, train_features, train_labels, test_features, test_labels
    )
    summary = " | ".join(f"{k}: {v}" for k, v in results.items())
    print(f"[{segment_id}] {summary}")


def _evaluate_model(
    model: XGBoostModel | LightGBMModel | LSTMModel,
    test_features: list[dict[str, float]],
    test_labels: list[int],
) -> str:
    """Evaluate a model and return a formatted summary string."""
    probas = [model.predict_proba(f) for f in test_features]
    preds = [round(p) for p in probas]
    acc = float(accuracy_score(test_labels, preds))
    brier = float(brier_score_loss(test_labels, probas))
    ll = float(log_loss(test_labels, probas, labels=[0, 1]))
    return f"acc={acc:.3f} brier={brier:.3f} logloss={ll:.3f}"


def _train_and_evaluate_models(
    segment_id: str,
    segment_dir: Path,
    train_features: list[dict[str, float]],
    train_labels: list[int],
    test_features: list[dict[str, float]],
    test_labels: list[int],
) -> dict[str, str]:
    """Train XGBoost, LightGBM, and LSTM; return evaluation results."""
    results: dict[str, str] = {}

    max_depth = _get_xgboost_max_depth(segment_id)
    xgb = XGBoostModel(segment_id=segment_id, max_depth=max_depth)
    xgb.fit(train_features, train_labels)
    xgb.save(segment_dir / "xgb.pkl")

    # Log top-10 feature importances from XGBoost
    if xgb._model is not None and xgb._feature_names is not None:
        importances = xgb._model.feature_importances_
        feat_imp = sorted(
            zip(xgb._feature_names, importances, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"[{segment_id}] XGBoost top-10 feature importances:")
        for name, score in feat_imp[:10]:
            print(f"  {name:>25s}: {score:.4f}")

    if test_features:
        results["XGB"] = _evaluate_model(xgb, test_features, test_labels)

    lgbm = LightGBMModel(segment_id=segment_id)
    lgbm.fit(train_features, train_labels)
    lgbm.save(segment_dir / "lgbm.pkl")
    if test_features:
        results["LGBM"] = _evaluate_model(lgbm, test_features, test_labels)

    lstm = LSTMModel(segment_id=segment_id, sequence_length=_SEQUENCE_LENGTH)
    lstm.fit(train_features, train_labels)
    lstm.save(segment_dir / "lstm.pkl")
    if test_features:
        results["LSTM"] = _evaluate_model(lstm, test_features, test_labels)

    return results


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
