"""Train ML models for MOEX segments using Tinkoff Invest API.

Fetches MOEX candle data via Tinkoff gRPC API (since yfinance can't access
Russian stocks post-sanctions), then trains XGBoost + LightGBM + LSTM models.

Usage:
    TINKOFF_TOKEN=t.xxx uv run python scripts/train_moex_models.py \
        --output-dir models/
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

# torch must be imported before lightgbm to prevent OpenMP conflicts
import torch  # noqa: F401, I001
import grpc
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from t_tech.invest import CandleInterval
from t_tech.invest.async_services import AsyncServices
from t_tech.invest.channels import _required_options

from finalayze.core.schemas import Candle
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.lstm_model import LSTMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.training import DEFAULT_WINDOW_SIZE, build_windows

_WINDOW_SIZE = DEFAULT_WINDOW_SIZE
_TRAIN_RATIO = 0.8
_SEQUENCE_LENGTH = 20
_MIN_CANDLES = _WINDOW_SIZE + 1
_MOEX_MAX_DEPTH = 3  # Shallower trees for smaller MOEX dataset
_DEFAULT_OUTPUT_DIR = "models/"

_NANO_DIVISOR = Decimal(1_000_000_000)
_TBANK_TARGET = "invest-public-api.tbank.ru:443"

# MOEX symbol → FIGI mapping
MOEX_FIGI: dict[str, str] = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0",
    "LKOH": "BBG004731032",
    "GMKN": "BBG004731489",
    "VTBR": "BBG004730ZJ9",
    "NVTK": "BBG00475KKY8",
    "ROSN": "BBG004731354",
    "MOEX": "BBG004730JJ5",
    "MGNT": "BBG004RVFCY3",
    "TATN": "BBG004RVFFC0",
    "NLMK": "BBG004S681B4",
    "CHMF": "BBG00475K6C3",
    "ALRS": "BBG004S68B31",
    "PLZL": "BBG000R607Y3",
}

# Segments for training
MOEX_TRAIN_SEGMENTS: dict[str, list[str]] = {
    "ru_blue_chips": ["SBER", "GAZP", "LKOH", "GMKN", "VTBR", "MOEX"],
    "ru_energy": ["GAZP", "LKOH", "NVTK", "ROSN", "TATN"],
    "ru_finance": ["SBER", "VTBR", "MOEX"],
}


def _quotation_to_decimal(q: Any) -> Decimal:
    """Convert Tinkoff Quotation(units, nano) to Decimal."""
    return Decimal(q.units) + Decimal(q.nano) / _NANO_DIVISOR


def _get_tbank_root_certs() -> bytes:
    """Fetch TLS certificate chain from tbank.ru."""
    cache_path = Path(__file__).parent / ".tbank_certs.pem"
    if cache_path.exists():
        return cache_path.read_bytes()
    result = subprocess.run(
        [  # noqa: S607
            "openssl",
            "s_client",
            "-showcerts",
            "-connect",
            "invest-public-api.tbank.ru:443",
            "-servername",
            "invest-public-api.tbank.ru",
        ],
        input=b"",
        capture_output=True,
        timeout=10,
        check=False,
    )
    pem_blocks: list[str] = []
    in_cert = False
    current: list[str] = []
    for line in result.stdout.decode().splitlines():
        if "BEGIN CERTIFICATE" in line:
            in_cert = True
            current = [line]
        elif "END CERTIFICATE" in line:
            current.append(line)
            pem_blocks.append("\n".join(current))
            in_cert = False
        elif in_cert:
            current.append(line)
    pem_data = "\n".join(pem_blocks).encode()
    cache_path.write_bytes(pem_data)
    return pem_data


def _create_tbank_channel() -> grpc.aio.Channel:
    """Create gRPC channel with Russian CA certs."""
    root_certs = _get_tbank_root_certs()
    creds = grpc.ssl_channel_credentials(root_certificates=root_certs)
    options: list[tuple[str, Any]] = list(_required_options)
    return grpc.aio.secure_channel(_TBANK_TARGET, creds, options)


async def _fetch_candles_async(
    token: str, figi: str, symbol: str, start: datetime, end: datetime
) -> list[Candle]:
    """Fetch daily candles via Tinkoff API."""
    channel = _create_tbank_channel()
    try:
        services = AsyncServices(channel, token)
        response = await services.market_data.get_candles(
            figi=figi,
            from_=start,
            to=end,
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
        )
    finally:
        await channel.close()

    candles: list[Candle] = []
    for raw in response.candles:
        ts = (
            raw.time
            if isinstance(raw.time, datetime)
            else datetime.fromtimestamp(raw.time.seconds + raw.time.nanos / 1e9, tz=UTC)
        )
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        candles.append(
            Candle(
                symbol=symbol,
                market_id="moex",
                timeframe="1d",
                timestamp=ts,
                open=_quotation_to_decimal(raw.open),
                high=_quotation_to_decimal(raw.high),
                low=_quotation_to_decimal(raw.low),
                close=_quotation_to_decimal(raw.close),
                volume=int(raw.volume),
                source="tinkoff",
            )
        )
    return candles


def _fetch_tinkoff_candles(token: str, symbol: str, start: datetime, end: datetime) -> list[Candle]:
    """Sync wrapper: fetch candles for a MOEX symbol via Tinkoff API.

    Splits the date range into yearly chunks (API limit).
    """
    figi = MOEX_FIGI.get(symbol)
    if not figi:
        print(f"  [warn] No FIGI for {symbol}, skipping")
        return []

    all_candles: list[Candle] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=365), end)
        try:
            chunk = asyncio.run(_fetch_candles_async(token, figi, symbol, chunk_start, chunk_end))
            all_candles.extend(chunk)
        except Exception as exc:
            print(
                f"  [warn] Tinkoff fetch failed for {symbol} "
                f"({chunk_start.date()} to {chunk_end.date()}): {exc}"
            )
        chunk_start = chunk_end

    return all_candles


def _build_dataset(
    token: str,
    _segment_id: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> tuple[list[dict[str, float]], list[int]]:
    """Fetch candles from Tinkoff and build (features, labels) dataset.

    Each symbol is processed independently, then sorted by timestamp
    for proper temporal ordering.
    """
    rows: list[tuple[datetime, dict[str, float], int]] = []

    for symbol in symbols:
        print(f"  Fetching {symbol}...", end="", flush=True)
        candles = _fetch_tinkoff_candles(token, symbol, start, end)
        print(f" {len(candles)} bars")

        if len(candles) < _MIN_CANDLES:
            print(f"  {symbol}: only {len(candles)} candles, need {_MIN_CANDLES}+, skipping")
            continue

        x_sym, y_sym, ts_sym = build_windows(candles, _WINDOW_SIZE)
        for ts, feat, lbl in zip(ts_sym, x_sym, y_sym, strict=True):
            rows.append((ts, feat, lbl))

    # Sort by timestamp for proper temporal ordering
    rows.sort(key=lambda r: r[0])
    features_out = [r[1] for r in rows]
    labels_out = [r[2] for r in rows]
    return features_out, labels_out


def _evaluate_model(
    model: XGBoostModel | LightGBMModel | LSTMModel,
    test_features: list[dict[str, float]],
    test_labels: list[int],
) -> str:
    """Evaluate a model and return formatted summary."""
    probas = [model.predict_proba(f) for f in test_features]
    preds = [round(p) for p in probas]
    acc = float(accuracy_score(test_labels, preds))
    brier = float(brier_score_loss(test_labels, probas))
    ll = float(log_loss(test_labels, probas, labels=[0, 1]))
    return f"acc={acc:.3f} brier={brier:.3f} logloss={ll:.3f}"


def _train_segment(
    token: str,
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
    start: datetime,
    end: datetime,
) -> None:
    """Train and save models for one MOEX segment."""
    print(f"\n[{segment_id}] Training with {len(symbols)} symbols: {symbols}")

    features, labels = _build_dataset(token, segment_id, symbols, start, end)
    if not features:
        print(f"[{segment_id}] No data — skipping.")
        return

    if len(features) < _WINDOW_SIZE:
        print(f"[{segment_id}] Only {len(features)} samples, need {_WINDOW_SIZE}+, skipping.")
        return

    # Train/test split with purge gap
    split = int(len(features) * _TRAIN_RATIO)
    gap_end = min(split + _WINDOW_SIZE, len(features))
    train_x = features[:split]
    test_x = features[gap_end:]
    train_y = labels[:split]
    test_y = labels[gap_end:]

    if len(train_x) < _SEQUENCE_LENGTH:
        print(f"[{segment_id}] Train set too small ({len(train_x)}), skipping.")
        return

    segment_dir = output_dir / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    # XGBoost (shallow for MOEX)
    xgb = XGBoostModel(segment_id=segment_id, max_depth=_MOEX_MAX_DEPTH)
    xgb.fit(train_x, train_y)
    xgb.save(segment_dir / "xgb.pkl")
    if xgb._model is not None and xgb._feature_names is not None:
        importances = xgb._model.feature_importances_
        feat_imp = sorted(
            zip(xgb._feature_names, importances, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"[{segment_id}] XGBoost top-10 features:")
        for name, score in feat_imp[:10]:
            print(f"  {name:>25s}: {score:.4f}")
    if test_x:
        results["XGB"] = _evaluate_model(xgb, test_x, test_y)

    # LightGBM
    lgbm = LightGBMModel(segment_id=segment_id)
    lgbm.fit(train_x, train_y)
    lgbm.save(segment_dir / "lgbm.pkl")
    if test_x:
        results["LGBM"] = _evaluate_model(lgbm, test_x, test_y)

    # LSTM
    lstm = LSTMModel(segment_id=segment_id, sequence_length=_SEQUENCE_LENGTH)
    lstm.fit(train_x, train_y)
    lstm.save(segment_dir / "lstm.pkl")
    if test_x:
        results["LSTM"] = _evaluate_model(lstm, test_x, test_y)

    summary = " | ".join(f"{k}: {v}" for k, v in results.items())
    print(f"[{segment_id}] {summary}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Train MOEX ML models via Tinkoff API")
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Train only this segment (default: all RU segments)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Start year (default: 2021, post-COVID)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year (default: 2025)",
    )
    args = parser.parse_args()

    # Try multiple env var names and .env file
    token = os.environ.get("TINKOFF_TOKEN", "") or os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        # Try loading from .env file
        env_path = _PROJECT_ROOT / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text().splitlines():
                stripped = raw_line.strip()
                if stripped.startswith("#") or "=" not in stripped:
                    continue
                key, _, val = stripped.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key in ("TINKOFF_TOKEN", "FINALAYZE_TINKOFF_TOKEN") and val:
                    token = val
                    break
    if not token:
        print("ERROR: TINKOFF_TOKEN not set (checked env vars and .env file)")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    start = datetime(args.start_year, 1, 1, tzinfo=UTC)
    end = datetime(args.end_year, 1, 1, tzinfo=UTC)

    if args.segment:
        segments = {args.segment: MOEX_TRAIN_SEGMENTS.get(args.segment, [])}
    else:
        segments = MOEX_TRAIN_SEGMENTS

    print(f"Training MOEX models: {list(segments.keys())}")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Output: {output_dir}")

    os.environ["GRPC_DNS_RESOLVER"] = "native"

    for segment_id, symbols in segments.items():
        try:
            _train_segment(token, segment_id, symbols, output_dir, start, end)
        except Exception as exc:
            import traceback  # noqa: PLC0415

            print(f"[{segment_id}] Error: {exc}")
            traceback.print_exc()

    print("\nDone! Models saved to:", output_dir)


if __name__ == "__main__":
    main()
