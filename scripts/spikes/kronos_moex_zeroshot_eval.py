"""SPIKE: zero-shot evaluation of the Kronos foundation model on MOEX candles.

This is a throw-away research spike (Group A — price/time-series forecasting).
It answers ONE question before any integration work is justified:

    Does Kronos, with NO fine-tuning, produce a usable directional edge on MOEX
    daily candles (e.g. SBER), or does it need MOEX-specific fine-tuning (which
    requires multi-GPU infrastructure we do not have)?

It is intentionally NOT wired into the trading system, adds no hard dependency,
and lives under scripts/spikes/. If the edge is real we promote it to a proper
GSD phase (Kronos-derived features → ml_ensemble reinforcer, validated through
backtest-iteration). If not, we delete this file and move on.

MOEX data is fetched via the T-Bank (Tinkoff Invest) gRPC API only — never
yfinance — per the project's MOEX data rule.

Prerequisites (kept out of the project deps on purpose):
    # 1. A T-Invest token
    export FINALAYZE_TINKOFF_TOKEN=...        # already in your .env
    # 2. Kronos + torch in a throwaway venv (do NOT add to pyproject.toml):
    pip install torch huggingface_hub einops safetensors
    git clone https://github.com/shiyu-coder/Kronos /tmp/Kronos
    export KRONOS_REPO=/tmp/Kronos            # dir containing the `model/` package

Usage:
    uv run python scripts/spikes/kronos_moex_zeroshot_eval.py \
        --symbol SBER --lookback 400 --horizon 5 --n-test 60 --device cpu

Outputs a directional-accuracy / correlation report vs naive baselines.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Project root importable (config/ lives at repo root, not under src/).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# gRPC env vars MUST be set before importing grpc (see tinkoff_data.py).
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
_GRPC_ROOTS = Path(_PROJECT_ROOT) / "certs" / "grpc_roots.pem"
if _GRPC_ROOTS.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_GRPC_ROOTS))

_TBANK_GRPC_TARGET = "invest-public-api.tbank.ru:443"
_NANO_DIVISOR = Decimal(1_000_000_000)
_MODEL_MAX_CONTEXT = 512  # Kronos-base/small hard context limit


@dataclass(frozen=True)
class Bar:
    """Minimal OHLCV bar in the shape Kronos expects."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ── MOEX candle fetch (T-Bank gRPC, direct AsyncClient pattern) ──────────────


def _quotation_to_float(q: object) -> float:
    units = getattr(q, "units", 0)
    nano = getattr(q, "nano", 0)
    return float(Decimal(units) + Decimal(nano) / _NANO_DIVISOR)


async def _resolve_figi(services: object, ticker: str) -> str | None:
    from t_tech.invest.schemas import InstrumentIdType  # noqa: PLC0415

    for class_code in ("TQBR", "TQTF", "TQPI"):
        try:
            resp = await services.instruments.share_by(  # type: ignore[attr-defined]
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                class_code=class_code,
                id=ticker,
            )
            return resp.instrument.figi  # type: ignore[attr-defined]
        except Exception:
            continue
    return None


async def _fetch_async(token: str, symbol: str, days: int) -> list[Bar]:
    from t_tech.invest import AsyncClient, CandleInterval  # noqa: PLC0415

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=days)

    client = AsyncClient(token, target=_TBANK_GRPC_TARGET)
    async with client as services:
        figi = await _resolve_figi(services, symbol)
        if figi is None:
            msg = f"FIGI not found for {symbol}"
            raise RuntimeError(msg)

        bars: list[Bar] = [
            Bar(
                ts=candle.time,
                open=_quotation_to_float(candle.open),
                high=_quotation_to_float(candle.high),
                low=_quotation_to_float(candle.low),
                close=_quotation_to_float(candle.close),
                volume=float(candle.volume),
            )
            async for candle in services.get_all_candles(  # type: ignore[attr-defined]
                figi=figi,
                from_=start,
                to=end,
                interval=CandleInterval.CANDLE_INTERVAL_DAY,
            )
        ]
    bars.sort(key=lambda b: b.ts)
    return bars


def fetch_moex_bars(symbol: str, days: int) -> list[Bar]:
    """Fetch daily MOEX bars for *symbol* via T-Bank. Sync wrapper."""
    import asyncio  # noqa: PLC0415

    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv()
    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")
    if not token:
        msg = "FINALAYZE_TINKOFF_TOKEN not set (see .env)."
        raise SystemExit(msg)
    return asyncio.run(_fetch_async(token, symbol, days))


# ── Kronos loading (lazy, optional dependency) ───────────────────────────────


def load_kronos(device: str, model_name: str):
    """Import and construct a KronosPredictor. Fails loudly with install help."""
    repo = os.environ.get("KRONOS_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "Could not import Kronos. This spike needs the Kronos repo + torch in a "
            "throwaway venv (NOT in pyproject.toml):\n"
            "  pip install torch huggingface_hub einops safetensors\n"
            "  git clone https://github.com/shiyu-coder/Kronos /tmp/Kronos\n"
            "  export KRONOS_REPO=/tmp/Kronos\n"
            f"Original error: {exc}"
        )
        raise SystemExit(msg) from exc

    # Tokenizer is shared: small/base → Kronos-Tokenizer-base; mini → Kronos-Tokenizer-2k.
    tok_name = "Kronos-Tokenizer-2k" if model_name == "Kronos-mini" else "Kronos-Tokenizer-base"
    tokenizer = KronosTokenizer.from_pretrained(f"NeoQuasar/{tok_name}")
    model = Kronos.from_pretrained(f"NeoQuasar/{model_name}")
    return KronosPredictor(model, tokenizer, device=device, max_context=_MODEL_MAX_CONTEXT)


# ── Evaluation ───────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    n: int
    kronos_dir_acc_1: float  # next-bar directional accuracy
    kronos_dir_acc_h: float  # horizon-bar directional accuracy
    persistence_dir_acc_1: float  # naive "tomorrow == today's direction" baseline
    up_rate: float  # base rate of up-moves (majority-class baseline)
    ret_corr: float  # corr(predicted next-bar return, actual next-bar return)


def _to_dataframe(bars: list[Bar]):
    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
            "amount": [b.close * b.volume for b in bars],  # turnover proxy
        }
    )
    ts = pd.Series([b.ts for b in bars])
    return df, ts


def evaluate(
    predictor,
    bars: list[Bar],
    lookback: int,
    horizon: int,
    n_test: int,
    temperature: float,
    top_p: float,
    sample_count: int,
) -> EvalResult:
    """Walk-forward zero-shot eval: predict `horizon` bars from `lookback` context."""
    import numpy as np  # noqa: PLC0415

    df, ts = _to_dataframe(bars)
    lookback = min(lookback, _MODEL_MAX_CONTEXT)

    total = len(bars)
    first = total - n_test - horizon
    if first < lookback:
        msg = (
            f"Not enough bars: have {total}, need lookback({lookback}) + "
            f"n_test({n_test}) + horizon({horizon})."
        )
        raise SystemExit(msg)

    k_correct_1 = k_correct_h = persist_correct_1 = ups = 0
    pred_rets: list[float] = []
    actual_rets: list[float] = []
    n = 0

    for t in range(first, first + n_test):
        ctx = df.iloc[t - lookback : t]
        ctx_ts = ts.iloc[t - lookback : t]
        fut_ts = ts.iloc[t : t + horizon]
        last_close = float(df["close"].iloc[t - 1])

        pred = predictor.predict(
            df=ctx,
            x_timestamp=ctx_ts,
            y_timestamp=fut_ts,
            pred_len=horizon,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
        pred_close_1 = float(pred["close"].iloc[0])
        pred_close_h = float(pred["close"].iloc[horizon - 1])

        actual_close_1 = float(df["close"].iloc[t])
        actual_close_h = float(df["close"].iloc[t + horizon - 1])
        prev_close = float(df["close"].iloc[t - 2]) if t >= 2 else last_close  # noqa: PLR2004

        # Directional accuracy (next bar and horizon bar)
        if (pred_close_1 > last_close) == (actual_close_1 > last_close):
            k_correct_1 += 1
        if (pred_close_h > last_close) == (actual_close_h > last_close):
            k_correct_h += 1
        # Persistence baseline: predict next move == previous move direction
        if (last_close > prev_close) == (actual_close_1 > last_close):
            persist_correct_1 += 1
        if actual_close_1 > last_close:
            ups += 1

        pred_rets.append((pred_close_1 - last_close) / last_close)
        actual_rets.append((actual_close_1 - last_close) / last_close)
        n += 1

    corr = float(np.corrcoef(pred_rets, actual_rets)[0, 1]) if n > 1 else float("nan")
    return EvalResult(
        n=n,
        kronos_dir_acc_1=k_correct_1 / n,
        kronos_dir_acc_h=k_correct_h / n,
        persistence_dir_acc_1=persist_correct_1 / n,
        up_rate=ups / n,
        ret_corr=corr,
    )


def _print_report(symbol: str, model_name: str, horizon: int, res: EvalResult) -> None:
    print("=" * 64)
    print(f"Kronos zero-shot MOEX eval — {symbol} (model={model_name})")
    print("=" * 64)
    print(f"  test points                : {res.n}")
    print(f"  up-move base rate          : {res.up_rate:6.1%}  (majority-class baseline)")
    print(f"  persistence dir-acc (1 bar): {res.persistence_dir_acc_1:6.1%}  (naive baseline)")
    print(f"  KRONOS dir-acc (1 bar)     : {res.kronos_dir_acc_1:6.1%}")
    print(f"  KRONOS dir-acc ({horizon}-bar)     : {res.kronos_dir_acc_h:6.1%}")
    print(f"  KRONOS return corr (1 bar) : {res.ret_corr:6.3f}")
    print("-" * 64)
    edge = res.kronos_dir_acc_1 - max(res.up_rate, 1 - res.up_rate, res.persistence_dir_acc_1)
    verdict = "PROMISING" if edge > 0.03 else "NO CLEAR EDGE"  # noqa: PLR2004
    print(f"  edge over best baseline    : {edge:+.1%}  →  {verdict}")
    print("=" * 64)
    print(
        "\nDecision gate: only promote to a GSD feature phase if dir-acc beats the "
        "best baseline by a stable margin across several symbols/seeds. A single-symbol "
        "coin-flip means zero-shot is insufficient and MOEX fine-tuning (multi-GPU) "
        "would be required — out of scope for now."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos zero-shot MOEX eval (spike)")
    parser.add_argument("--symbol", default="SBER")
    parser.add_argument("--days", type=int, default=1200, help="calendar days of history to fetch")
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--n-test", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="Kronos-base")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sample-count", type=int, default=1)
    args = parser.parse_args()

    print(f"Fetching {args.days}d of {args.symbol} daily candles via T-Bank…")
    bars = fetch_moex_bars(args.symbol, args.days)
    print(f"  got {len(bars)} bars ({bars[0].ts.date()} → {bars[-1].ts.date()})")

    print(f"Loading Kronos ({args.model}, device={args.device})…")
    predictor = load_kronos(args.device, args.model)

    print("Running walk-forward zero-shot evaluation…")
    res = evaluate(
        predictor,
        bars,
        lookback=args.lookback,
        horizon=args.horizon,
        n_test=args.n_test,
        temperature=args.temperature,
        top_p=args.top_p,
        sample_count=args.sample_count,
    )
    _print_report(args.symbol, args.model, args.horizon, res)


if __name__ == "__main__":
    main()
