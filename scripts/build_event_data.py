"""Fetch event data (dividends, earnings, CBR rates) for backtesting.

Usage:
    uv run python scripts/build_event_data.py \
        --symbols-from-universe \
        --start 2023-01-01 --end 2024-12-31

    uv run python scripts/build_event_data.py \
        --symbols-from-universe \
        --start 2023-01-01 --end 2024-12-31 \
        --output results/event_data/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
from pathlib import Path

# Ensure config/ at project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yfinance as yf

# ── Symbol universe (matches run_iteration.py) ───────────────────────────────

UNIVERSE: dict[str, list[str]] = {
    "us_tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSM",
        "AVGO",
        "ADBE",
        "CRM",
        "INTC",
        "AMD",
        "CSCO",
        "ORCL",
        "QCOM",
        "TXN",
        "ASML",
        "AMAT",
        "MU",
        "NOW",
    ],
    "us_broad": [
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "JNJ",
        "PG",
        "KO",
        "WMT",
        "XOM",
        "CVX",
        "PEP",
        "COST",
        "MCD",
        "NKE",
        "DIS",
        "HD",
        "LOW",
        "TGT",
        "SBUX",
        "CL",
    ],
    "us_finance": [
        "JPM",
        "BAC",
        "GS",
        "MS",
        "V",
        "MA",
        "BRK-B",
        "C",
        "SCHW",
        "AXP",
        "USB",
        "PNC",
        "TFC",
        "BLK",
        "SPGI",
    ],
    "us_healthcare": [
        "UNH",
        "LLY",
        "PFE",
        "ABBV",
        "MRK",
        "TMO",
        "ABT",
        "AMGN",
        "JNJ",
        "BMY",
        "GILD",
        "VRTX",
        "ISRG",
        "MDT",
        "ZTS",
    ],
    "us_industrial": [
        "CAT",
        "DE",
        "HON",
        "UNP",
        "BA",
        "GE",
        "RTX",
        "LMT",
        "MMM",
        "ETN",
        "ITW",
        "EMR",
        "PH",
        "WM",
        "RSG",
    ],
    "ru_blue_chips": [
        "SBER",
        "GAZP",
        "LKOH",
        "GMKN",
        "YNDX",
        "VTBR",
        "SBERP",
        "MGNT",
        "POLY",
        "ALRS",
    ],
    "ru_energy": [
        "ROSN",
        "LKOH",
        "NVTK",
        "TATN",
        "GAZP",
        "SNGS",
        "TRNFP",
        "IRAO",
    ],
}

# ── Known MOEX FIGIs (for future use with real MOEX symbols) ─────────────────

MOEX_FIGIS: dict[str, str] = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0",
    "LKOH": "BBG004731032",
    "VTBR": "BBG004730ZJ9",
    "SBERP": "BBG0047315Y7",
    "ROSN": "BBG004731354",
    "TATN": "BBG004RVFFC0",
    "NVTK": "BBG00475KKY8",
    "GMKN": "BBG004731489",
    "MGNT": "BBG004RVFCY3",
    "POLY": "BBG004PYF2N3",
    "ALRS": "BBG004S68B31",
    "SNGS": "BBG004S681W1",
    "TRNFP": "BBG00475K6C3",
    "IRAO": "BBG004S68473",
}

# ── CBR Rate Decisions 2023-2024 (hardcoded public data) ─────────────────────

CBR_DECISIONS: list[dict[str, object]] = [
    # 2023
    {"date": "2023-02-10", "rate_decision": 7.50, "expected_rate": 7.50, "surprise_bps": 0},
    {"date": "2023-03-17", "rate_decision": 7.50, "expected_rate": 7.50, "surprise_bps": 0},
    {"date": "2023-04-28", "rate_decision": 7.50, "expected_rate": 7.50, "surprise_bps": 0},
    {"date": "2023-06-09", "rate_decision": 7.50, "expected_rate": 7.50, "surprise_bps": 0},
    {"date": "2023-07-21", "rate_decision": 8.50, "expected_rate": 8.00, "surprise_bps": 50},
    {"date": "2023-09-15", "rate_decision": 13.00, "expected_rate": 12.00, "surprise_bps": 100},
    {"date": "2023-10-27", "rate_decision": 15.00, "expected_rate": 14.00, "surprise_bps": 100},
    {"date": "2023-12-15", "rate_decision": 16.00, "expected_rate": 15.50, "surprise_bps": 50},
    # 2024
    {"date": "2024-02-16", "rate_decision": 16.00, "expected_rate": 16.00, "surprise_bps": 0},
    {"date": "2024-03-22", "rate_decision": 16.00, "expected_rate": 16.00, "surprise_bps": 0},
    {"date": "2024-04-26", "rate_decision": 16.00, "expected_rate": 16.00, "surprise_bps": 0},
    {"date": "2024-06-07", "rate_decision": 16.00, "expected_rate": 16.00, "surprise_bps": 0},
    {"date": "2024-07-26", "rate_decision": 18.00, "expected_rate": 17.00, "surprise_bps": 100},
    {"date": "2024-09-13", "rate_decision": 19.00, "expected_rate": 18.50, "surprise_bps": 50},
    {"date": "2024-10-25", "rate_decision": 21.00, "expected_rate": 20.00, "surprise_bps": 100},
    {"date": "2024-12-20", "rate_decision": 21.00, "expected_rate": 22.00, "surprise_bps": -100},
]


def _all_us_symbols() -> list[str]:
    """Collect unique US symbols from the universe (us_* segments)."""
    seen: set[str] = set()
    result: list[str] = []
    for segment, symbols in UNIVERSE.items():
        if segment.startswith("us_"):
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    result.append(s)
    return result


def _all_moex_symbols_in_universe() -> list[str]:
    """Collect symbols from ru_* segments that are real MOEX tickers (have known FIGIs)."""
    result: list[str] = []
    for segment, symbols in UNIVERSE.items():
        if segment.startswith("ru_"):
            result.extend(s for s in symbols if s in MOEX_FIGIS)
    return result


def _write_json(path: Path, data: list[dict[str, object]]) -> None:
    """Write JSON data to file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"    Wrote {len(data)} records -> {path}")


# ── US Dividends (yfinance) ──────────────────────────────────────────────────


def fetch_us_dividends(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    """Fetch dividend history for US symbols via yfinance."""
    print("\n=== US Dividends ===")
    out = output_dir / "dividends"

    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            dividends: pd.Series = ticker.dividends  # type: ignore[assignment]

            if dividends.empty:
                print("no dividends")
                continue

            # Filter to date range (tz-aware to match yfinance index)
            idx_tz = dividends.index.tz
            ts_start = pd.Timestamp(start, tz=idx_tz)
            ts_end = pd.Timestamp(end, tz=idx_tz)
            mask = (dividends.index >= ts_start) & (dividends.index <= ts_end)
            filtered = dividends[mask]

            if filtered.empty:
                print("none in range")
                continue

            records: list[dict[str, object]] = []
            for dt, amount in filtered.items():
                records.append(
                    {
                        "ex_date": pd.Timestamp(dt).strftime("%Y-%m-%d"),  # type: ignore[arg-type]
                        "amount": round(float(amount), 4),
                    }
                )

            _write_json(out / f"{symbol}.json", records)
        except Exception as exc:
            print(f"ERROR: {exc}")


# ── US Earnings (yfinance) ───────────────────────────────────────────────────


def fetch_us_earnings(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    """Fetch earnings dates and compute SUE scores via yfinance."""
    print("\n=== US Earnings ===")
    out = output_dir / "earnings"

    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(symbol)
            earnings_df = ticker.earnings_dates

            if earnings_df is None or earnings_df.empty:
                print("no earnings data")
                continue

            # Filter to date range (tz-aware to match yfinance index)
            idx_tz = earnings_df.index.tz
            ts_start = pd.Timestamp(start, tz=idx_tz)
            ts_end = pd.Timestamp(end, tz=idx_tz)
            mask = (earnings_df.index >= ts_start) & (earnings_df.index <= ts_end)
            filtered = earnings_df[mask]

            if filtered.empty:
                print("none in range")
                continue

            records: list[dict[str, object]] = []
            for dt, row in filtered.iterrows():
                actual_eps = row.get("Reported EPS")
                expected_eps = row.get("EPS Estimate")

                record: dict[str, object] = {
                    "announcement_date": pd.Timestamp(dt).strftime("%Y-%m-%d"),  # type: ignore[arg-type]
                    "actual_eps": None,
                    "expected_eps": None,
                    "sue_score": None,
                }

                if actual_eps is not None and not (
                    isinstance(actual_eps, float) and math.isnan(actual_eps)
                ):
                    record["actual_eps"] = round(float(actual_eps), 4)

                if expected_eps is not None and not (
                    isinstance(expected_eps, float) and math.isnan(expected_eps)
                ):
                    record["expected_eps"] = round(float(expected_eps), 4)

                # SUE = (actual - expected) / |expected| * 10.0
                if (
                    record["actual_eps"] is not None
                    and record["expected_eps"] is not None
                    and record["expected_eps"] != 0
                ):
                    sue = (
                        (float(record["actual_eps"]) - float(record["expected_eps"]))
                        / abs(float(record["expected_eps"]))
                        * 10.0
                    )
                    record["sue_score"] = round(sue, 4)

                records.append(record)

            _write_json(out / f"{symbol}.json", records)
        except Exception as exc:
            print(f"ERROR: {exc}")


# ── MOEX Dividends (Tinkoff API) ────────────────────────────────────────────


def fetch_moex_dividends(
    symbols: list[str],  # noqa: ARG001 — kept for API compat; function uses MOEX_FIGIS directly
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    """Fetch dividend history for MOEX symbols via Tinkoff Invest API.

    Always fetches for all known MOEX_FIGIS tickers regardless of whether
    they appear in the backtest universe (the universe uses ETF proxies).
    """
    print("\n=== MOEX Dividends (via Tinkoff API) ===")

    # Always use the full MOEX_FIGIS list — the universe has ETF proxies,
    # but we want real MOEX dividend data for DividendGapStrategy.
    all_moex = list(MOEX_FIGIS.keys())
    if not all_moex:
        print("  No MOEX tickers configured, skipping")
        return

    print(f"  Fetching dividends for {len(all_moex)} MOEX tickers: {', '.join(all_moex)}")

    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN")

    # Try Tinkoff API first, fall back to yfinance with .ME suffix
    if token:
        _fetch_moex_dividends_tinkoff(all_moex, start, end, output_dir, token)
    else:
        print("  FINALAYZE_TINKOFF_TOKEN not set, using yfinance .ME suffix fallback")
        _fetch_moex_dividends_yfinance(all_moex, start, end, output_dir)


def _fetch_moex_dividends_tinkoff(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: Path,
    token: str,
) -> None:
    """Fetch MOEX dividends via Tinkoff Invest API."""
    try:
        from tinkoff.invest import Client  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        print("  tinkoff-investments not installed, falling back to yfinance")
        _fetch_moex_dividends_yfinance(symbols, start, end, output_dir)
        return

    out = output_dir / "dividends"
    start_dt = datetime.strptime(start, "%Y-%m-%d")  # noqa: DTZ007
    end_dt = datetime.strptime(end, "%Y-%m-%d")  # noqa: DTZ007

    with Client(token) as client:
        for symbol in symbols:
            figi = MOEX_FIGIS.get(symbol)
            if not figi:
                print(f"  {symbol}: no known FIGI, skipping")
                continue

            print(f"  {symbol} (FIGI={figi})...", end=" ", flush=True)
            try:
                response = client.instruments.get_dividends(figi=figi)
                dividends = response.dividends

                records: list[dict[str, object]] = []
                for div in dividends:
                    ex_date = div.ex_dividend_date
                    if ex_date is None:
                        continue

                    div_date = ex_date.replace(tzinfo=None)
                    if div_date < start_dt or div_date > end_dt:
                        continue

                    # Tinkoff MoneyValue: units + nano
                    amount = float(div.dividend_net.units) + float(div.dividend_net.nano) / 1e9

                    records.append(
                        {
                            "ex_date": div_date.strftime("%Y-%m-%d"),
                            "amount": round(amount, 4),
                        }
                    )

                if records:
                    _write_json(out / f"{symbol}.json", records)
                else:
                    print("none in range")
            except Exception as exc:
                print(f"ERROR: {exc}")


# yfinance suffix for Moscow Exchange tickers
_MOEX_YF_SUFFIX = ".ME"


def _fetch_moex_dividends_yfinance(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: Path,
) -> None:
    """Fetch MOEX dividends via yfinance using .ME suffix (fallback)."""
    out = output_dir / "dividends"

    for symbol in symbols:
        yf_symbol = f"{symbol}{_MOEX_YF_SUFFIX}"
        print(f"  {symbol} (yf: {yf_symbol})...", end=" ", flush=True)
        try:
            ticker = yf.Ticker(yf_symbol)
            dividends: pd.Series = ticker.dividends  # type: ignore[assignment]

            if dividends.empty:
                print("no dividends")
                continue

            idx_tz = dividends.index.tz
            ts_start = pd.Timestamp(start, tz=idx_tz)
            ts_end = pd.Timestamp(end, tz=idx_tz)
            mask = (dividends.index >= ts_start) & (dividends.index <= ts_end)
            filtered = dividends[mask]

            if filtered.empty:
                print("none in range")
                continue

            records: list[dict[str, object]] = []
            for dt, amount in filtered.items():
                records.append(
                    {
                        "ex_date": pd.Timestamp(dt).strftime("%Y-%m-%d"),  # type: ignore[arg-type]
                        "amount": round(float(amount), 4),
                    }
                )

            # Save under the original MOEX symbol name (not .ME)
            _write_json(out / f"{symbol}.json", records)
        except Exception as exc:
            print(f"ERROR: {exc}")


# ── CBR Rate Decisions ───────────────────────────────────────────────────────


def write_cbr_decisions(start: str, end: str, output_dir: Path) -> None:
    """Write hardcoded CBR rate decisions filtered to date range."""
    print("\n=== CBR Rate Decisions ===")

    filtered = [d for d in CBR_DECISIONS if start <= str(d["date"]) <= end]

    out = output_dir / "cbr" / "decisions.json"
    _write_json(out, filtered)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch event data (dividends, earnings, CBR rates) for backtesting.",
    )
    parser.add_argument(
        "--symbols-from-universe",
        action="store_true",
        help="Use the built-in UNIVERSE dict for symbol selection",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--output",
        default="results/event_data/",
        help="Output directory (default: results/event_data/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.symbols_from_universe:
        print("ERROR: --symbols-from-universe is required (only source currently supported)")
        sys.exit(1)

    output_dir = Path(PROJECT_ROOT) / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Event data builder")
    print(f"  Date range: {args.start} to {args.end}")
    print(f"  Output dir: {output_dir}")

    us_symbols = _all_us_symbols()

    print(f"  US symbols: {len(us_symbols)}")
    print(f"  MOEX tickers (for dividends): {len(MOEX_FIGIS)}")

    # 1. US Dividends
    fetch_us_dividends(us_symbols, args.start, args.end, output_dir)

    # 2. US Earnings
    fetch_us_earnings(us_symbols, args.start, args.end, output_dir)

    # 3. MOEX Dividends (always fetched for all known MOEX tickers)
    fetch_moex_dividends(list(MOEX_FIGIS.keys()), args.start, args.end, output_dir)

    # 4. CBR Rate Decisions
    write_cbr_decisions(args.start, args.end, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
