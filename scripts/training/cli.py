"""CLI entry point for the training pipeline.

Parses command-line arguments and dispatches training to either
walk-forward or single-segment modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure src/ and project root are importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))  # for config.settings

# torch must be imported before lightgbm to prevent OpenMP thread-pool conflicts
import torch  # noqa: F401
from config.segments import _BOOTSTRAP_SEGMENT_SYMBOLS, DEFAULT_SEGMENTS
from scripts.training.data_loader import (
    build_market_data_loader,
    get_lookback_days,
)
from scripts.training.dataset_builder import (
    LABEL_MODE_DIRECTION,
    LABEL_MODE_TREND_SCANNING,
    LABEL_MODE_TRIPLE_BARRIER,
)
from scripts.training.model_trainer import FEAT_SEL_EFFICIENT, FEAT_SEL_MI
from scripts.training.walk_forward import (
    apply_bh_across_segments,
    train_walk_forward,
)

from finalayze.markets.liquidity import select_segment_symbols

# Default output directory
DEFAULT_OUTPUT_DIR = "models/"


def _bootstrap_for(segment_id: str) -> list[str]:
    """Prior-hardcoded bootstrap list for ``segment_id`` (WR-04: missing-key safe).

    A new enabled MOEX stock segment added to ``DEFAULT_SEGMENTS`` without a matching
    ``_BOOTSTRAP_SEGMENT_SYMBOLS`` key must NOT raise ``KeyError`` at module import time
    (that would break ``import scripts.training.cli`` and every consumer before any clear
    error surfaces). Use ``.get`` with an empty default; once the committed snapshot lands
    the selector supplies the liquid set regardless. Warn (non-fatal, self-healing) so the
    missing key is still visible to an operator.
    """
    if segment_id not in _BOOTSTRAP_SEGMENT_SYMBOLS:
        warnings.warn(
            f"no _BOOTSTRAP_SEGMENT_SYMBOLS key for enabled MOEX segment {segment_id!r}; "
            "bootstrapping with [] until the liquidity snapshot supplies it",
            stacklevel=2,
        )
    return _BOOTSTRAP_SEGMENT_SYMBOLS.get(segment_id, [])


# Map segment_id -> representative symbols for training data
SEGMENT_SYMBOLS: dict[str, list[str]] = {
    "us_tech": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "NVDA",
        "META",
        "AMZN",
        "TSLA",
        "CRM",
        "ADBE",
        "INTC",
        "AMD",
        "AVGO",
        "CSCO",
        "ORCL",
        "QCOM",
    ],
    "us_healthcare": [
        "JNJ",
        "PFE",
        "UNH",
        "ABBV",
        "MRK",
        "LLY",
        "TMO",
        "ABT",
        "BMY",
        "AMGN",
        "GILD",
        "MDT",
    ],
    "us_finance": [
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
        "BLK",
        "SCHW",
        "AXP",
        "USB",
        "PNC",
        "TFC",
    ],
    "us_broad": ["SPY", "QQQ", "DIA", "IWM", "VTI"],
    # ru_* SHARE training universes resolve through the liquidity selector (LIQ-08) --
    # the SAME single source as config.segments and run_iteration.UNIVERSE. The old
    # hardcoded lists carried DELISTED tickers (YNDX/TCSG/SNGS/CIAN); the selector
    # supersedes them. train_walk_forward is unchanged (D-10) -- only the set widens.
    #
    # 66-04: derive a key for EVERY enabled MOEX ru_* SHARE segment directly from
    # DEFAULT_SEGMENTS (single source -- it can never drift again, the same missing-keys
    # bug fixed in run_iteration.UNIVERSE: only ru_blue_chips/ru_energy/ru_tech/ru_finance
    # had keys here, so the newer sector segments were silently untrainable).
    # Prior-list bootstrap (pre-66-04): when the committed snapshot FILE is absent the
    # selector returns these hardcoded lists (single source: config._BOOTSTRAP_SEGMENT_SYMBOLS)
    # so this seam stays populated and identical to the other two; once the snapshot lands
    # the liquid set replaces them.
    **{
        seg.segment_id: select_segment_symbols(
            seg.segment_id, bootstrap=_bootstrap_for(seg.segment_id)
        )
        for seg in DEFAULT_SEGMENTS
        if seg.market == "moex" and seg.enabled and seg.instrument_type == "stock"
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost + LightGBM + CatBoost models per segment"
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Segment ID to train (default: all segments)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--label-mode",
        default=LABEL_MODE_TRIPLE_BARRIER,
        choices=[LABEL_MODE_TRIPLE_BARRIER, LABEL_MODE_DIRECTION, LABEL_MODE_TREND_SCANNING],
        help=(
            f"Labeling mode: '{LABEL_MODE_TRIPLE_BARRIER}' uses ATR-scaled triple barrier "
            f"labels (default), '{LABEL_MODE_DIRECTION}' uses simple next-bar direction labels, "
            f"'{LABEL_MODE_TREND_SCANNING}' uses OLS trend-scanning labels (Prado 2020)."
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help="Use walk-forward validation (D1) instead of single split.",
    )
    parser.add_argument(
        "--excess-returns",
        action="store_true",
        default=False,
        help=(
            "Use market-neutral (excess return) labels by subtracting "
            "benchmark return (SPY for US, IMOEX for MOEX). "
            "Only applies to triple_barrier label mode."
        ),
    )
    parser.add_argument(
        "--force-save",
        action="store_true",
        default=False,
        help=(
            "Save models even when quality gates fail. "
            "For development use only -- production models should pass gates."
        ),
    )
    parser.add_argument(
        "--sequential-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use sequential bootstrapping (AFML Ch. 4) to debias training samples "
            "by reducing overlap redundancy. Requires triple_barrier labels with hold_bars. "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--feature-selection",
        default=FEAT_SEL_EFFICIENT,
        choices=[FEAT_SEL_MI, FEAT_SEL_EFFICIENT],
        help=(
            f"Feature selection method: '{FEAT_SEL_MI}' uses Mutual Information only, "
            f"'{FEAT_SEL_EFFICIENT}' uses MI weighted by feature complexity "
            f"(prefers cheap informative features). Default: {FEAT_SEL_EFFICIENT}."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:  # noqa: PLR0912, PLR0915
    """Entry point."""
    from types import SimpleNamespace  # noqa: PLC0415

    from scripts.training.model_trainer import train_one_segment  # noqa: PLC0415

    from finalayze.core.schemas import MarketContext  # noqa: PLC0415

    args = parse_args()
    output_dir = Path(args.output_dir)
    label_mode: str = args.label_mode
    walk_forward: bool = args.walk_forward
    excess_returns: bool = args.excess_returns
    force_save: bool = args.force_save
    seq_bootstrap: bool = args.sequential_bootstrap
    feat_sel_mode: str = args.feature_selection

    if args.segment:
        segments = {args.segment: SEGMENT_SYMBOLS.get(args.segment, [])}
    else:
        segments = SEGMENT_SYMBOLS

    print(
        f"Label mode: {label_mode}, Walk-forward: {walk_forward}, "
        f"Excess returns: {excess_returns}, Force save: {force_save}, "
        f"Sequential bootstrap: {seq_bootstrap}, Feature selection: {feat_sel_mode}"
    )

    # Build MarketDataLoader -- single instance reused across all segments.
    segment_ids = list(segments.keys())
    loader = build_market_data_loader(segment_ids)

    # Collect p-values for BH correction (D3) across all segments
    segment_accuracies: dict[str, float] = {}

    try:
        for segment_id, symbols in segments.items():
            # Load ambient market data for this segment's full training window.
            # The loader routes by market: US -> SPY + ^VIX; MOEX -> IMOEX + CBR + turnover + Brent.
            market_id = "moex" if segment_id.startswith("ru_") else "us"
            lookback_days = get_lookback_days(segment_id)
            end_date = datetime.now(tz=UTC).date()
            start_date = (datetime.now(tz=UTC) - timedelta(days=lookback_days)).date()
            _seg_cfg = SimpleNamespace(market=market_id, symbols=symbols)
            try:
                market_context: MarketContext | None = loader.load(_seg_cfg, start_date, end_date)
                if loader.fetch_failures:
                    print(
                        f"[{segment_id}] Market data warnings: {', '.join(loader.fetch_failures)}"
                    )
            except Exception as exc:
                print(f"[{segment_id}] Could not load market context ({exc}), proceeding without.")
                market_context = None

            try:
                if walk_forward:
                    gate_rates = train_walk_forward(
                        segment_id=segment_id,
                        symbols=symbols,
                        output_dir=output_dir,
                        label_mode=label_mode,
                        excess_returns=excess_returns,
                        force_save=force_save,
                        seq_bootstrap=seq_bootstrap,
                        market_context=market_context,
                        feat_sel_mode=feat_sel_mode,
                    )
                    if gate_rates and "accuracy" in gate_rates:
                        # Load best accuracy from saved results
                        results_path = output_dir / segment_id / "wf_gate_results.json"
                        if results_path.exists():
                            wf_data = json.loads(results_path.read_text())
                            segment_accuracies[segment_id] = wf_data.get("best_accuracy", 0.5)
                else:
                    train_one_segment(
                        segment_id=segment_id,
                        symbols=symbols,
                        output_dir=output_dir,
                        label_mode=label_mode,
                        excess_returns=excess_returns,
                        seq_bootstrap=seq_bootstrap,
                        market_context=market_context,
                        feat_sel_mode=feat_sel_mode,
                    )
            except FileNotFoundError as exc:
                print(f"[{segment_id}] FileNotFoundError -- {exc}, skipping.")
            except Exception as exc:
                print(f"[{segment_id}] Unexpected error -- {exc}, skipping.")
    finally:
        loader.close()

    # BH correction across all segments (D3)
    if walk_forward and segment_accuracies:
        apply_bh_across_segments(segment_accuracies, output_dir)
