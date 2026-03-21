#!/usr/bin/env python3
"""Capital ladder validation for MOEX instruments.

Validates that position sizing produces valid lot sizes at each capital tier
for each rollout phase. Used before going live at a new capital level.

Usage:
    uv run python scripts/validate_capital_ladder.py
    uv run python scripts/validate_capital_ladder.py --phase minimal
    uv run python scripts/validate_capital_ladder.py --tiers 50000,100000
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finalayze.core.modes import RolloutPhase
from finalayze.risk.rollout import ROLLOUT_LIMITS

# Representative MOEX prices (approximate 2026 market prices)
# These are for lot-size viability validation, not trading decisions
DEFAULT_PRICES: dict[str, Decimal] = {
    "SBER": Decimal(300),
    "GAZP": Decimal(150),
    "LKOH": Decimal(7000),
    "GMKN": Decimal(14000),
    "YNDX": Decimal(4000),
    "NVTK": Decimal(1100),
    "VTBR": Decimal("0.02"),
    "ROSN": Decimal(500),
    "MTSS": Decimal(300),
    "MGNT": Decimal(5000),
    "ALRS": Decimal(70),
    "PLZL": Decimal(14000),
    "TATN": Decimal(600),
    "SNGS": Decimal(30),
    "CHMF": Decimal(1200),
}

DEFAULT_LOT_SIZES: dict[str, int] = {
    "SBER": 10,
    "GAZP": 10,
    "LKOH": 1,
    "GMKN": 1,
    "YNDX": 1,
    "NVTK": 1,
    "VTBR": 10000,
    "ROSN": 1,
    "MTSS": 10,
    "MGNT": 1,
    "ALRS": 10,
    "PLZL": 1,
    "TATN": 1,
    "SNGS": 100,
    "CHMF": 1,
}

DEFAULT_TIERS = [
    Decimal(50000),
    Decimal(150000),
    Decimal(500000),
    Decimal(2500000),
]


def validate_position(
    capital: Decimal,
    max_position_pct: Decimal,
    price: Decimal,
    lot_size: int,
) -> dict:
    """Validate position sizing for a single instrument at a capital tier.

    Returns dict with: target_value, lots, actual_shares, actual_value, viable, utilization_pct.
    """
    target_value = capital * max_position_pct
    if price <= 0:
        return {
            "target_value": target_value,
            "lots": 0,
            "actual_shares": 0,
            "actual_value": Decimal(0),
            "viable": False,
            "utilization_pct": 0.0,
        }
    shares_target = target_value / price
    lots = int(shares_target) // lot_size
    actual_shares = lots * lot_size
    actual_value = Decimal(actual_shares) * price
    utilization = float(actual_value / target_value * 100) if target_value > 0 else 0.0
    return {
        "target_value": target_value,
        "lots": lots,
        "actual_shares": actual_shares,
        "actual_value": actual_value,
        "viable": lots >= 1,
        "utilization_pct": round(utilization, 1),
    }


def run_ladder(
    phases: list[RolloutPhase] | None = None,
    tiers: list[Decimal] | None = None,
    prices: dict[str, Decimal] | None = None,
    lot_sizes: dict[str, int] | None = None,
) -> list[dict]:
    """Run capital ladder validation across tiers, phases, and instruments.

    Returns list of result dicts with keys:
    tier, phase, symbol, price, lot_size, target_value, lots, actual_value, viable, utilization_pct
    """
    phases = phases or list(RolloutPhase)
    tiers = tiers or DEFAULT_TIERS
    prices = prices or DEFAULT_PRICES
    lot_sizes = lot_sizes or DEFAULT_LOT_SIZES

    results = []
    for tier in tiers:
        for phase in phases:
            limits = ROLLOUT_LIMITS[phase]
            for symbol in sorted(prices.keys()):
                if symbol not in lot_sizes:
                    continue
                price = prices[symbol]
                lot_size = lot_sizes[symbol]
                pos = validate_position(tier, limits.max_position_pct, price, lot_size)
                results.append(
                    {
                        "tier": tier,
                        "phase": phase.value,
                        "symbol": symbol,
                        "price": price,
                        "lot_size": lot_size,
                        **pos,
                    }
                )
    return results


def print_report(results: list[dict]) -> None:
    """Print formatted capital ladder report."""
    print(
        f"{'Tier':>12} {'Phase':<10} {'Symbol':<6} {'Price':>8} {'Lot':>6} "
        f"{'Target':>10} {'Lots':>5} {'Actual':>10} {'Util%':>6} {'OK':>3}"
    )
    print("-" * 90)
    for r in results:
        ok = "YES" if r["viable"] else "NO"
        print(
            f"{r['tier']:>12} {r['phase']:<10} {r['symbol']:<6} {r['price']:>8} "
            f"{r['lot_size']:>6} {r['target_value']:>10.0f} {r['lots']:>5} "
            f"{r['actual_value']:>10.0f} {r['utilization_pct']:>5.1f}% {ok:>3}"
        )


def main() -> int:
    """Run capital ladder validation CLI."""
    parser = argparse.ArgumentParser(description="Capital ladder validation for MOEX")
    parser.add_argument(
        "--phase",
        choices=["minimal", "standard", "full"],
        help="Only validate a single rollout phase",
    )
    parser.add_argument("--tiers", help="Comma-separated capital tiers in RUB (e.g., 50000,150000)")
    args = parser.parse_args()

    phases = [RolloutPhase(args.phase)] if args.phase else None
    tiers = [Decimal(t.strip()) for t in args.tiers.split(",")] if args.tiers else None

    results = run_ladder(phases=phases, tiers=tiers)
    print_report(results)

    not_viable = [r for r in results if not r["viable"]]
    if not_viable:
        print(
            f"\nWARNING: {len(not_viable)} instrument/tier/phase combinations "
            f"are NOT viable (0 lots)"
        )
        for r in not_viable:
            print(
                f"  - {r['symbol']} at {r['tier']} RUB ({r['phase']}): "
                f"target={r['target_value']:.0f} RUB, price={r['price']}, "
                f"lot_size={r['lot_size']}"
            )
        return 1
    print(f"\nAll {len(results)} combinations are viable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
