"""Tests for capital ladder validation script (ROLL-03)."""

from __future__ import annotations

# Import from the script module (project root scripts/)
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.validate_capital_ladder import run_ladder, validate_position


class TestValidatePosition:
    def test_validate_position_viable(self) -> None:
        """500K capital, 3% max, price=300, lot=10 -> viable with lots>=1."""
        result = validate_position(
            capital=Decimal(500000),
            max_position_pct=Decimal("0.03"),
            price=Decimal(300),
            lot_size=10,
        )
        assert result["viable"] is True
        assert result["lots"] == 5  # target=15000, shares=50, lots=5
        assert result["actual_shares"] == 50
        assert result["actual_value"] == Decimal(15000)

    def test_validate_position_not_viable(self) -> None:
        """50K capital, 3% max, price=7000, lot=1 -> target=1500, lots=0, not viable."""
        result = validate_position(
            capital=Decimal(50000),
            max_position_pct=Decimal("0.03"),
            price=Decimal(7000),
            lot_size=1,
        )
        assert result["viable"] is False
        assert result["lots"] == 0


class TestCapitalTiers:
    def test_50k_tier(self) -> None:
        """At 50K MINIMAL, expensive instruments (LKOH ~7000) produce 0 lots."""
        # LKOH: price=7000, lot_size=1, target=50000*0.03=1500, lots=0
        result = validate_position(
            capital=Decimal(50000),
            max_position_pct=Decimal("0.03"),
            price=Decimal(7000),
            lot_size=1,
        )
        assert not result["viable"]

        # SBER: price=300, lot_size=10, target=1500, shares=5, lots=0 (5//10=0)
        result_sber = validate_position(
            capital=Decimal(50000),
            max_position_pct=Decimal("0.03"),
            price=Decimal(300),
            lot_size=10,
        )
        # 1500/300 = 5 shares, 5 // 10 = 0 lots -- NOT viable at this small tier
        assert not result_sber["viable"]

    def test_2500k_tier(self) -> None:
        """At 2.5M MINIMAL (3%), all standard instruments should be viable."""
        # Target = 2500000 * 0.03 = 75000
        # LKOH: 75000/7000 = 10.7 -> lots=10, viable
        result_lkoh = validate_position(
            capital=Decimal(2500000),
            max_position_pct=Decimal("0.03"),
            price=Decimal(7000),
            lot_size=1,
        )
        assert result_lkoh["viable"]
        assert result_lkoh["lots"] >= 1

        # SBER: 75000/300 = 250 shares, 250//10 = 25 lots
        result_sber = validate_position(
            capital=Decimal(2500000),
            max_position_pct=Decimal("0.03"),
            price=Decimal(300),
            lot_size=10,
        )
        assert result_sber["viable"]
        assert result_sber["lots"] >= 1

        # GMKN: 75000/14000 = 5.35 -> lots=5, viable
        result_gmkn = validate_position(
            capital=Decimal(2500000),
            max_position_pct=Decimal("0.03"),
            price=Decimal(14000),
            lot_size=1,
        )
        assert result_gmkn["viable"]


class TestLadderReport:
    def test_ladder_report(self) -> None:
        """run_ladder() returns dicts with required keys."""
        from finalayze.core.modes import RolloutPhase

        results = run_ladder(
            phases=[RolloutPhase.MINIMAL],
            tiers=[Decimal(500000)],
            prices={"SBER": Decimal(300)},
            lot_sizes={"SBER": 10},
        )
        assert len(results) == 1
        r = results[0]
        expected_keys = {
            "tier",
            "phase",
            "symbol",
            "price",
            "lot_size",
            "target_value",
            "lots",
            "actual_value",
            "viable",
            "utilization_pct",
        }
        assert expected_keys.issubset(set(r.keys()))
        assert r["tier"] == Decimal(500000)
        assert r["phase"] == "minimal"
        assert r["symbol"] == "SBER"
        assert r["viable"] is True
