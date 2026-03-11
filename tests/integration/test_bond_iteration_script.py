"""Integration tests for bond and portfolio iteration scripts.

Verifies that scripts can be imported and have the expected public API.
Actual execution requires FINALAYZE_TINKOFF_TOKEN and is not tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Scripts live in the project root's scripts/ dir, not in a package.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent.parent / "scripts")


@pytest.fixture(autouse=True)
def _scripts_on_path() -> None:  # noqa: PT004
    """Ensure scripts/ is importable."""
    if _SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _SCRIPTS_DIR)


class TestRunBondIterationImports:
    """Verify run_bond_iteration.py has expected structure."""

    def test_main_function_exists(self) -> None:
        import run_bond_iteration

        assert hasattr(run_bond_iteration, "main")
        assert callable(run_bond_iteration.main)

    def test_bond_universe_defined(self) -> None:
        import run_bond_iteration

        assert hasattr(run_bond_iteration, "BOND_UNIVERSE")
        universe = run_bond_iteration.BOND_UNIVERSE
        assert "ru_ofz_pd" in universe
        assert "ru_ofz_pk" in universe
        assert len(universe["ru_ofz_pd"]) > 0
        assert len(universe["ru_ofz_pk"]) > 0

    def test_ofz_tickers_defined(self) -> None:
        import run_bond_iteration

        assert hasattr(run_bond_iteration, "OFZ_PD_TICKERS")
        assert hasattr(run_bond_iteration, "OFZ_PK_TICKERS")
        assert len(run_bond_iteration.OFZ_PD_TICKERS) >= 4
        assert len(run_bond_iteration.OFZ_PK_TICKERS) >= 4

    def test_helper_functions_exist(self) -> None:
        import run_bond_iteration

        assert callable(getattr(run_bond_iteration, "_load_preset", None))
        assert callable(getattr(run_bond_iteration, "_make_tinkoff_fetcher", None))
        assert callable(getattr(run_bond_iteration, "_fetch_bond_data", None))
        assert callable(getattr(run_bond_iteration, "_build_carry_strategy", None))
        assert callable(getattr(run_bond_iteration, "_build_duration_rotation_strategy", None))
        assert callable(getattr(run_bond_iteration, "_build_cbr_event_strategy", None))
        assert callable(getattr(run_bond_iteration, "_run_bond_segment", None))


class TestRunPortfolioIterationImports:
    """Verify run_portfolio_iteration.py has expected structure."""

    def test_main_function_exists(self) -> None:
        import run_portfolio_iteration

        assert hasattr(run_portfolio_iteration, "main")
        assert callable(run_portfolio_iteration.main)

    def test_allocate_cash(self) -> None:
        from decimal import Decimal

        import run_portfolio_iteration

        allocations = run_portfolio_iteration._allocate_cash(Decimal(1_000_000))
        assert "core" in allocations
        assert "strategic" in allocations
        assert "tactical" in allocations
        assert "short" in allocations

        # Verify allocations sum to total
        total = sum(allocations.values())
        assert total == Decimal(1_000_000)

    def test_allocate_cash_percentages(self) -> None:
        from decimal import Decimal

        import run_portfolio_iteration

        allocations = run_portfolio_iteration._allocate_cash(Decimal(1_000_000))
        # Core = 45%, Strategic = 27.5%, Tactical = 17.5%, Short = 10%
        assert allocations["core"] == Decimal("450000.00")
        assert allocations["strategic"] == Decimal("275000.0")
        assert allocations["tactical"] == Decimal("175000.0")
        assert allocations["short"] == Decimal("100000.00")

    def test_layer_runner_functions_exist(self) -> None:
        import run_portfolio_iteration

        assert callable(getattr(run_portfolio_iteration, "_run_core_layer", None))
        assert callable(getattr(run_portfolio_iteration, "_run_strategic_layer", None))
        assert callable(getattr(run_portfolio_iteration, "_run_tactical_layer", None))
        assert callable(getattr(run_portfolio_iteration, "_run_short_layer", None))
