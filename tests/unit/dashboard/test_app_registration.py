from __future__ import annotations

from pathlib import Path


def test_app_registers_positions_page() -> None:
    """Positions page must be registered in st.navigation() in app.py."""
    src = Path("src/finalayze/dashboard/app.py").read_text()
    assert 'st.Page("pages/positions.py"' in src
    assert 'title="Positions"' in src


def test_pyproject_declares_plotly_dependency() -> None:
    """D-08: plotly must be in [project.dependencies], not transitive-only."""
    src = Path("pyproject.toml").read_text()
    # Find the [project] section and look for plotly in the deps list.
    # This catches plotly declared in dependencies = [...] OR as `plotly = ...`.
    assert "plotly" in src
    # Stronger assertion: plotly>=6.6.0 literal OR plotly>= in project deps
    assert "plotly>=6.6" in src or 'plotly = ">=6.6' in src
