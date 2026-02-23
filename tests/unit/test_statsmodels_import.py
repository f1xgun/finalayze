"""Smoke test: statsmodels is importable and coint() is callable."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_coint_importable() -> None:
    from statsmodels.tsa.stattools import coint  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 100
    x = rng.standard_normal(n).cumsum()
    y = x + rng.standard_normal(n) * 0.1  # strongly cointegrated
    score, pvalue, _ = coint(x, y)
    assert isinstance(score, float)
    assert 0.0 <= pvalue <= 1.0
