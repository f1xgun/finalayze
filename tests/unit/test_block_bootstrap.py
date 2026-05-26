"""S3.3 — Stationary block bootstrap for Monte Carlo.

Naive iid resampling of daily equity returns destroys volatility clustering
and momentum regimes, producing overly-optimistic max-drawdown CIs.
Politis-Romano (1994) stationary block bootstrap fixes that by drawing
geometrically-distributed-length blocks with wrap-around.

Contract:
  BLK-01: stationary_block_sample returns the requested length, drawn from
          the source series only, with reproducibility under a seeded RNG.
  BLK-02: mean_block_length=1 is equivalent to iid (each draw restarts).
  BLK-03: For an AR(1) series with strong autocorrelation, the lag-1
          autocorrelation of a block-bootstrap sample is closer to the
          original than an iid sample. Probabilistic — large n_samples
          and seeded RNG keep the test stable.
  BLK-04: bootstrap_from_snapshots dispatches on method='iid' vs
          'stationary_block'; both produce structurally-valid results.
  BLK-05: ValueError on mean_block_length < 1.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from finalayze.backtest.monte_carlo import (
    bootstrap_from_snapshots,
    stationary_block_sample,
)
from finalayze.core.schemas import PortfolioState

_SEED = 7
_BLOCK_LEN = 10
_N_OBS = 500
_N_SAMPLE = 500
_N_BOOTSTRAP = 200
_AR1_PHI = 0.7
_AR1_SIGMA = 1.0


def _make_ar1(n: int, phi: float, sigma: float, seed: int) -> np.ndarray:
    """Generate an AR(1) series: x_t = phi * x_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _lag1_autocorr(x: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation; returns 0 if std is zero."""
    if x.size < 2 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _make_snapshots(n_bars: int, returns: list[float], start_equity: float = 100_000.0):
    """Build a synthetic snapshot trail with prescribed daily % returns."""
    assert len(returns) == n_bars - 1
    snaps = []
    equity = start_equity
    base = datetime(2024, 1, 1, tzinfo=UTC)
    snaps.append(
        PortfolioState(
            cash=Decimal(str(equity)),
            positions={},
            equity=Decimal(str(equity)),
            timestamp=base,
        )
    )
    for i, r in enumerate(returns, start=1):
        equity *= 1 + r / 100
        snaps.append(
            PortfolioState(
                cash=Decimal(str(equity)),
                positions={},
                equity=Decimal(str(equity)),
                timestamp=base + timedelta(days=i),
            )
        )
    return snaps


# ─── BLK-01 ──────────────────────────────────────────────────────────────────
def test_block_sample_shape_and_membership() -> None:
    data = np.arange(20, dtype=float)
    rng = np.random.default_rng(_SEED)
    sample = stationary_block_sample(data, n_samples=_N_SAMPLE, mean_block_length=5, rng=rng)
    assert sample.shape == (_N_SAMPLE,)
    assert set(sample).issubset(set(data.tolist()))


def test_block_sample_reproducible_with_same_seed() -> None:
    data = np.linspace(-1, 1, 30)
    a = stationary_block_sample(data, _N_SAMPLE, 5, np.random.default_rng(_SEED))
    b = stationary_block_sample(data, _N_SAMPLE, 5, np.random.default_rng(_SEED))
    assert np.array_equal(a, b)


# ─── BLK-02 ──────────────────────────────────────────────────────────────────
def test_block_length_one_is_iid_like() -> None:
    """At L=1 every step restarts → lag-1 autocorr collapses to ~0."""
    data = _make_ar1(_N_OBS, _AR1_PHI, _AR1_SIGMA, seed=_SEED)
    sample = stationary_block_sample(
        data, _N_SAMPLE, mean_block_length=1, rng=np.random.default_rng(_SEED)
    )
    assert abs(_lag1_autocorr(sample)) < 0.15  # near zero (allow noise)


# ─── BLK-03 ──────────────────────────────────────────────────────────────────
def test_block_bootstrap_preserves_autocorr_better_than_iid() -> None:
    """Lag-1 autocorr of a block sample > lag-1 of an iid sample (on AR(1))."""
    data = _make_ar1(_N_OBS, _AR1_PHI, _AR1_SIGMA, seed=_SEED)
    rng_block = np.random.default_rng(_SEED)
    rng_iid = np.random.default_rng(_SEED + 1)

    block_sample = stationary_block_sample(data, _N_SAMPLE, _BLOCK_LEN, rng_block)
    iid_indices = rng_iid.choice(len(data), size=_N_SAMPLE, replace=True)
    iid_sample = data[iid_indices]

    block_ac = abs(_lag1_autocorr(block_sample))
    iid_ac = abs(_lag1_autocorr(iid_sample))

    # Strong AR(1) -> original ~0.7. Block should beat iid by a clear margin.
    assert block_ac > iid_ac + 0.2, f"block={block_ac:.3f} vs iid={iid_ac:.3f}"


# ─── BLK-04 ──────────────────────────────────────────────────────────────────
def test_bootstrap_from_snapshots_block_method_produces_result() -> None:
    rng_data = np.random.default_rng(_SEED)
    daily_pct = rng_data.normal(0.1, 1.0, size=100).tolist()
    snaps = _make_snapshots(101, daily_pct)

    res = bootstrap_from_snapshots(
        snaps,
        n_simulations=_N_BOOTSTRAP,
        seed=_SEED,
        method="stationary_block",
        mean_block_length=_BLOCK_LEN,
    )

    assert res.n_simulations == _N_BOOTSTRAP
    assert res.n_trades == len(daily_pct)
    # CIs should be well-defined floats
    assert res.sharpe_ratio.lower <= res.sharpe_ratio.upper
    assert res.max_drawdown.lower <= res.max_drawdown.upper


def test_bootstrap_from_snapshots_iid_method_still_works() -> None:
    """Back-compat: explicit method='iid' continues to work."""
    rng_data = np.random.default_rng(_SEED)
    daily_pct = rng_data.normal(0.0, 0.5, size=50).tolist()
    snaps = _make_snapshots(51, daily_pct)

    res = bootstrap_from_snapshots(snaps, n_simulations=_N_BOOTSTRAP, seed=_SEED, method="iid")
    assert res.n_trades == len(daily_pct)
    assert res.n_simulations == _N_BOOTSTRAP


def test_block_bootstrap_yields_wider_drawdown_ci_than_iid() -> None:
    """For a strongly autocorrelated equity curve, the block-bootstrap
    max-drawdown upper bound should be at least as large as iid's
    (autocorrelated draw downs are deeper than iid suggests)."""
    # AR(1) shocks compounded into an equity curve = serially-correlated returns.
    data = _make_ar1(200, _AR1_PHI, 0.6, seed=_SEED).tolist()
    snaps = _make_snapshots(201, data)

    res_iid = bootstrap_from_snapshots(snaps, n_simulations=_N_BOOTSTRAP, seed=_SEED, method="iid")
    res_block = bootstrap_from_snapshots(
        snaps,
        n_simulations=_N_BOOTSTRAP,
        seed=_SEED,
        method="stationary_block",
        mean_block_length=_BLOCK_LEN,
    )

    # Allow tiny slack for sampling variance; key is block isn't drastically
    # tighter (which would be the iid-style optimism we are trying to avoid).
    assert res_block.max_drawdown.upper >= res_iid.max_drawdown.upper - 0.5


# ─── BLK-05 ──────────────────────────────────────────────────────────────────
def test_block_length_zero_raises() -> None:
    with pytest.raises(ValueError, match="mean_block_length must be >= 1"):
        stationary_block_sample(
            np.arange(5), n_samples=3, mean_block_length=0, rng=np.random.default_rng(_SEED)
        )


def test_block_sample_empty_data_returns_empty() -> None:
    out = stationary_block_sample(
        np.array([], dtype=float),
        n_samples=10,
        mean_block_length=5,
        rng=np.random.default_rng(_SEED),
    )
    assert out.size == 0
