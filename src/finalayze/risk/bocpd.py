"""Bayesian Online Changepoint Detection (Adams-MacKay 2007).

Maintains a run-length posterior updated each observation using a
Normal-Inverse-Gamma conjugate prior.  The predictive distribution at
each run length is Student-t.

Usage::

    detector = BOCPDDetector(hazard_rate=1/250)
    for x in daily_returns:
        p_cp = detector.update(x)
        if p_cp > 0.5:
            print("Changepoint detected!")

Reference:
    Adams, R. P. & MacKay, D. J. C. (2007).
    Bayesian Online Changepoint Detection. arXiv:0710.3742.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RUN_LENGTH = 500
_PRUNE_THRESHOLD = 1e-10
# Run-lengths <= this threshold are considered "recently changed".
# A step t with high P(r_t <= _CP_RUN_THRESHOLD) signals a changepoint.
_CP_RUN_THRESHOLD = 1


class BOCPDDetector:
    """Online changepoint detector using Adams-MacKay BOCPD.

    Parameters
    ----------
    hazard_rate:
        Prior probability of a changepoint at each step.
        Default ``1/250`` corresponds to ~1 changepoint per trading year.
    mu0, kappa0, alpha0, beta0:
        Normal-Inverse-Gamma prior hyper-parameters for the observation model.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 250,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        self._hazard = hazard_rate
        self._mu0 = mu0
        self._kappa0 = kappa0
        self._alpha0 = alpha0
        self._beta0 = beta0

        # Run-length posterior: R[r] = P(run_length = r | data_{1:t})
        # Initialised so that P(r=0) = 1 before any data.
        self._run_length_probs: NDArray[np.float64] = np.array([1.0])

        # Sufficient statistics for each run length (vectorised).
        # Arrays are aligned with _run_length_probs (index = run length).
        self._mu: NDArray[np.float64] = np.array([mu0])
        self._kappa: NDArray[np.float64] = np.array([kappa0])
        self._alpha: NDArray[np.float64] = np.array([alpha0])
        self._beta: NDArray[np.float64] = np.array([beta0])

        self._step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x: float) -> float:
        """Process one observation and return P(changepoint at this step).

        Returns the posterior probability that the run length is very short
        (r <= 1), which indicates a recent regime change.  This is more
        informative than the raw P(r=0) mass, which is bounded by the
        hazard rate.
        """
        self._step += 1

        # 1. Evaluate predictive probabilities under each run length.
        pred = self._predictive_pdf(x)

        # 2. Growth probabilities: existing run continues.
        growth_probs = self._run_length_probs * pred * (1.0 - self._hazard)

        # 3. Changepoint probability: new run starts.
        cp_prob = float(np.sum(self._run_length_probs * pred * self._hazard))

        # 4. Assemble new run-length distribution [cp, growth...].
        new_probs = np.empty(len(growth_probs) + 1, dtype=np.float64)
        new_probs[0] = cp_prob
        new_probs[1:] = growth_probs

        # 5. Normalise.
        evidence = new_probs.sum()
        if evidence > 0:
            new_probs /= evidence

        # 6. Update sufficient statistics (grow arrays by 1 for new run).
        self._update_sufficient_stats(x)

        # 7. Store and prune.
        self._run_length_probs = new_probs
        self._prune()

        # Return P(run_length <= _CP_RUN_THRESHOLD) as the changepoint signal.
        # During the first few steps, short run lengths are expected --
        # suppress by requiring enough history.
        upper = min(_CP_RUN_THRESHOLD + 1, len(new_probs))
        cp_signal = float(new_probs[:upper].sum())

        # Suppress the signal for the first few observations (warmup)
        # where run length is naturally short.
        warmup = _CP_RUN_THRESHOLD + 3
        if self._step <= warmup:
            return 0.0

        return cp_signal

    def reset(self) -> None:
        """Reset all state so the detector can be reused on a new series."""
        self._run_length_probs = np.array([1.0])
        self._mu = np.array([self._mu0])
        self._kappa = np.array([self._kappa0])
        self._alpha = np.array([self._alpha0])
        self._beta = np.array([self._beta0])
        self._step = 0

    def is_changepoint(self, x: float, threshold: float = 0.5) -> bool:
        """Convenience: return True if P(cp) exceeds *threshold*."""
        return self.update(x) > threshold

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _predictive_pdf(self, x: float) -> NDArray[np.float64]:
        """Evaluate Student-t predictive density at *x* for every run length.

        Under the NIG posterior the predictive is::

            t_{2*alpha}(mu, beta*(kappa+1) / (alpha*kappa))
        """
        df = 2.0 * self._alpha
        loc = self._mu
        scale2 = self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa)
        scale = np.sqrt(scale2)

        from math import lgamma as _lg  # noqa: PLC0415

        # Vectorised log-pdf of Student-t.
        z = (x - loc) / scale
        log_pdf = (
            np.array([_lg((d + 1) / 2) - _lg(d / 2) for d in df])
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
            - (df + 1) / 2.0 * np.log(1.0 + z**2 / df)
        )
        result: NDArray[np.float64] = np.exp(log_pdf)
        return result

    def _update_sufficient_stats(self, x: float) -> None:
        """Grow sufficient-statistic arrays by prepending the prior (new run).

        Then update all existing entries with the new observation.
        """
        # Update existing entries *before* prepending.
        kappa_new = self._kappa + 1.0
        mu_new = (self._kappa * self._mu + x) / kappa_new
        alpha_new = self._alpha + 0.5
        beta_new = self._beta + 0.5 * self._kappa * (x - self._mu) ** 2 / kappa_new

        # Prepend prior for the fresh run (run length = 0).
        self._mu = np.concatenate([[self._mu0], mu_new])
        self._kappa = np.concatenate([[self._kappa0], kappa_new])
        self._alpha = np.concatenate([[self._alpha0], alpha_new])
        self._beta = np.concatenate([[self._beta0], beta_new])

    def _prune(self) -> None:
        """Prune negligible run lengths and cap at _MAX_RUN_LENGTH."""
        n = len(self._run_length_probs)

        # Hard cap on length.
        if n > _MAX_RUN_LENGTH:
            self._run_length_probs = self._run_length_probs[:_MAX_RUN_LENGTH]
            self._mu = self._mu[:_MAX_RUN_LENGTH]
            self._kappa = self._kappa[:_MAX_RUN_LENGTH]
            self._alpha = self._alpha[:_MAX_RUN_LENGTH]
            self._beta = self._beta[:_MAX_RUN_LENGTH]
            # Re-normalise after truncation.
            total = self._run_length_probs.sum()
            if total > 0:
                self._run_length_probs /= total

        # Soft prune: zero out negligible entries (keeps array contiguous).
        mask = self._run_length_probs < _PRUNE_THRESHOLD
        self._run_length_probs[mask] = 0.0


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def detect_changepoints(
    returns: list[float],
    hazard_rate: float = 1 / 250,
    threshold: float = 0.5,
) -> list[int]:
    """Return indices where changepoints are detected in *returns*.

    Parameters
    ----------
    returns:
        Sequence of observations (e.g. daily log-returns).
    hazard_rate:
        Prior changepoint probability per step.
    threshold:
        Detection threshold for P(changepoint).

    Returns
    -------
    list[int]
        Indices into *returns* where P(cp) > threshold.
    """
    detector = BOCPDDetector(hazard_rate=hazard_rate)
    changepoints: list[int] = []
    for i, x in enumerate(returns):
        if detector.is_changepoint(x, threshold=threshold):
            changepoints.append(i)
    return changepoints
