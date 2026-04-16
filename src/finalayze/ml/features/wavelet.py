"""Wavelet energy feature computation (Layer 3)."""

from __future__ import annotations

import logging

import numpy as np

_WAVELET_LEVEL = 3
_MIN_WAVELET_SAMPLES = 16  # pywt.wavedec('db4', level=3) needs at least 2^level samples

_log = logging.getLogger(__name__)

try:
    import pywt as _pywt

    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _pywt = None
    _HAS_PYWT = False


def compute_wavelet_features(log_returns: list[float]) -> dict[str, float]:
    """Compute wavelet energy features from log returns via Daubechies-4 decomposition.

    Decomposes the signal into 1 approximation level and 3 detail levels.
    Returns the fraction of total energy in each level (normalized so they sum to ~1.0).

    If pywt is not available or there is insufficient data, returns 0.0 for all 4 features.
    """
    _zero = {
        "wavelet_approx_energy": 0.0,
        "wavelet_detail1_energy": 0.0,
        "wavelet_detail2_energy": 0.0,
        "wavelet_detail3_energy": 0.0,
    }

    if not _HAS_PYWT or len(log_returns) < _MIN_WAVELET_SAMPLES:
        return _zero

    try:
        coeffs = _pywt.wavedec(log_returns, "db4", level=_WAVELET_LEVEL)
    except Exception:
        _log.debug("Wavelet decomposition failed, returning zeros")
        return _zero

    # coeffs = [cA3, cD3, cD2, cD1]
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    total_energy = sum(energies)

    if total_energy <= 0.0:
        return _zero

    return {
        "wavelet_approx_energy": energies[0] / total_energy,
        "wavelet_detail3_energy": energies[1] / total_energy,
        "wavelet_detail2_energy": energies[2] / total_energy,
        "wavelet_detail1_energy": energies[3] / total_energy,
    }
