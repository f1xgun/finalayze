"""Backward-compatibility shim -- alerts moved to finalayze.api.alerts.

Makes ``finalayze.core.alerts`` an alias for the canonical module so that
both ``from finalayze.core.alerts import X`` and
``patch("finalayze.core.alerts.X")`` continue to work transparently.
"""

from __future__ import annotations

import sys

import finalayze.api.alerts as _canonical

sys.modules[__name__] = _canonical
