"""Russian-securities tax-optimization decision-support engine (first slice).

Decision-support ONLY: this package computes action items and RUB savings
ESTIMATES. It NEVER places an order, never trades, never touches the network.
Real money = HARD STOP. See docs/research/tax_optimization_engine_design.md.

Layering: this package (L1/L2) imports ONLY ``finalayze.core.ndfl`` and
``finalayze.core.constants`` (both L0) plus stdlib. No upward imports.
"""

from __future__ import annotations
