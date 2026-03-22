"""Backward-compatibility shim -- telegram_bot moved to finalayze.api.telegram_bot.

Makes ``finalayze.core.telegram_bot`` an alias for the canonical module so that
both ``from finalayze.core.telegram_bot import X`` and
``patch("finalayze.core.telegram_bot.X")`` continue to work transparently.
"""

from __future__ import annotations

import sys

import finalayze.api.telegram_bot as _canonical

sys.modules[__name__] = _canonical
