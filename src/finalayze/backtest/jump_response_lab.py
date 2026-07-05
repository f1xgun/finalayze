"""Jump-response primitives — measure the alpha-decay window of a *reactive* news trade.

The news event study (``event_study_lab``) tested the SLOW regime: entry at the next daily open,
where 81-92% of a single-name shock is already priced. It could not resolve whether the residual is
reachable by a *reactive* trader who acts within minutes. This lab answers that at minute resolution
without needing to source per-headline timestamps.

The design is a **conditional forward path after a large instantaneous move**. A reactive news bot
enters an *already-started* move — so the honest question is not "can you predict the jump" but
"once a large 1-minute move fires, how much move is left at t0+1, +5, +15 minutes, net of cost?".
So:

  1. :func:`detect_jumps` flags bars whose 1-minute return exceeds ``z_threshold`` times the
     *trailing* realised vol (the vol window strictly precedes the bar — no look-ahead; a shock
     never inflates its own denominator).
  2. :func:`forward_path` measures the cumulative return AFTER the jump bar's close (the jump itself
     is already gone — you only learn of it once the bar closes, so ``path[0] == 0`` is the earliest
     realistic entry reference).
  3. :func:`mean_path` / :func:`capture` / :func:`half_life_bars` turn the averaged forward path
     into the money numbers: how fast the continuation decays, and what a reactor entering at
     latency L and exiting at horizon H captures — :func:`net_after_cost` nets round-trip + NDFL.

Pure, no I/O, Decimal-native for deterministic certs. Measurement only — authorises no order.
"""

from __future__ import annotations

from decimal import Decimal

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)


def bar_returns(levels: list[Decimal]) -> list[Decimal]:
    """Simple 1-bar returns aligned to ``levels``; ``returns[0] == 0`` (bar 0 has no prior)."""
    out = [_ZERO]
    for i in range(1, len(levels)):
        prev = levels[i - 1]
        out.append((levels[i] - prev) / prev if prev > _ZERO else _ZERO)
    return out


def trailing_stdev(returns: list[Decimal], i: int, window: int) -> Decimal | None:
    """Sample stdev of the ``window`` real returns just before bar ``i`` (``returns[i-window:i]``).

    Returns ``None`` until a full window of real returns exists — real returns start at index 1
    (index 0 is the placeholder), so a full window needs ``i - window >= 1``. Crucially this NEVER
    includes ``returns[i]`` itself: the bar being tested cannot inflate its own volatility.
    """
    if i - window < 1:
        return None
    sample = returns[i - window : i]
    mean = sum(sample, _ZERO) / len(sample)
    var = sum(((x - mean) ** 2 for x in sample), _ZERO) / (len(sample) - 1)
    return var.sqrt()


def detect_jumps(
    levels: list[Decimal], *, vol_window: int, z_threshold: Decimal
) -> list[tuple[int, int]]:
    """Bars whose 1-minute return clears ``z_threshold`` times the trailing vol.

    Returns ``[(index, sign)]`` with ``sign`` +1 (up-jump) / -1 (down-jump). The vol denominator is
    :func:`trailing_stdev` (strictly prior window, look-ahead-free). Bars without a full vol
    window, or with zero trailing vol, are skipped.
    """
    returns = bar_returns(levels)
    out: list[tuple[int, int]] = []
    for i in range(1, len(levels)):
        sd = trailing_stdev(returns, i, vol_window)
        if sd is None or sd <= _ZERO:
            continue
        if abs(returns[i]) >= z_threshold * sd:
            out.append((i, 1 if returns[i] > _ZERO else -1))
    return out


def forward_path(levels: list[Decimal], event_idx: int, horizon: int) -> list[Decimal]:
    """Cumulative return after the jump bar's close: ``path[h] = levels[event_idx+h]/base - 1``.

    ``path[0] == 0`` is the entry reference (the jump bar's close — the earliest a reactor can act).
    Length ``horizon + 1``. Raises if the window runs past the data.
    """
    if event_idx + horizon >= len(levels):
        raise ValueError("forward window runs past the data")
    base = levels[event_idx]
    if base <= _ZERO:
        raise ValueError("non-positive base level")
    return [(levels[event_idx + h] - base) / base for h in range(horizon + 1)]


def align_sign(path: list[Decimal], sign: int) -> list[Decimal]:
    """Flip a down-jump path so continuation reads positive for both directions."""
    s = Decimal(sign)
    return [s * x for x in path]


def mean_path(paths: list[list[Decimal]]) -> list[Decimal]:
    """Element-wise mean across equal-length forward paths."""
    if not paths:
        raise ValueError("no paths")
    n = len(paths)
    length = len(paths[0])
    return [sum((p[h] for p in paths), _ZERO) / n for h in range(length)]


def capture(mean_forward: list[Decimal], entry_h: int, exit_h: int) -> Decimal:
    """Gross return a reactor captures entering at bar ``entry_h`` and exiting at ``exit_h``."""
    return mean_forward[exit_h] - mean_forward[entry_h]


def net_after_cost(gross: Decimal, round_trip_cost: Decimal, ndfl: Decimal) -> Decimal:
    """Net of round-trip cost then NDFL on the positive net (no loss offset — conservative)."""
    net_pretax = gross - round_trip_cost
    return net_pretax * (_ONE - ndfl) if net_pretax > _ZERO else net_pretax


def half_life_bars(mean_forward: list[Decimal]) -> int | None:
    """Bars until half the *terminal* continuation is realised; ``None`` if no continuation.

    Interpretable when the (sign-aligned) mean path is near-monotone in the jump direction. If the
    terminal value is <= 0 the move reverses/dies — there is no continuation half-life to report.
    """
    terminal = mean_forward[-1]
    if terminal <= _ZERO:
        return None
    target = terminal / _TWO
    for h, v in enumerate(mean_forward):
        if v >= target:
            return h
    return None
