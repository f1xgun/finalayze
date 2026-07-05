# Phase C — Real-Estate Sleeve: Income Diversifier or Deposit-Dominated? (Cert)

Window `2022-01-05`->`2026-06-10` · 1109 bars · RUONIA-excess 15.0%
Base = deposit 0.4 / equity 0.6; real estate (MREDC) carved from equity (sweep ['0.05', '0.10', '0.15']); rent overlay net ['3.0', '4.0', '6.0']%.

> **Two structural limits.** (1) MREDC is ~weekly/appraisal-smoothed (52 bars/yr vs equity 250) → low measured drawdown is partly an artifact the investable rental-ZPIF wrapper (illiquid, 1-3%/yr fees) would erase. (2) The ~+8.5%/yr appreciation was largely subsidised-mortgage driven — not a forward expectation.

## BINDING VERDICT: **SMOOTHED_ILLIQUID_DIVERSIFIER_DEPOSIT_DOMINATED**

Real estate (MREDC) is the STRONGEST of the three candidates — the only income-payer, and a genuine equity DIVERSIFIER (corr vs equity=0.004<0.5, vs deposit leg=0.042); over 2022-2026 residential price even BEAT equity (MCFTRR net was negative). BUT it is NOT a robust deposit-beater. Charging the investable rental-ZPIF wrapper fee (2.0%/yr, which the bare index hides), the MEASURED price-only investable form returns 19.9% vs the 100%-deposit 59.7% -> price_beats_deposit=False (deposit-dominated). A realistic 4% net rent -> 43.1% (does NOT close the gap); a generous 6% commercial-grade net rent -> 56.3% (does NOT close the gap) — and even that generous case rests on TWO fragile props: (1) MREDC is SMOOTHED (52 bars/yr ~weekly vs equity 250 ~daily) so its near-zero correlation and low drawdown are partly APPRAISAL ARTIFACTS an illiquid ZPIF would expose; (2) the ~+8.5%/yr appreciation was largely SUBSIDISED-MORTGAGE (lgotnaya ipoteka) driven, a policy now wound down -> NOT a forward expectation. Verdict: SMOOTHED_ILLIQUID_DIVERSIFIER_DEPOSIT_DOMINATED. Same family conclusion as gold/ZO: in the 16-21% rate regime the deposit anchor holds; real estate is a policy-driven, illiquid, smoothed income-diversifier, NOT a robust deposit-beater (N=1 easing cycle, one atypical sticky-price crash).

## Canonical Instrument Integration Gate (battery-comparable)
Same pre-registered gate as the beyond-edge battery (gold -> REJECT, ZO -> PROBATION). MREDC's tail IS backtestable, so real estate is held to the strict INTEGRATE bar and cannot take ZO's tail-untestable PROBATION toe-hold.

**GATE TIER: `REJECT`** (proposed weight 0, carved from equity) — no measurable net benefit over the deposit+equity core

| scorecard | value |
| --- | ---: |
| window bars / regimes | 1109 / 2 |
| tail backtestable | True |
| Δ Sharpe (10% eval) | -0.069 |
| Δ Sortino (10% eval) | -0.084 |
| Δ MaxDD pp (+ = cut) | +5.19 |
| crash-year Δ MaxDD pp (+ = raised) | -5.19 |
| toe-hold Δ Sortino (3%) | -0.023 |
| max \|corr\| to existing legs | 0.043 |

## Correlation & deposit anchor
| measure | value |
| --- | ---: |
| corr(real estate, equity) | 0.004 (1108 pairs) |
| corr(real estate, deposit leg) | 0.042 |
| 100%-deposit total return | 59.7% |
| real-estate price-only TR (after 2.0% ZPIF wrapper fee) | 19.9% |
| real-estate price + 3.0% net rent TR | 36.9% |
| real-estate price + 4.0% net rent TR | 43.1% |
| real-estate price + 6.0% net rent TR | 56.3% |
| **price-only beats deposit? (robust)** | **False** |
| base 4% rent beats deposit? | False |
| generous 6% rent beats deposit? | False |
| 0%-RE reproduces baseline curve | True |

## In-window blend (real estate carved from equity)
| window | arm | Sharpe* | Sortino* | MaxDD% | TR% | diversifies |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| full_window *(N=1)* | baseline | -0.530 | -0.658 | 30.190 | 17.117 | — |
| full_window *(N=1)* | +RE price_0.05 | -0.575 | -0.713 | 27.737 | 18.960 | no |
| full_window *(N=1)* | +RE price_0.10 | -0.628 | -0.777 | 25.235 | 20.740 | no |
| full_window *(N=1)* | +RE price_0.15 | -0.692 | -0.853 | 22.682 | 22.457 | no |
| full_window *(N=1)* | +RE total_0.05 | -0.562 | -0.697 | 27.623 | 20.024 | no |
| full_window *(N=1)* | +RE total_0.10 | -0.599 | -0.742 | 25.001 | 22.908 | no |
| full_window *(N=1)* | +RE total_0.15 | -0.643 | -0.795 | 22.324 | 25.765 | no |
| crash_year_2022 *(N=1)* | baseline | -0.812 | -0.954 | 21.114 | -10.878 | — |
| crash_year_2022 *(N=1)* | +RE price_0.05 | -0.875 | -1.024 | 19.307 | -9.886 | no |
| crash_year_2022 *(N=1)* | +RE price_0.10 | -0.947 | -1.105 | 17.488 | -8.922 | no |
| crash_year_2022 *(N=1)* | +RE price_0.15 | -1.031 | -1.200 | 15.657 | -7.985 | no |
| crash_year_2022 *(N=1)* | +RE total_0.05 | -0.867 | -1.015 | 19.201 | -9.726 | no |
| crash_year_2022 *(N=1)* | +RE total_0.10 | -0.929 | -1.084 | 17.273 | -8.600 | no |
| crash_year_2022 *(N=1)* | +RE total_0.15 | -1.002 | -1.166 | 15.330 | -7.500 | no |
| high_rate_2024_25 | baseline | -0.520 | -0.747 | 14.455 | 9.219 | — |
| high_rate_2024_25 | +RE price_0.05 | -0.577 | -0.826 | 13.124 | 9.271 | no |
| high_rate_2024_25 | +RE price_0.10 | -0.644 | -0.918 | 11.889 | 9.320 | no |
| high_rate_2024_25 | +RE price_0.15 | -0.724 | -1.027 | 10.712 | 9.367 | no |
| high_rate_2024_25 | +RE total_0.05 | -0.561 | -0.804 | 13.018 | 9.580 | no |
| high_rate_2024_25 | +RE total_0.10 | -0.609 | -0.869 | 11.777 | 9.939 | no |
| high_rate_2024_25 | +RE total_0.15 | -0.666 | -0.947 | 10.543 | 10.297 | no |
| easing_2025_26 *(N=1)* | baseline | -0.940 | -1.298 | 9.705 | 3.313 | — |
| easing_2025_26 *(N=1)* | +RE price_0.05 | -0.947 | -1.305 | 8.682 | 4.241 | no |
| easing_2025_26 *(N=1)* | +RE price_0.10 | -0.953 | -1.311 | 7.647 | 5.172 | no |
| easing_2025_26 *(N=1)* | +RE price_0.15 | -0.960 | -1.317 | 6.601 | 6.107 | no |
| easing_2025_26 *(N=1)* | +RE total_0.05 | -0.926 | -1.278 | 8.642 | 4.457 | no |
| easing_2025_26 *(N=1)* | +RE total_0.10 | -0.909 | -1.253 | 7.568 | 5.607 | no |
| easing_2025_26 *(N=1)* | +RE total_0.15 | -0.885 | -1.219 | 6.481 | 6.765 | yes |

_*RUONIA-excess on a fixed 15% basis (apt for the high-rate era); MaxDD is basis-free._
_`price_*` arms are MREDC price-only; `total_*` arms add the net rental overlay._