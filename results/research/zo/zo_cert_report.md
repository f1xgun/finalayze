# Phase B — ZO (Replacement-Bond) Sleeve: FX-Linked Diversifier? (Cert)

Window `2023-01-03`->`2026-06-10` · 874 bars · RUONIA-excess 15.0%
Base = deposit 0.4 / equity 0.6; ZO carved from equity (sweep ['0.05', '0.10', '0.15']). ZO = RURPLRUBTR net-NDFL.

> **Structural limit:** ZO + RUCNYTR start 2023 — they POSTDATE the 2022 crash they would hedge (created *by* the freeze). The acute-2022 tail benefit is **un-backtestable**; this cert measures in-window FX-linkage + diversification only, and reports the tail as a forward-structural argument, never measured.

## BINDING VERDICT: **FX_LINKED_DIVERSIFIER_TAIL_UNTESTED** (tail un-backtestable)

ZO (RURPLRUBTR) is FX-LINKED: daily-return beta vs CNYRUB (durable daily proxy)=0.330, vs USDRUB=0.401 (pre-Jun-2024-sanction window only; the exchange USD series then halts). It is a genuine, non-redundant diversifier — corr vs equity=0.052 (<0.5) and corr vs the deposit leg=-0.059 (not >=0.9). The CNY-bond index RUCNYTR is by contrast only WEAKLY FX-linked (CNY beta 0.063) — RURPL is the FX-linked one. Verdict: FX_LINKED_DIVERSIFIER_TAIL_UNTESTED. CRITICAL: THE 2022 TAIL BENEFIT IS UN-BACKTESTABLE — ZO postdates the crash (index starts 2023), so the hedge is a forward-structural argument, NOT measured. In-window (2023-2026, NO crash) ZO is INSURANCE WITH A COST: it modestly cuts MaxDD (~1-2pp) but LOWERS total return (full 49%->38% at 15%) and worsens Sortino — there was no crash in-sample to reward it. Not alpha; a forward-looking FX-tail insurance leg whose payoff is structurally sound but unproven (N=1, no in-window crash).

## FX-linkage & diversification (raw daily-return beta / correlation)
| pair | beta | corr | n |
| --- | ---: | ---: | ---: |
| zo_vs_usdrub_presanction | 0.401 | 0.406 | 365 |
| zo_vs_cnyrub | 0.330 | 0.384 | 873 |
| cnybond_vs_cnyrub | 0.063 | 0.227 | 873 |
| zo_vs_equity | 0.035 | 0.052 | 873 |
| zo_vs_deposit_leg | — | -0.059 | — |

- 0%-ZO reproduces baseline curve: **True**

## In-window diversification blend (no crash in 2023-2026)
| window | arm | Sharpe* | Sortino* | MaxDD% | TR% | diversifies |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| full_2023_26 *(N=1)* | baseline | -0.167 | -0.239 | 14.455 | 48.551 | — |
| full_2023_26 *(N=1)* | +ZO 0.05 | -0.258 | -0.368 | 13.561 | 44.950 | no |
| full_2023_26 *(N=1)* | +ZO 0.10 | -0.365 | -0.515 | 12.911 | 41.406 | no |
| full_2023_26 *(N=1)* | +ZO 0.15 | -0.489 | -0.682 | 12.261 | 37.920 | no |
| high_rate_2024_25 | baseline | -0.520 | -0.747 | 14.455 | 9.219 | — |
| high_rate_2024_25 | +ZO 0.05 | -0.633 | -0.902 | 13.561 | 8.170 | no |
| high_rate_2024_25 | +ZO 0.10 | -0.765 | -1.079 | 12.911 | 7.118 | no |
| high_rate_2024_25 | +ZO 0.15 | -0.919 | -1.281 | 12.261 | 6.061 | no |
| easing_2025_26 *(N=1)* | baseline | -0.940 | -1.298 | 9.705 | 3.313 | — |
| easing_2025_26 *(N=1)* | +ZO 0.05 | -1.061 | -1.456 | 9.071 | 2.951 | no |
| easing_2025_26 *(N=1)* | +ZO 0.10 | -1.200 | -1.633 | 8.432 | 2.587 | no |
| easing_2025_26 *(N=1)* | +ZO 0.15 | -1.357 | -1.830 | 7.789 | 2.221 | no |

_*RUONIA-excess on a fixed 15% basis (apt for the high-rate era)._