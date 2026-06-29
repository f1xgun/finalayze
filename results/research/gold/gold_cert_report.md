# Phase A — Gold Sleeve vs Deposit-Anchored Allocation (Cert)

Window `2022-01-03`->`2026-06-10` · 1126 bars · 19 rebalances · RUONIA-excess 15.0%
Base = deposit 0.4 / equity 0.6; gold carved from equity (sweep ['0.05', '0.10', '0.15']). Gold = GLDRUB spot, net-NDFL + TER haircut.

> **Metric caveat:** the RUONIA-excess Sharpe/Sortino use a fixed 15% basis apt only for the high-rate era. Over a 2022-start window the deposit underperforms that basis in the 2022-2023 low-rate era, so a 'vs deposit' Sharpe test is meaningless here — the deposit-anchor point is made on **basis-free total return**, the gold verdict on the **baseline-vs-+gold MaxDD/Sortino delta** (common basis cancels).

## Honesty controls
- 0%-gold reproduces baseline curve: **True**
- gold RAW price round-trip — acute (Feb21-Apr30): **-12.8%**, crash YEAR: **-17.1%** (the spike then capital-controlled ruble recovery — the market move)
- gold NET leg (after daily-mark NDFL + TER) — acute: **-20.9%**, crash YEAR: **-35.1%** (extra drag = the conservative daily-mark NDFL)
- gold vs equity daily-return corr (crash year): **-0.127** (low/negative ⇒ genuinely uncorrelated)

## BINDING: deposit anchor holds (raw return) = **True** · gold crash effect = **NO** (N=1)

Deposit anchor holds on raw total return (high-rate + full): True. Gold crash de-risking (pre-registered >=3.0pp MaxDD cut in a crash window): NO — the best acute-crash cut is ~3pp (just under the bar) and gold INCREASES MaxDD in the crash YEAR (the give-back). Gold DOES shave MaxDD modestly in the calm/acute regimes (full, acute, high-rate, easing all lower in the table) but ALWAYS worsens risk-adjusted return (Sortino) — the zero-yield drag. Its 2022 RUB hedge was a ~2-week flash: raw price round-tripped -17% over the crash year (spike then capital-controlled ruble recovery); the net leg ended -35% after the conservative daily-mark NDFL + TER haircut. Net: gold is at most a MARGINAL drawdown-reducer at a risk-adjusted cost — not a diversifier, not a reliable crash hedge, not alpha (N=1).

| window | arm | Sharpe* | Sortino* | MaxDD% | TR% | diversifies |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| full_window | deposit | -12.760 | -10.050 | 0.000 | 60.927 | — |
| full_window | baseline | -0.541 | -0.672 | 30.313 | 16.973 | — |
| full_window | +gold 0.05 | -0.604 | -0.750 | 29.602 | 17.837 | no |
| full_window | +gold 0.10 | -0.667 | -0.831 | 28.925 | 18.602 | no |
| full_window | +gold 0.15 | -0.725 | -0.915 | 28.283 | 19.264 | no |
| acute_crash_2022 *(N=1)* | deposit | -3.546 | -3.574 | 0.000 | 2.462 | — |
| acute_crash_2022 *(N=1)* | baseline | -1.148 | -1.333 | 18.198 | -10.400 | — |
| acute_crash_2022 *(N=1)* | +gold 0.05 | -1.386 | -1.597 | 16.003 | -10.488 | no |
| acute_crash_2022 *(N=1)* | +gold 0.10 | -1.631 | -1.886 | 15.248 | -10.586 | no |
| acute_crash_2022 *(N=1)* | +gold 0.15 | -1.858 | -2.199 | 15.444 | -10.691 | no |
| crash_year_2022 *(N=1)* | deposit | -25.989 | -13.579 | 0.000 | 7.110 | — |
| crash_year_2022 *(N=1)* | baseline | -0.804 | -0.944 | 20.788 | -10.513 | — |
| crash_year_2022 *(N=1)* | +gold 0.05 | -0.951 | -1.111 | 21.297 | -10.931 | no |
| crash_year_2022 *(N=1)* | +gold 0.10 | -1.103 | -1.294 | 21.826 | -11.374 | no |
| crash_year_2022 *(N=1)* | +gold 0.15 | -1.248 | -1.489 | 23.174 | -11.843 | no |
| high_rate_2024_25 | deposit | 0.941 | 1.388 | 0.000 | 22.410 | — |
| high_rate_2024_25 | baseline | -0.520 | -0.747 | 14.455 | 9.219 | — |
| high_rate_2024_25 | +gold 0.05 | -0.542 | -0.779 | 13.054 | 9.931 | no |
| high_rate_2024_25 | +gold 0.10 | -0.563 | -0.810 | 11.895 | 10.634 | no |
| high_rate_2024_25 | +gold 0.15 | -0.582 | -0.838 | 10.733 | 11.328 | no |
| easing_2025_26 *(N=1)* | deposit | -17.044 | -11.941 | 0.000 | 13.700 | — |
| easing_2025_26 *(N=1)* | baseline | -0.940 | -1.298 | 9.705 | 3.313 | — |
| easing_2025_26 *(N=1)* | +gold 0.05 | -1.028 | -1.410 | 8.210 | 3.114 | no |
| easing_2025_26 *(N=1)* | +gold 0.10 | -1.106 | -1.502 | 7.773 | 2.901 | no |
| easing_2025_26 *(N=1)* | +gold 0.15 | -1.166 | -1.562 | 8.786 | 2.673 | no |

_*Sharpe/Sortino are RUONIA-excess on a fixed 15% basis — see the metric caveat._