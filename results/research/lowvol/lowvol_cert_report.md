# Step 3 — Low-Vol Blend vs REAL IMOEX Cap-Weight (Cert)

Window `2022-01-03`->`2026-06-29` · 1225 bars · 19 rebalances · 64 names · RUONIA-excess 15.0%
Tilt: FINAL = (1-0.25)·cap_weight + 0.25·inverse_vol(lowest-vol half).

## Honesty controls
- lambda=0 reproduces cap-weight curve: **True**
- index-weight coverage: avg **94.8%**, min **84.5%**
- cap-basket vs IMOEX return corr: **0.856**
- low-vol-half vs cap-top-half overlap: **58.9%** (>80% ⇒ cosmetic)

## BINDING VERDICT: **HARD_FAIL** (N=1 caveat)

low-vol blend does NOT beat the real IMOEX cap-weight baseline (full_window+high_rate) net of cost/tax

| window | armSharpe | armSortino | armMaxDD% | capSharpe | capSortino | capMaxDD% | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| full_window | -0.634 | -0.795 | 49.789 | -0.600 | -0.757 | 49.352 | FAIL |
| high_rate | -0.442 | -0.549 | 49.789 | -0.405 | -0.506 | 49.352 | FAIL |
| early_cut *(N=1)* | -1.463 | -1.989 | 24.299 | -1.405 | -1.926 | 25.348 | FAIL |