# News-event study -- JUMP vs tradeable DRIFT (deterministic cert)

**Verdict: `JUMP_UNCAPTURABLE__POST_SHOCK_LONG_DRIFT_NOT_SYSTEMATICALLY_CAPTURABLE`**
(jump: `JUMP_UNCAPTURABLE` · drift: `POST_SHOCK_LONG_DRIFT_NOT_SYSTEMATICALLY_CAPTURABLE`)

Costs: 0.0055/side round-trip, NDFL 0.13 on gains. Abnormal = asset minus **price-return** IMOEX (beta=1; total-return would shave ~0.15%/5d off long drift -- conservative for the long test). Realistic entry = next-session open for intraday news, the gap open for overnight/weekend. `missed_favourable_jump` (an absolute abnormal %) is the STABLE lead metric; `jump_share` is shown ONLY when it is a meaningful in-`[0,1]` fraction (near-monotone move) and `n/a` when the move overshot and reversed.

## 1. The JUMP -- was the move gone before a retail reader could act?

### GAZP 2021 dividend cancelled (AGM)  (2022-06-30, naive dir -1, SHORT-only (retail cannot short))

_Shareholders voted down the record 2021 dividend intraday; single-name shock._

- predicted direction correct: **1/1** tickers
- median favourable jump MISSED before entry: **+25.05%**
- median NET tradeable drift @H5: **+1.27%**

| ticker | day-of raw | day-of abn | pred ok | conf | missed jump | jump_share (real/aggr) | net drift H1/H3/H5/H10 | beats deposit H5/H10 |
| --- | ---: | ---: | :-: | :-: | ---: | ---: | ---: | :-: |
| GAZP | -30.46% | -23.15% | Y | N | +25.05% | 0.92/0.85 | +2.81%/+1.54%/+1.27%/+0.61% | n/a |

### Gasoline/diesel export ban  (2023-09-21, naive dir -1, SHORT-only (retail cannot short))

_Government banned fuel exports (operator's fuel example); DIRECTION AMBIGUOUS._

- predicted direction correct: **5/6** tickers
- median favourable jump MISSED before entry: **+0.53%**
- median NET tradeable drift @H5: **-2.28%**

| ticker | day-of raw | day-of abn | pred ok | conf | missed jump | jump_share (real/aggr) | net drift H1/H3/H5/H10 | beats deposit H5/H10 |
| --- | ---: | ---: | :-: | :-: | ---: | ---: | ---: | :-: |
| ROSN | -4.03% | -2.44% | Y | N | +1.76% | n/a/n/a | -0.57%/-1.24%/-2.85%/-1.88% | n/a |
| LKOH | -0.69% | +0.91% | N | N | -1.49% | 0.69/0.42 | -1.32%/+0.40%/-1.71%/-2.11% | n/a |
| SIBN | -3.32% | -1.72% | Y | N | +1.12% | n/a/n/a | -1.51%/-1.86%/-3.95%/-6.62% | n/a |
| TATN | -2.32% | -0.73% | Y | Y | +0.52% | n/a/n/a | -2.10%/-0.78%/-1.58%/-3.19% | n/a |
| SNGS | -2.55% | -0.96% | Y | Y | +0.55% | n/a/n/a | -0.42%/-0.58%/-3.22%/-15.29% | n/a |
| GAZP | -1.93% | -0.33% | Y | Y | -0.13% | n/a/0.33 | -0.15%/+0.64%/+0.06%/+2.56% | n/a |

### VK apps pulled from Apple App Store  (2022-09-26, naive dir -1, SHORT-only (retail cannot short))

_Operator's VK example; CONFOUNDED by the 21 Sep mobilisation crash._

- predicted direction correct: **1/1** tickers
- median favourable jump MISSED before entry: **+12.58%**
- median NET tradeable drift @H5: **-12.28%**

| ticker | day-of raw | day-of abn | pred ok | conf | missed jump | jump_share (real/aggr) | net drift H1/H3/H5/H10 | beats deposit H5/H10 |
| --- | ---: | ---: | :-: | :-: | ---: | ---: | ---: | :-: |
| VKCO | -21.34% | -13.85% | Y | N | +12.58% | n/a/n/a | -11.85%/-6.11%/-12.28%/-15.79% | n/a |

### SBER record dividend recommended  (2023-03-17, naive dir +1, LONG-accessible)

_Supervisory board recommended a record dividend; positive single-name shock._

- predicted direction correct: **1/1** tickers
- median favourable jump MISSED before entry: **+8.20%**
- median NET tradeable drift @H5: **+0.38%**

| ticker | day-of raw | day-of abn | pred ok | conf | missed jump | jump_share (real/aggr) | net drift H1/H3/H5/H10 | beats deposit H5/H10 |
| --- | ---: | ---: | :-: | :-: | ---: | ---: | ---: | :-: |
| SBER | +10.34% | +7.48% | Y | N | +8.20% | 0.81/0.74 | +0.25%/-0.00%/+0.38%/+3.93% | Y/Y |

### Wagner mutiny weekend  (2023-06-26, naive dir -1, SHORT-only (retail cannot short))

_Broke Fri night / Sat 24 Jun; market first trades Mon 26 Jun -> pure open gap._

- predicted direction correct: **1/1** tickers
- median favourable jump MISSED before entry: **+0.84%**
- median NET tradeable drift @H5: **-1.90%**

| ticker | day-of raw | day-of abn | pred ok | conf | missed jump | jump_share (real/aggr) | net drift H1/H3/H5/H10 | beats deposit H5/H10 |
| --- | ---: | ---: | :-: | :-: | ---: | ---: | ---: | :-: |
| IMOEX | -1.36% | -1.36% | Y | N | +0.84% | n/a/n/a | -1.28%/-1.99%/-1.90%/-4.14% | n/a |

## 2. The DRIFT -- direction-blind base rate (the real tradeable question)

Only 1 of the 5 named events (SBER) is LONG-accessible, so the named set cannot settle retail capturability. This scans EVERY large abnormal daily move across all 8 names (a news-shock proxy, not hand-picked) and reports the median NET LONG drift of chasing the pop / buying the dip, vs the deposit carry over the same window (~0.2-0.3% net per 10 trading days).

| shock >= | horizon | chase pop (n, median net) | buy dip (n, median net) |
| --- | :-: | ---: | ---: |
| |3%| | H5 | n=115, -0.83% | n=64, -1.68% |
| |3%| | H10 | n=114, -1.29% | n=64, -1.13% |
| |5%| | H5 | n=32, -0.84% | n=20, -3.68% |
| |5%| | H10 | n=32, -1.82% | n=20, -2.24% |

## Honest limits

- **N and selection.** The 5 named events are hand-picked ex-post-large shocks; that biases the JUMP finding toward 'uncapturable'. The DRIFT finding leans on the direction-blind base rate instead (dozens of shocks), which is the fairer test.
- **Daily data.** With no intraday bars the whole reaction-day move is charged to the un-capturable jump, so measured jump_share is an UPPER bound -- conservative.
- **Short-only exclusion.** 4/5 named events are bad news; retail on MOEX cannot short single names, so those carry no capturability information (excluded from the drift verdict; shown only for the jump).
- **One long event.** SBER's post-jump long drift is net-positive at H1/H5/H10 and beats the deposit at H10 -- but it is N=1 and could be the name's 2023 uptrend; the base rate is what tells us whether that generalises.

