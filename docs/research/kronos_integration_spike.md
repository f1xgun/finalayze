# Kronos Foundation Model — Integration Spike

**Status:** Research spike (no production wiring)
**Date:** 2026-05-30
**Owner:** ml-agent
**Spike script:** [`scripts/spikes/kronos_moex_zeroshot_eval.py`](../../scripts/spikes/kronos_moex_zeroshot_eval.py)
**Upstream:** <https://github.com/shiyu-coder/Kronos> · paper [arXiv:2508.02739](https://arxiv.org/pdf/2508.02739) (AAAI 2026)

---

## 1. What Kronos is

Kronos is the first open-source **foundation model for financial candlesticks (K-lines)**.
Two-stage design:

1. **Tokenizer** — quantizes continuous multi-dimensional OHLCV into hierarchical
   discrete tokens.
2. **Decoder-only autoregressive Transformer** — pre-trained on those tokens (12B+
   K-lines from 45+ exchanges), generates future candles autoregressively.

| Variant | Params | Context | Weights |
|---|---|---|---|
| Kronos-mini | 4.1M | 2048 | open (MIT) |
| Kronos-small | 24.7M | 512 | open (MIT) |
| Kronos-base | 102.3M | 512 | open (MIT) |
| Kronos-large | 499.2M | 512 | **closed** |

- **Task type:** *generative* — predicts probabilistic future OHLCV paths
  (`temperature`, `top_p`, `sample_count`), not a buy/sell label.
- **License:** MIT (code + mini/small/base weights) → commercial use allowed.
- **Inference:** ~50 ms for base **on an A100**; PyTorch; CPU is materially slower.

## 2. Why it does NOT replace our ML stack

Our `ml_ensemble` is **discriminative** (`predict_proba → P(BUY)` from 45 hand-crafted
features, reinforcer-only). Kronos is **generative** price forecasting on raw OHLCV.
They live in the same *price* dimension; Kronos is complementary, not a substitute.

Crucially, Kronos sees **only candles** — it does **not** address the system's
fundamental/news gaps (earnings, forward EPS, analyst targets, guidance). Those are a
separate data epic. Kronos is purely a **Group A (price/time-series)** play.

## 3. Integration thesis (only if the spike passes)

Promote Kronos to a forward-looking **feature source feeding `ml_ensemble`**, never a
standalone signal — consistent with the existing reinforcer-only contract:

- `kronos_expected_return_h` — predicted return over horizon h.
- `kronos_pred_vol_h` — dispersion across sampled paths (forward volatility).
- `kronos_p_up` — fraction of sampled paths closing up.

These 2–3 numbers join the 45 existing features and pass the standard
`feature_selection` + `quality_gates` pipeline. Secondary uses: forward-vol for
ATR-stop / Half-Kelly sizing; sampled paths as conditioned Monte-Carlo scenarios.

## 4. The one question this spike answers first

> **Does zero-shot Kronos (no fine-tuning) beat naive baselines on MOEX daily
> candles?**

This gates everything because:

- If **yes** → cheap path: derive features, validate one MOEX segment through
  `backtest-iteration`.
- If **no** → Kronos needs **MOEX fine-tuning**, which requires multi-GPU
  (`torchrun`, 2+ GPUs) infrastructure we do not have. That conflicts with the
  current MOEX-first-without-GPU posture → **stop / defer**.

It is unknown whether MOEX was among the 45 training exchanges, so zero-shot quality
on SBER/GAZP is genuinely uncertain and must be measured, not assumed.

## 5. Spike method (`kronos_moex_zeroshot_eval.py`)

- Fetch daily MOEX candles via **T-Bank gRPC only** (never yfinance), direct
  `AsyncClient` + FIGI resolution (same pattern as `fetch_moex_dividends.py`).
- Walk-forward over a held-out window: from `lookback` (≤512) context candles,
  `predict()` the next `horizon` candles; compare predicted vs actual **close
  direction**.
- Report **directional accuracy (1-bar and h-bar)** and **predicted-vs-actual return
  correlation**, against three baselines: majority-class (up-rate), persistence, and
  the implicit 50% coin-flip.
- **Decision gate:** promote only if Kronos beats the best baseline by a *stable*
  margin (>~3pp) across several symbols/seeds. A single-symbol coin-flip ⇒ zero-shot
  insufficient.

Run (Kronos + torch in a throwaway venv, **not** added to `pyproject.toml`):

```bash
pip install torch huggingface_hub einops safetensors
git clone https://github.com/shiyu-coder/Kronos /tmp/Kronos
export KRONOS_REPO=/tmp/Kronos
uv run python scripts/spikes/kronos_moex_zeroshot_eval.py --symbol SBER --device cpu
```

## 6. Risks & blockers

| Risk | Note |
|---|---|
| GPU latency in live loop | base ~50 ms on A100; CPU much slower for a multi-symbol cycle. Mini/small lighter but weaker. |
| MOEX coverage unknown | likely needs fine-tuning → multi-GPU we lack. |
| Black box vs our gates | Brier/calibration/purged-CV gates are for classification; Kronos validated only indirectly via derived features + backtest. |
| Priority | our own `ml_ensemble` is currently **disabled** (force-saved, gate-failed) on all segments; adding a generative model before the base ensemble passes gates is a research detour. |
| Dependency weight | large weights + torch; keep as optional/offline, never a hard dep. |

## 6a. Results — RUN 2026-05-30 (zero-shot, daily, MPS)

Ran the spike on 4 MOEX names (SBER, GAZP, LKOH, GMKN), walk-forward, `sample_count=1`,
`lookback=400`, `horizon=5`, against the best of {majority-class, persistence} baselines.

| Model | pooled n (1-bar) | Kronos dir-acc | best baseline | edge | z vs 50% | 95% CI |
|---|---|---|---|---|---|---|
| Kronos-small (24.7M) | 280 | 48.6% | 54.6% | **−6.1%** | −0.48 | [42.7%, 54.4%] |
| Kronos-base (102.3M) | 200 | 51.0% | 56.5% | **−5.5%** | +0.28 | [44.1%, 57.9%] |

- Both models are **statistically indistinguishable from a coin flip** (CI straddles 50%)
  and sit **below** the naive baselines on 1-bar direction.
- The promising single-symbol SBER read (60% on n=30) was small-sample noise — it
  reversed to 44% at n=70.
- Return-magnitude correlation was weakly positive (0.02–0.23), i.e. a faint signal that
  is not strong enough for tradable direction.

**Gate verdict: FAIL.** Zero-shot Kronos has no usable directional edge on MOEX daily
candles. An edge, if any, would require **MOEX fine-tuning** (multi-GPU) — out of scope.

Caveats (kept honest): only the daily timeframe and `sample_count=1` were tested; the
closed Kronos-large (499M) and intraday timeframes were not. The result is consistent
across two model sizes and four symbols, so it is robust *for the daily-MOEX-zero-shot
question* — which is exactly the gate.

## 7. Recommendation (post-run)

**SHELVED.** The zero-shot gate failed (§6a) — do **not** open a Kronos feature phase now.

1. ~~Run the zero-shot spike on 3–5 MOEX names.~~ Done 2026-05-30 → no edge.
2. Revisit **only if** one of these changes: (a) we acquire multi-GPU infra to fine-tune
   Kronos on MOEX; (b) we want to test intraday timeframes or the closed Kronos-large;
   (c) a future Kronos release claims MOEX/equity-daily coverage.
3. The spike harness (`scripts/spikes/kronos_moex_zeroshot_eval.py`) is kept as the
   re-runnable gate for any such revisit.
4. The fundamental/news gaps (Group B) remain the higher-value next step — a separate
   data epic that Kronos never addressed.
