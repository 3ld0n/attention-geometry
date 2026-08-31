# exp-130 — Theory-of-A Level 3: Per-Sequence σ_delta Protocol

**Pre-registration.** This file was committed before the analysis script was written.
Commit hash appended after push.

**Date pre-registered:** 2026-08-31  
**Analysis type:** Forward passes (100 sequences × 512 tokens, same as exp-128)  
**Follows directly from:** exp-129 (INCONCLUSIVE — Frobenius protocol ruled out; per-sequence protocol identified as likely match to exp-117)

---

## Background

exp-117 reported σ_delta = 0.249 for the accumulated attention delta at L2H1's input,
with C_delta values: dx=8 → 0.950, dx=32 → 0.931, dx=128 → 0.656, dx=256 → 0.359.

exp-128 computed the **mean over 100 sequences** of the residual stream components,
then measured cosine similarity: σ(Δ_total) = 0.189.

exp-129 showed that Frobenius normalization gives σ ≈ 0 (wrong direction entirely)
and identified the discrepancy as likely due to per-sequence vs. mean-first computation.

The per-sequence protocol:
1. For each sequence s, compute delta_s[i] = h^(2)_s[i] - h^(0)_s[i]  (shape 768 per position)
2. Per-row normalize: delta_n_s[i] = delta_s[i] / ||delta_s[i]||
3. Compute position-cosine profile: C_s(dx) = pooled_window_profile(delta_n_s @ delta_n_s^T)
4. Average over sequences: C_avg(dx) = mean_s C_s(dx)
5. σ = -ols_slope(log C_avg, log lags)

This preserves position-specific structure within each sequence (nearby positions share
more attended context → higher cosine similarity) before averaging.

---

## Hypothesis

Computing σ_delta using the per-sequence protocol (compute cosine similarity per sequence,
then average the profile) reproduces exp-117's measurement: σ ≈ 0.249.

The mean-first protocol (exp-128, σ = 0.189) washes out per-sequence position-specificity,
giving a systematically lower estimate.

---

## Protocol

**Model:** GPT-2 small, fp32, eager attention. Same 100 WikiText-103 sequences as exp-128.

**Hooks:** Same as exp-128 — capture h^(0), attn^(0), mlp^(0), attn^(1), mlp^(1), h^(2)
per sequence. (No accumulation across sequences.)

**Per-sequence sigma measurement for Δ_total:**
- For each sequence s (shape 512×768 per component):
  - delta_s = attn0_s + mlp0_s + attn1_s + mlp1_s
  - Apply per-row normalize; compute C_s = (512×512) cosine matrix
  - C_s_profile = pooled_window_profile(C_s)  → (249,) array
- σ_delta = -ols_slope(mean_s(C_s_profile), WINDOW)

Also compute per-layer breakdown per-sequence:
- σ(attn0_s), σ(mlp0_s), σ(attn1_s), σ(mlp1_s) (each per-sequence, then mean-averaged profile)
- σ(Δ after block 0_s), σ(Δ after attn1_s), σ(Δ_total_s) — cumulative per sequence

---

## Pre-registered Predictions

**P1 (primary — reproduce exp-117):**
σ(Δ_total, per-seq) ∈ [0.22, 0.28], and the pooled profile value at dx=8 ∈ [0.90, 1.00].
*If this reproduces exp-117, the protocol difference is identified.*

**P2:** σ(Δ_total, per-seq) > σ(Δ_total, mean-first) = 0.189.
*Per-sequence protocol gives higher σ than mean-first.*

**P3:** σ(attn0, per-seq) ≈ σ(attn0, mean-first) = 0.223, within 0.05.
*The mean-first and per-seq protocols agree for attn0 since attn0 is computed
identically in both (the position-correlation structure of attn writes is intrinsic
to the model, not artifact of averaging). Wait — this might not hold; delete if wrong.*

Actually, revise P3:
**P3 (revised):** σ(Δ after block 0, per-seq) > σ(Δ after block 0, mean-first) = 0.226.
*Even the block-0 cumulative delta shows higher σ per-sequence than mean-first.*

---

## Kill Conditions

**K1:** σ(Δ_total, per-seq) < 0.20 AND pooled profile dx=8 < 0.85. The per-sequence
protocol also fails to reproduce exp-117; the measurement in exp-117 came from a
fundamentally different data source or input distribution.

---

## Verdict Criteria

| Outcome | Verdict |
|---|---|
| P1 confirmed | `confirmed` — protocol difference identified; exp-117 reproduced |
| P2 confirmed, P1 fails (σ > 0.189 but < 0.22) | `partial` |
| K1 fires | `inconclusive` |
| P2 false | `falsified` |
