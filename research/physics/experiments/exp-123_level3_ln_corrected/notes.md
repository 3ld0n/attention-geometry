# exp-123 — Theory-of-A Level 3: Layer-Norm-Corrected Quantitative Closure

**Date:** 2026-08-20  
**Type:** Analysis-only (existing GPT-2 weights; no forward passes)  
**Registered:** 2026-08-20, before any analysis was run  
**Follows:** exp-122 (pos_emb propagation route confirmed, but d_model σ_out = 0.214 < σ_delta = 0.249)

---

## Background

exp-122 confirmed that GPT-2's learned positional embeddings, projected through W_V and
convolved with the causal conformal kernel (Δ=0.249), produce a power-law position-correlation
profile with σ_out ≈ Δ in d_head space (L2H1: 0.282). However, in d_model space the result
was σ_out = 0.214 for L2H1 — below σ_delta = 0.249 by ~14%.

exp-122's interpretation: "The discrepancy likely comes from layer-norm not being applied to
bare pos_emb (in the actual model, LN normalizes h̄^(0) = emb_mean + pos_emb, not pos_emb
alone). Layer-norm-corrected version is the next step."

This experiment runs the corrected protocol: instead of projecting bare pos_emb through W_V,
we first compute h̄^(0)[i] = emb_mean + pos_emb[i] (where emb_mean = mean of all token
embeddings), apply the layer norm from the target head's transformer block (layer.ln_1), and
then project through W_V and convolve with the conformal kernel.

---

## Pre-registration

**Hypothesis (stated before any computation):**

Applying layer norm to h̄^(0)[i] = emb_mean + wpe[i] before W_V projection — matching the
actual model's computation — closes the quantitative gap between exp-122's d_model result
(σ_out = 0.214 at L2H1) and the observed σ_delta = 0.249 from exp-117.

The layer norm subtracts the per-vector mean and normalizes variance; this changes the
position-specific variation structure that enters W_V. Since exp-122 used bare pos_emb while
the real model operates on LN(emb_mean + pos_emb), this correction should bring the analytic
prediction into better alignment with the measured σ_delta.

**Protocol:**

1. Load GPT-2 small weights (cpu, no forward pass).
2. Extract wpe (positional embeddings), shape (1024, 768); take positions 0..511.
3. Extract wte (token embeddings), shape (50257, 768); compute emb_mean = wte.mean(axis=0).
4. For each structural head (L2H1, L3H4, L5H0, L7H11, L10H8), using the block index `ℓ`:
   a. Compute h̄^(0)[i] = emb_mean + wpe[i] for i=0..511. Shape (512, 768).
   b. Apply ln_1 from transformer block ℓ:
      - weight γ (768,), bias β (768,) from model.transformer.h[ℓ].ln_1
      - per-vector: μ_i = mean(h̄^(0)[i]), σ_i = std(h̄^(0)[i])
      - h̄_LN[i] = γ ⊙ (h̄^(0)[i] − μ_i) / (σ_i + ε) + β
   c. Extract W_V_h from block ℓ: c_attn.weight[:, 2*768:3*768][:, h*64:(h+1)*64], shape (768, 64).
      Add value bias component.
   d. Compute v[j] = h̄_LN[j] @ W_V_h, shape (512, 64).
   e. Compute out_dh[i] = Σ_j A[i,j] × v[j] using the same causal conformal A as exp-122.
   f. Normalize row-wise; compute correlation matrix; apply pooled_window_profile (lags 8..256,
      deep_lo=256); fit log-log slope σ_out and R².
   g. Also compute d_model version via W_O projection (same as exp-122).
5. Compare to exp-122 baseline (bare pos_emb) and to σ_delta = 0.249.

**Pre-registered predictions:**

- **P1 (directional improvement):** σ_out (d_model) for L2H1 > 0.22 — the LN correction moves
  the d_model result toward σ_delta = 0.249, beyond exp-122's 0.214. The improvement is
  directional: LN should tighten the pos_emb variation structure and bring the analytic
  result into better agreement with the census measurement.

- **P2 (quantitative match):** σ_out (d_model) for L2H1 ∈ [0.22, 0.28] — the LN-corrected
  result falls in the window within 12% of σ_delta = 0.249. This is the quantitative closure
  claim: the analytic pos_emb propagation route fully accounts for σ_delta when the actual
  model computation (LN before projection) is respected.

- **P3 kill:** σ_out (d_model) for L2H1 ≤ 0.20 — if the LN correction fails to improve over
  exp-122's 0.214 (or worsens it below 0.20), the pos_emb propagation route cannot
  quantitatively account for σ_delta = 0.249 through first-layer LN + W_V alone.
  Multi-layer compositional effects would remain as the explanation for the gap.

---

## Results

**Run:** 2026-08-20. Pre-registration commit `c4cac26` preceded this run.

| Head | σ_out (d_head) | R² | σ_out (d_model) | exp-122 baseline | Improved? |
|---|---|---|---|---|---|
| L2H1 | 0.114 | 0.819 | **0.116** | 0.214 | NO |
| L3H4 | 0.088 | 0.828 | 0.084 | 0.175 | NO |
| L5H0 | 0.125 | 0.816 | 0.123 | 0.203 | NO |
| L7H11 | 0.153 | 0.818 | 0.160 | 0.241 | NO |
| L10H8 | 0.112 | 0.851 | 0.108 | 0.180 | NO |

Reference: bare pos_emb (no LN, no emb_mean): σ = 6.109, R² = 0.861 — same as exp-122.

**Registered verdicts:**
- P1 (d_model L2H1 > 0.22): **FAIL** — σ = 0.116
- P2 (d_model L2H1 ∈ [0.22, 0.28]): **FAIL**
- P3 kill (d_model L2H1 ≤ 0.20): **FIRED** — σ = 0.116 << 0.20
- Heads improved over exp-122: **0/5**

**Overall verdict: P3 KILL / FALSIFIED.** The LN correction makes every head's σ_out worse, not
better. The hypothesis that the exp-122 quantitative gap (0.214 vs σ_delta = 0.249) was due
to missing layer norm is wrong.

---

## Interpretation

This is an honest negative that clarifies the theory-of-A chain.

The emb_mean norm is 2.05; pos_emb norm at position 0 is 9.88. So h̄^(0) = emb_mean + pos_emb
is dominated by pos_emb in raw magnitude (~10× difference). After applying LN:

1. **The per-vector mean subtraction** removes the "DC" component of each position vector —
   in the process, it partially removes the shared emb_mean contribution and leaves the
   position-specific deviation.

2. **The per-vector normalization** scales each position to the same variance. This homogenizes
   the position vectors — after LN, each vector has similar norm (scaled by γ), regardless of
   how different the original pos_emb values were.

3. **The effect on σ_out:** the raw pos_emb carries large oscillatory correlations (σ_raw =
   6.109) that the conformal convolution in exp-122 transformed into σ ≈ 0.18–0.28. After LN,
   the position-correlation structure of h̄_LN[i] is *weaker* than raw pos_emb — the LN
   normalization homogenizes the vectors and reduces the cross-position correlations that drive
   σ_out. The conformal mechanism still operates (R² ≈ 0.81–0.85), but at lower amplitude
   (σ_out ≈ 0.09–0.16 vs 0.18–0.28 without LN).

**What the falsification means for Level-3:**

exp-122 established the mechanism: the conformal kernel transmits its exponent to the
pos_emb-driven output. But the quantitative gap (d_model L2H1: 0.214 vs σ_delta = 0.249)
is NOT explained by adding LN to the first-layer input. LN makes the gap larger.

From exp-117: by the time we reach structural head L2H1, the *accumulated delta* dominates
h̄^(ℓ) by 13–32× in norm. The actual input to L2's W_V is the residual stream, which is
dominated by accumulated attention outputs from layers 0 and 1 — not by h̄^(0) at all.

**Correct explanation path for σ_delta ≈ 0.249:** the quantitative account requires a
multi-layer analysis — the residual stream at layer ℓ = 2 already carries the power-law
position structure from the first two layers of conformal processing. The Level-3
quantitative closure is genuinely a multi-layer question. The LN-correction route to a
single-layer analytic closure is falsified.

**Next step:** The multi-layer route requires either (a) an analytic derivation of how σ
accumulates across layers of conformal attention applied to the evolving residual stream, or
(b) a numerical experiment that isolates each layer's contribution to σ_delta. This is a
harder question than the single-layer analysis, and the correct next step is to understand
whether σ_delta ≈ 0.249 is set in the first two layers, or whether it requires the full depth.
This is **not** registered — register before computing.

---

## Artifacts

- `notes.md` — pre-registration + results (this file)
- `run.py` — analysis script (written after pre-registration commit c4cac26)
- `results.json` — numerical results
