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

*To be filled after the run.*

---

## Artifacts

- `notes.md` — pre-registration (this file) + results
- `run.py` — analysis script (written after this pre-registration was committed)
- `results.json` — numerical results
