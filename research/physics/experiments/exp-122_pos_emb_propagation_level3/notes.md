# exp-122 — Theory-of-A Level 3: Position-Embedding Propagation Route

**Date:** 2026-08-17  
**Type:** Analysis-only (existing GPT-2 weights; no forward passes)  
**Registered:** 2026-08-17, before any analysis was run  
**Follows:** exp-119 (self-consistency falsified), exp-117 (accumulated delta has σ_delta ≈ 0.249 at L2H1)

---

## Background

exp-117 found that the accumulated attention delta (h̄^(ℓ) − h̄^(0)) dominates the mean
residual stream at structural head layers by 13–32× in norm and 12–26× in positional
variance. The position-correlation profile of this accumulated delta has slope σ_delta ≈ 0.249
at L2H1 — matching the census exponent Δ ≈ 0.249 (exp-107).

exp-119 falsified the single-layer self-consistency mechanism: single-layer head output is
approximately position-independent under random-token census inputs (σ_within ≈ 0, R² < 0.70
for all structural heads).

Two remaining candidates for the origin of σ_delta ≈ 0.249:

1. **Position-embedding propagation (this experiment):** pos_emb[i] has position-dependent
   structure. After projection through W_V and convolution with the conformal attention kernel
   ā(dx) ~ dx^{-2Δ}, the output may inherit a power-law position-correlation profile.

2. **Multi-layer compositional effects:** The power law emerges from many layers interacting,
   not from the first-layer application of the attention mechanism to pos_emb.

---

## Pre-registration

**Hypothesis under test (stated before any computation):**

When GPT-2's learned positional embeddings pos_emb[j] are projected through W_V for a
structural head and convolved with an analytic causal conformal attention kernel
ā(i,j) = C × (i−j)^{−2Δ} (Δ = 0.249, causal mask j < i, normalized per row), the
resulting output's position-correlation profile has a measurable power-law slope σ_out.

The theoretical prior: for white-noise pos_emb (iid rows), the cross-correlation of a
power-law kernel with exponent 2Δ = 0.498 gives C_out(dx) ~ dx^{1−4Δ} = dx^{0.008} ≈
const — predicting σ_out ≈ 0. GPT-2's learned pos_emb is not white-noise (exp-117: sign
flip in C_pos at dx ≈ 180), so the actual result may differ.

**Protocol:**

1. Load GPT-2 small weights (cpu, no forward pass).
2. Extract wpe (positional embeddings), shape (1024, 768). Use positions 0..511.
3. For each structural head (L2H1, L3H4, L5H0, L7H11, L10H8):
   a. Extract W_V_h: c_attn.weight[:, 2*768:3*768][:, h*64:(h+1)*64], shape (768, 64).
   b. Compute v[j] = pos_emb[j] @ W_V_h, shape (512, 64).
   c. Build causal power-law attention matrix: A[i,j] = (i−j)^{−2Δ} for j < i (else 0),
      normalized per row so each row sums to 1 (row i = 0 → zero output, skip).
   d. Compute out[i] = Σ_j A[i,j] × v[j], shape (512, 64).
   e. Normalize: out_norm[i] = out[i] / ‖out[i]‖ (skip i=0 where A is zero).
   f. Compute correlation matrix: corr[i,j] = out_norm[i] · out_norm[j].
   g. Apply pooled_window_profile (lags 8..256, deep_lo=256 baseline), giving C(dx) of length 249.
   h. Fit log-log slope σ_out = −ols_slope(C, WINDOW) and R².
4. Compare σ_out to σ_delta = 0.249 (exp-117 L2H1 reference).

**Pre-registered predictions:**

- **P1 (any positive slope):** σ_out > 0.10 with R² ≥ 0.70 for L2H1. The convolved pos_emb
  has meaningful positive position correlation — the mechanism is at least partially active.
- **P2 (slope match):** σ_out ∈ [0.20, 0.30] for L2H1. The conformal kernel transmits the
  census exponent to the pos_emb-driven output — route confirmed.
- **P3 kill (route falsified):** σ_out < 0.05 for all 5 structural heads, OR σ_out is
  negative for L2H1. The pos_emb propagation route is not the source of σ_delta ≈ 0.249;
  multi-layer composition is the remaining candidate.

If P3 kills, the Level-3 question has a single remaining candidate: multi-layer compositional
effects (exp-123, not this experiment).

---

## Results

*(To be filled after running run.py)*

| Head | σ_out | R² | C(8) | C(256) | P1 | P2 |
|---|---|---|---|---|---|---|
| L2H1 | — | — | — | — | — | — |
| L3H4 | — | — | — | — | — | — |
| L5H0 | — | — | — | — | — | — |
| L7H11 | — | — | — | — | — | — |
| L10H8 | — | — | — | — | — | — |

Overall verdict: —

---

## Artifacts

- `notes.md` — this file (pre-registration; written and committed before any computation)
- `run.py` — analysis script (written after pre-registration commit)
- `results.json` — results (written after run)
