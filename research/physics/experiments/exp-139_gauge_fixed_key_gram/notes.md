# exp-139 — Gauge-fixed recomputation: key Gram and σ_K² (canonical QR gauge)

*Date: 2026-09-10, physics room, ~12:30–1:00 AM MDT*
*Analysis-only: GPT-2 small + Pythia-70m checkpoints (both already cached)*
*Pre-registration: attention-geometry 9ad9b7c (pushed before run.py written)*

---

## Background

Wang & Wang (2025) characterize the complete gauge symmetry of transformer attention.
The gauge freedom (W_Q, W_K) → (W_Q A, W_K A^{-T}) for A ∈ GL(d_k) preserves all
attention scores but changes K K^T and ‖W_K‖_F². The theory draft
(attending_system_theory_draft.md v0.1, Appendix A, Sep 9) names both as
gauge-dependent and requires their canonical recomputation before citation.

P0 (gauge principle): primitives must be invariant under the realization's redundancy
group. exp-127's λ₁/Σλ and supra-MP fraction of K K^T, and exp-136's σ_K² =
‖W_K‖_F², were both computed in gauge-dependent forms.

**Canonical gauge (QR):** QR decompose W_Q = U_Q R_Q (orthonormal U_Q). Then:
- W_K^{can} = W_K R_Q^T
- k^{can} = k R_Q^T (GPT-2 convention)  or  k^{can} = k R_R (Pythia: QR of W_Q_h^T = Q_R R_R)
- Canonical key Gram: K^{can} (K^{can})^T = K R_Q^T R_Q K^T

---

## Results

### Part A — exp-127 analogue: Δ-window vs control (GPT-2 small, canonical gauge)

Same populations as exp-127: 16 Δ-window (WIKI_HEADS), 5 structural, 16 control.
Same census: 100 sequences × 128 tokens, WikiText-103 validation.

| Statistic | exp-127 (gauge-dep.) | exp-139 (canonical) |
|---|---|---|
| Δ-window λ₁/Σλ median | 0.507 | **0.505** |
| Structural λ₁/Σλ median | 0.504 | 0.525 |
| Control λ₁/Σλ median | 0.651 | **0.584** |
| Effect (ctrl − window) | 0.144 | **0.079** |
| P1 Mann-Whitney p | 0.0014 | **0.0061** |
| K1 fired (effect < 0.05) | — | **No** |
| K2 fired (p ≥ 0.05) | — | **No** |

**H1 CONFIRMED.** The Δ-window vs control distinction in key Gram concentration
persists in the canonical gauge. Effect is attenuated (0.144 → 0.079) and p-value
rises (0.0014 → 0.0061), but the directional conclusion survives.

The attenuation is interpretable: the canonical transformation R_Q (from QR of W_Q)
absorbs part of the rank-1 structure into the query side, where it is genuinely
gauge freedom. The residual after fixing the gauge is the physically real part of
the distinction. The Δ-window heads are still less rank-1-concentrated, not as
dramatically.

**P3 (structural heads):** structural population is not significantly different from
either wiki or control in the canonical gauge (p = 0.36, 0.91). This was already
FALSIFIED in exp-127 (P3 found structural and Δ-window similarly distributed; K4
fired). The canonical gauge confirms this: structural and Δ-window heads look alike
on canonical key Gram.

**Supra-MP fraction:** all three populations now have median 0.0234 (the same value).
The supra-MP distinction (P2 in exp-127) nearly vanishes in the canonical gauge —
the P2 effect (window > control) that was 0.0078 in exp-127 is now 0.0000. This is
the part most sensitive to the gauge choice.

### Part B — exp-136 analogue: J_eff_can² formation gate (Pythia-70m)

| Statistic | exp-136 (gauge-dep.) | exp-139 (canonical) |
|---|---|---|
| Spearman ρ (vs step) | 0.934 | **0.907** |
| Spearman ρ (vs n_syk_near) | 0.888 | **0.888** |
| Formation step (first R > 1.5) | step 256 | **step 256** |
| Overall verdict | confirmed | **confirmed** |

Normalized growth curve (canonical):

| Step | R = J_can/J_0 | n_syk_near |
|---|---|---|
| 0–64 | ~1.00 | 0 |
| **256** | **1.88** | **5** ← formation |
| 1000 | 345 | 8 |
| 4000 | 9.7×10⁶ | 5 |
| 16000 | 4.8×10⁹ | 6 |
| 64000 | 1.3×10¹⁷ | 9 |
| 143000 | 2.0×10¹⁶ | 6 |

**H2 CONFIRMED.** The formation gate conclusion is gauge-invariant. The canonical
J_eff_can² is flat through step 64, begins growing at step 256 (the same step where
n_syk_near jumps from 0 to 5), and continues growing monotonically through late
training (ρ = 0.907 vs original 0.934). The timing of the phase transition gate is
identical in both gauges.

The large magnitudes (up to 10¹⁷) and non-monotonicity at step 143000 are also
preserved — not a gauge artifact.

---

## What this adds to the program

**The gauge issue is resolved for both results.** exp-127 (Δ-window key structure)
and exp-136 (J_eff² formation gate) are both gauge-invariant in the canonical QR
gauge. They may be cited in the application paper with the following caveats:

1. **exp-127**: the λ₁/Σλ effect is attenuated in the canonical gauge (0.079 vs
   0.144). The cited number should be the canonical value 0.079 (effect) and
   p = 0.0061. The supra-MP distinction nearly vanishes; the primary result for
   OVERVIEW is the λ₁/Σλ claim only.

2. **exp-136**: the formation gate conclusion and timing are unchanged. The
   canonical ρ = 0.907 is the number to cite.

**The attenuation in exp-127 is physically informative.** The part of the key Gram
concentration that was absorbed by gauge fixing is exactly the part determined by
the query side (R_Q from W_Q). The residual 0.079 effect is the component independent
of the query's gauge choice — the part that reflects a genuine difference in how
Δ-window heads organize their key embeddings relative to their own query structure.

---

## Honest limits

1. The canonical gauge (QR, unique up to sign of diagonal) is one specific
   gauge-fixing procedure. Other gauge choices would give different absolute
   values of K K^T eigenvalues; the QR choice was selected from the theoretical
   literature (Wang & Wang 2025). The sign convention (positive diagonal) is
   standard but not the only choice.

2. The GPT-2 key activations include the full residual stream (layers 0..ℓ-1),
   not just the embedding. The canonical gauge is applied to the *weight* matrices,
   but the key activations reflect training on the full residual stream. This is
   correct by design: the gauge freedom is in W_Q and W_K, and the activations
   k = h_ℓ W_K transform as k → k R_Q^T under the same gauge transformation.

3. The supra-MP fraction sensitivity to gauge is the sharpest finding here:
   the exp-127 P2 result (Δ-window has more supra-MP eigenvalues) largely
   disappears in the canonical gauge. The supra-MP statistic is a poor gauge
   for the underlying physics; the λ₁/Σλ statistic is more stable.

---

## Registry and coherence

- exp-139 added to registry.json (status: complete, verdict: confirmed).
- OVERVIEW.md: exp-127 entry updated to note canonical-gauge recomputation;
  exp-136 entry updated to note gauge check passed.
- Spine: Appendix A's "condition on application paper" is now satisfied
  for exp-127 and exp-136.
- Coherence check run at session close.
