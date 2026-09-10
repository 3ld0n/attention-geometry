# exp-139 — Gauge-fixed recomputation: key Gram (exp-127 analogue) and σ_K² (exp-136 analogue)

*Pre-registration — written before any analysis code; commit hash to be appended after push.*

**Date:** 2026-09-10 (physics room, 12:30 AM MDT)
**Motivation:** Wang & Wang (2025) characterize the complete gauge group of transformer
attention. For a single head, the gauge freedom is

  (W_Q, W_K) → (W_Q A, W_K A^{-T})  for any A ∈ GL(d_k)

preserving all attention scores. Under this freedom:
- K K^T → K A^{-T} A^{-1} K^T  (not gauge-invariant for non-orthogonal A)
- ‖W_K‖_F²  →  changes

The theory draft (attending_system_theory_draft.md v0.1, 2026-09-09, Appendix A) names
exp-127's key Gram eigenvalues and exp-136's σ_K² as gauge-dependent quantities and states
that a gauge-fixed recomputation is a condition on the application paper before those
results are cited. This experiment is that recomputation.

**Canonical gauge (QR):** Fix the gauge by requiring W_Q to have orthonormal columns.
QR decomposition of W_Q (shape n_embd × d_k): W_Q = U_Q R_Q, with U_Q orthonormal and
R_Q upper triangular. Then:
- W_Q^{can} = U_Q  (orthonormal columns)
- W_K^{can} = W_K R_Q^T  (the corresponding canonical key weight)
- k^{can} = k R_Q^T  (the canonical key activation, where k = h_ℓ W_K)

Canonical key Gram: K^{can} (K^{can})^T = K R_Q^T R_Q K^T

*(Pythia convention has W_Q_h, W_K_h of shape d_k × d_model; there the gauge transform
is (W_Q_h, W_K_h) → (A W_Q_h, A^{-T} W_K_h) and the canonical choice gives
W_K_h^{can} = R_R^T W_K_h where QR of W_Q_h^T = Q_R R_R.  Equivalently: k^{can} = k R_R
where k = (W_K_h @ x^T)^T.)*

---

## Part A — exp-127 analogue: key Gram eigenspectrum in canonical gauge (GPT-2 small)

**Hypothesis H1:** The exp-127 conclusion (Δ-window heads have lower λ₁/Σλ of the key
Gram than control heads, p = 0.0014, effect = 0.144) is preserved in the canonical QR gauge.

**Population definitions:** same as exp-127 (taken from exp-126):
- WIKI_HEADS (Δ-window, text-native, 16 heads)
- STRUCTURAL_HEADS (positional-mean carriers, 5 heads)
- CONTROL_HEADS (random non-window, 16 heads, same seed-42 selection as exp-126)

**Protocol:**
1. Load GPT-2 small (gpt2 checkpoint from HuggingFace cache).
2. For each head (layer ℓ, head h): extract W_Q_h and W_K_h from `block.attn.c_attn.weight`
   (Conv1D weight, shape n_embd × 3*n_embd). W_Q_h = weight[:, h*d_k:(h+1)*d_k] (n_embd × d_k).
3. QR decomposition of W_Q_h: W_Q_h = U_Q R_Q. Keep R_Q (d_k × d_k upper triangular).
4. Run 100 sequences × 128 tokens from WikiText-103 validation (same protocol as exp-126:
   sequential non-overlapping windows, seed 42 for the random token part does not apply here
   since the inputs are deterministic WikiText windows).
5. For each sequence, extract K_h (128 × d_k key activations per head) via forward hook.
6. Apply K_h^{can} = K_h @ R_Q^T  (canonical gauge key activations).
7. Compute eigenvalues of K_h^{can} (K_h^{can})^T / d_k  (128 × 128 Gram); sort descending.
8. Average eigenvalues (sorted) across 100 sequences.
9. Compute λ₁/Σλ (top eigenvalue share) and supra-MP fraction (same MP parameters as exp-127:
   p=64, n=128, σ²=Σλ/p, MP upper edge = σ²(1+√(p/n))²).

**Kill conditions:**
- K1 fires if effect size (median λ₁/Σλ_control − median λ₁/Σλ_Δ-window) < 0.05
- K2 fires if Mann-Whitney p (window < control, one-sided) ≥ 0.05
- H1 confirmed iff neither K1 nor K2 fires.
- H1 falsified iff either fires.

**Secondary (P3 analogue):** Is the structural population still distinguishable from both
window and control in the canonical gauge?

---

## Part B — exp-136 analogue: J_eff² with canonical coupling (Pythia-70m)

**Hypothesis H2:** The exp-136 conclusion (J_eff² grows monotonically with training step
and gates the formation phase transition, Spearman ρ = 0.934) holds when σ_K² is
replaced by the gauge-invariant canonical coupling σ_K_can².

**Protocol:** Same 11 checkpoints as exp-136 [0, 1, 4, 16, 64, 256, 1000, 4000, 16000, 64000, 143000],
same N=500 random tokens (seed 136).
1. For each checkpoint, load Pythia-70m.
2. For each head: extract W_Q_h (rows h*d_k to (h+1)*d_k of qkv_weight[0:d_model, :]) and
   W_K_h (rows h*d_k to (h+1)*d_k of qkv_weight[d_model:2*d_model, :]).
   Pythia shape: W_Q_h is d_k × d_model.
3. QR of W_Q_h^T (d_model × d_k): W_Q_h^T = Q_R R_R (R_R is d_k × d_k upper triangular).
4. Canonical key weight: W_K_h^{can} = R_R^T @ W_K_h  (shape d_k × d_model).
5. Compute canonical key activations k_can = (W_K_h^{can} @ x^T)^T / √d_k for the N tokens.
6. σ_K_can² = ‖W_K_h^{can}‖_F² / (d_k × d_model) = mean squared entry of W_K_h^{can}.
7. Recompute J_eff_can² = σ_K_can²² × Ω̂_can (Ω̂_can using K_can = k_can k_can^T, same
   double-centering formula as exp-136).
8. J_eff_can²_mean per checkpoint = mean of J_eff_can²[ℓ, h] across all layers and heads.

**Kill condition:**
- K3 fires if Spearman ρ(J_eff_can²_mean_series, training_step_list) < 0.50
- H2 confirmed iff K3 does not fire (ρ ≥ 0.50).
- H2 falsified iff K3 fires.

---

## Expected outcomes and interpretations

- **Both confirmed:** exp-127 and exp-136 conclusions are gauge-invariant; they may be
  cited in the application paper without reservation. The gauge issue is resolved.
- **H1 falsified:** the exp-127 λ₁/Σλ difference was a gauge artifact; remove from
  OVERVIEW.md, spine, and any draft. The weights-level structural signature needs
  a gauge-invariant redesign before it can be claimed.
- **H2 falsified:** the formation-gate timing (J_eff² growth at step 256) was
  gauge-dependent; the coupling-gate interpretation of A2 must be walked back.
- **H2 confirmed, H1 falsified (or vice versa):** the two results have different
  gauge properties; update each individually.

---

*Pre-registration complete. Commit this file before writing run.py.*
