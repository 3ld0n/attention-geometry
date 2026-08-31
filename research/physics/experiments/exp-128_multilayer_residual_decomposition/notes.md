# exp-128 — Theory-of-A Level 3: Multi-Layer Residual Stream Decomposition

**Pre-registration.** This file was committed to attention-geometry before the
analysis script was written or any forward passes were run. Commit hash appended
after push.

**Date pre-registered:** 2026-08-31  
**Analysis type:** Forward-pass extraction (new forward passes required)  
**Follows:** exp-123 (LN-corrected single-layer route FALSIFIED), exp-117 (σ_delta = 0.249 at L2H1)

---

## Background

The theory-of-A Level-3 question is: **why does the accumulated attention delta at
L2H1's input have σ_delta ≈ 0.249, matching the census exponent Δ?**

Two single-layer routes have been exhausted:

- **exp-119 (self-consistency, FALSIFIED):** Single-layer head output is
  position-independent under random-token census inputs (σ_within ≈ 0 for all
  structural heads). The self-consistency mechanism does not operate at a single layer.

- **exp-122 (pos_emb propagation, CONFIRMED as mechanism, not quantitative):**
  Positional embeddings projected through W_V and convolved with the causal conformal
  kernel produce σ_out ∈ [0.18, 0.28]. The conformal kernel is self-transmitting.
  But quantitative closure required LN correction.

- **exp-123 (LN-corrected single-layer, FALSIFIED):** Applying layer norm to
  h̄^(0) = emb_mean + wpe before W_V projection *reduces* σ_out (L2H1 d_model: 0.116
  vs 0.214 baseline). The single-layer pos_emb route cannot explain σ_delta = 0.249.
  LN normalization homogenizes position vectors and reduces the correlations the
  conformal convolution was exploiting.

exp-123's conclusion: the actual input to structural heads is the residual stream
**after layers 0..ℓ−1**, which exp-117 showed is dominated by accumulated attention
delta (13–32× larger than the embedding contribution in norm). The multi-layer
accumulated residual stream is the correct object to analyze.

The conformal self-transmitting mechanism (exp-122) is the right picture — but it
must operate through multi-layer composition, not from a single application of the
conformal kernel to the embedding. **This experiment tests whether the per-layer
attention writes build constructively across layers 0 and 1 to produce σ_delta ≈ 0.249
at L2's input.**

---

## Hypothesis

The position-correlated delta at L2H1's input (σ_delta ≈ 0.249, exp-117) arises from
multi-layer accumulation of attention and MLP writes across GPT-2's blocks 0 and 1.
The per-layer attention contributions are each positively position-correlated with
slope > 0, and their cumulative sum (with MLP contributions) accounts for σ_delta ≈ 0.249.

---

## Protocol

**Model:** GPT-2 small (`gpt2`), fp32, eager attention.

**Data:** WikiText-103 validation split, non-empty lines, consecutive non-overlapping
512-token windows, 100 windows. Same tokenizer and stride as exp-113 (mean-field
analysis). Sequence length: 512 tokens.

**Residual stream extraction:** Forward pass with hooks capturing:
- `h^(0)`: output of the embedding layer (wte + wpe, before block 0) — hook on
  `model.transformer.h[0]` pre-input.
- `attn^(0)`: output of block 0's attention module (c_proj applied, before residual
  add) — hook on `model.transformer.h[0].attn`.
- `mlp^(0)`: output of block 0's MLP module (c_proj applied, before residual add) —
  hook on `model.transformer.h[0].mlp`.
- `attn^(1)`: output of block 1's attention module — hook on
  `model.transformer.h[1].attn`.
- `mlp^(1)`: output of block 1's MLP module — hook on `model.transformer.h[1].mlp`.
- `h^(2)`: input to block 2 (= output of block 1) — hook on
  `model.transformer.h[2]` pre-input.

Decomposition identity (verified in script):
`h^(2) = h^(0) + attn^(0) + mlp^(0) + attn^(1) + mlp^(1)`

**Position-correlation slope measurement:** For each component X (shape after averaging
over 100 sequences: (512, 768)):
1. Compute positional mean: X̄[i] = mean over sequences of X[:, i, :], shape (512, 768).
2. Per-row normalize: X̄_n[i] = X̄[i] / ||X̄[i]||₂, shape (512, 768).
3. Cosine similarity matrix: C_X = X̄_n @ X̄_n^T, shape (512, 512).
4. Apply pooled_window_profile (deep_lo=256, lags dx ∈ [8, 256]) → 249-element profile.
5. OLS slope in log-log: σ_X = −ols_slope(profile, lags). R² of the fit.

Components measured:
- σ(h^(0)): baseline embedding slope (expected: small, from exp-117 ~0.034 pos-var norm)
- σ(attn^(0)): layer-0 attention write slope
- σ(mlp^(0)): layer-0 MLP write slope
- σ(attn^(1)): layer-1 attention write slope
- σ(mlp^(1)): layer-1 MLP write slope
- σ(Δ₀ = attn^(0) + mlp^(0)): cumulative delta after block 0
- σ(Δ₁ = Δ₀ + attn^(1)): cumulative after block 1's attention (before MLP)
- σ(Δ_total = h^(2) - h^(0)): full cumulative delta at L2 input — primary measurement

Also record: decomposition verification (does h^(0) + attn^(0) + mlp^(0) + attn^(1) + mlp^(1) ≈ h^(2)?).

---

## Pre-registered Predictions

**P1 (primary — reproducing exp-117):**  
σ(Δ_total) ∈ [0.20, 0.30], R² ≥ 0.70.  
*Reproduces the exp-117 finding that the accumulated delta has power-law position
correlation with slope ≈ Δ ≈ 0.249.*

**P2 (layer-0 attention write):**  
σ(attn^(0)) > 0.05.  
*Layer-0 attention writes are positively position-correlated — consistent with the
conformal self-transmitting mechanism contributing at the first layer.*

**P3 (layer-1 attention write):**  
σ(attn^(1)) > 0.05.  
*Layer-1 attention writes are also positively position-correlated.*

**P4 (cumulative build after block 0):**  
σ(Δ₀ = attn^(0) + mlp^(0)) > 0.05.  
*The residual stream delta is already positively position-correlated after block 0 alone,
before layer 1 contributes.*

---

## Kill Conditions

**K1:** σ(Δ_total) < 0.15. Cannot reproduce exp-117; protocol mismatch or methodology
failure. Record as INCONCLUSIVE.

**K2:** σ(Δ_total) > 0.35. Measured value inconsistent with exp-117's ≈ 0.249. Record
as INCONCLUSIVE; investigate why.

**K3:** Both P2 and P3 false (σ(attn^(0)) ≤ 0 and σ(attn^(1)) ≤ 0). Attention
writes are not the source of the position-correlation; MLP writes dominate. Record this
explicitly.

---

## Verdict Criteria

| Outcome | Verdict |
|---|---|
| P1 confirmed and (P2 or P3 confirmed) | `confirmed` — multi-layer attention accumulation explained |
| P1 confirmed, P2 and P3 both false, P4 confirmed | `partial` — MLP writes are the dominant source |
| K1 or K2 fires | `inconclusive` |
| P1 confirmed but no other prediction holds | `partial` |
| P1 false and no kill condition | `falsified` |

---

## Connection to the Program

Level 3 of the theory-of-A chain. If confirmed, this closes the quantitative
explanation of how the conformal geometry in the score function arises: not from
a single conformal layer applied to positional embeddings, but from multi-layer
accumulation of attention writes that are each themselves conformal-self-transmitting.
This is consistent with the self-transmitting mechanism (exp-122) operating
iteratively, with each layer's conformal attention writes feeding the next layer's
conformal scoring.

Connects to:
- exp-117: the σ_delta ≈ 0.249 measurement this experiment is explaining
- exp-122: the conformal kernel self-transmitting mechanism (single-layer)
- exp-123: single-layer LN-corrected route falsified — multi-layer is the residual
- The spine's T3/T4 links (formation and propagation of conformal structure)
