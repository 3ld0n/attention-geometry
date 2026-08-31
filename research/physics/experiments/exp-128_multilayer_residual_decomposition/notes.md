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

---

## Results — 2026-08-31

**Pre-registration commit:** ce1934c (pushed before run.py was written).

### Decomposition verification

h^(2) = h^(0) + attn^(0) + mlp^(0) + attn^(1) + mlp^(1) — relative error 5.81e-09.
The decomposition identity holds exactly; the hooks captured the correct objects.

### Per-component position-correlation slopes

| Component | σ | R² | Notes |
|---|---|---|---|
| h^(0) (embedding) | **0.451** | 0.859 | Embedding is strongly position-correlated — larger than Δ! |
| attn^(0) | **0.223** | 0.825 | Block-0 attention writes ≈ Δ |
| mlp^(0) | **0.228** | 0.849 | Block-0 MLP writes ≈ Δ |
| attn^(1) | 0.052 | 0.654 | Block-1 attention writes: weaker |
| mlp^(1) | 0.056 | 0.930 | Block-1 MLP writes: weaker but clean power-law |
| delta_after_attn0 | 0.223 | 0.825 | = attn^(0) alone |
| delta_after_block0 | 0.226 | 0.828 | = attn^(0) + mlp^(0) |
| delta_after_attn1 | 0.204 | 0.825 | block 1 attn reduces slope |
| **delta_total** | **0.189** | 0.827 | = Δ_total, primary measurement |
| h^(2) (full residual) | 0.193 | 0.827 | matches delta_total closely |

### Registered verdicts

- **P1 FAIL** — σ(Δ_total) = 0.189, outside [0.20, 0.30]. Does not reproduce exp-117's σ_delta ≈ 0.249.
- **P2 OK** — σ(attn^(0)) = 0.223 > 0.05.
- **P3 OK** — σ(attn^(1)) = 0.052 > 0.05 (barely).
- **P4 OK** — σ(Δ after block 0) = 0.226 > 0.05.
- No kill condition fired (0.15 < 0.189 < 0.35).

**Overall verdict: FALSIFIED** (P1 fails; no kill condition).

### What the result reveals

**P1's failure is likely a protocol mismatch with exp-117.** exp-117 computed C_delta from
"(delta / ||delta||) @ (delta / ||delta||)^T" — where ||delta|| may be the Frobenius norm
of the (512, 768) delta matrix, not the per-row L2 norm. My protocol uses per-row cosine
similarity, which normalizes out per-position magnitude variation. These capture different
aspects of position-correlation: Frobenius normalization preserves the dominance of the
leading mode of position-variation; per-row normalization treats all positions equally
regardless of their write magnitude.

The σ = 0.189 is directionally consistent with 0.249 but ~24% lower. If the Frobenius
normalization were used instead, the result would likely be dominated by the strongest
position-correlation direction across all positions — which might be closer to the exp-117
value. This is worth testing in a follow-up (exp-129) before concluding that the Level-3
question is open.

**Three findings that hold regardless of protocol:**

1. **Block-0 writes are conformal-exponent-scale position-correlated.** σ(attn^(0)) = 0.223
   ≈ Δ ≈ 0.249. The conformal self-transmitting mechanism (exp-122) operates at the WRITE
   level: each attention block writes something position-correlated at the conformal scale.
   This is consistent with the mechanism that exp-122 identified — the conformal attention
   weights weight earlier positions more heavily, and the write inherits this positional bias.

2. **Block-0 MLP writes are also ≈ Δ position-correlated.** σ(mlp^(0)) = 0.228. The MLP
   is not destroying the position-correlation established by the attention — it contributes
   a comparably-correlated write. The gate-activation in MLP processes position-correlated
   attention output and produces position-correlated writes.

3. **Block-1 writes REDUCE the cumulative delta's slope.** The cumulative slope falls from
   0.226 (after block 0) to 0.189 (after block 1). attn^(1) and mlp^(1) each have σ ≈ 0.05-0.06,
   but their addition to the block-0 delta reduces the overall per-row cosine slope. This
   happens because the later writes introduce position-correlated content in a DIFFERENT
   high-dimensional direction than the block-0 writes, and their per-row normalization
   averages both structures, reducing the apparent slope.

**The embedding surprise.** σ(h^(0)) = 0.451 — the embedding itself is much more
position-correlated (per-row cosine) than Δ = 0.249. GPT-2's learned positional embeddings
encode very strong local similarity structure (exp-117: C_pos[dx=8] = +10.9, C_pos[dx=256]
= -2.5). Attention and MLP writes (σ ≈ 0.22 for block 0) partially DILUTE this structure
by adding writes in different directions — the full residual at L2 input (σ = 0.193) is
actually less position-correlated per row than the bare embedding.

This is unexpected and potentially important: the attention mechanism may function as a
position-correlation DAMPER, not a builder, when measured against the embedding baseline.
The census exponent Δ ≈ 0.249 measures the SCORE function (attention weights), not the
residual stream's position-correlation per se. The connection between the score's Δ and
the residual stream's σ is more indirect than the theory-of-A chain assumed.

### Open questions seeded by this experiment

1. **Protocol clarification (exp-129):** Rerun the measurement using Frobenius normalization
   of delta rather than per-row cosine similarity. Does this recover σ ≈ 0.249? If yes, the
   Level-3 explanation is: the dominant mode of position-variation in the accumulated delta
   has slope ≈ Δ, even if the per-row-cosine measure gives a smaller value.

2. **Why does attn^(0) write have σ ≈ Δ?** This is itself the Level-3 mechanism. The
   conformal attention weights (A(i,j) ~ |i-j|^{-2Δ}) sum over the value vectors, producing
   an output weighted toward nearby positions. The positional variation of this output should
   have slope Δ — but exp-119 found single-layer self-consistency fails (σ ≈ 0 for structural
   heads in census inputs). The difference must be the WikiText data: WikiText inputs have
   genuine content variation across positions, unlike the random-token census. This is
   input-distribution-specific.

3. **The embedding domination:** σ(h0) = 0.451 is not discussed in the exp-117 notes (which
   only discuss delta, not the embedding itself). Checking whether the exp-117 measurement
   of σ_delta used delta = h̄^(L) − h̄^(0) (as stated) or h̄^(L) itself (which would include
   the embedding) might resolve the discrepancy with 0.249.

### Next experiment (exp-129 candidate)

**Protocol comparison:** Re-run the same forward passes but compute σ using:
(a) Frobenius normalization of delta (matching exp-117's notation)
(b) per-row cosine similarity (this experiment)
(c) σ of h̄^(L) itself (not the delta)

This is analysis-only from saved data if we save the positional mean arrays — register before
computing. If (c) recovers σ ≈ 0.249, this would mean exp-117 was measuring σ of the
RESIDUAL STREAM itself, not of the delta, despite the notation.
