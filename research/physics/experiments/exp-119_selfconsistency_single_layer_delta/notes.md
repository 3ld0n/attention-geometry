# exp-119 — Theory-of-A Level 3: self-consistency of the single-layer attention delta

**Pre-registered:** 2026-08-11, physics room session (12:17 PM MDT), Ariel solo.
**Committed to attention-geometry before any forward pass.**
**Context:** exp-117 falsified the embedding-geometry route to Level 3 and found
a self-consistency candidate: the accumulated attention delta (h̄^(ℓ) - h̄^(0))
has power-law position correlation with slope σ_delta ≈ 0.249 at L2H1 ≈ Δ_census ≈ 0.25.

**The open question from exp-117:** Is σ_delta ≈ Δ because each individual
attention layer deposits a Δ-correlated delta (self-consistent single-layer), or
because multi-layer accumulation happens to produce Δ in the aggregate (non-trivial
compositional effect)? exp-117 measured only the accumulated (multi-layer) delta.

---

## Hypothesis

For the 5 structural heads (L2H1, L3H4, L5H0, L7H11, L10H8), the **single-layer**
attention output — the head's contribution to the residual stream from one forward
pass through one layer — has measurable power-law position-correlation structure.

Two measures are registered:

**Measure A (within-input):** For each input n, the head output head_out^(n)[i]
(shape: seq × d_head) has within-sequence position correlation
C^(n)(dx) = mean_i( head_out^(n)[i] · head_out^(n)[i+dx] ), averaged over N inputs.
Slope: σ_within.

**Measure B (mean-field):** The mean head output over N inputs,
head_out_mean[i] = (1/N) Σ_n head_out^(n)[i], has position correlation
C_mf(dx) = mean_i( head_out_mean[i] · head_out_mean[i+dx] ).
Slope: σ_mf_single. This parallels the measure used in exp-117 (mean residual stream).

The census protocol (N_INPUTS=50 random-token sequences, SEQ_LEN=512) is used
throughout, matching the existing Δ measurement protocol.

---

## Registered predictions

### P1 — Within-input power-law structure
**Prediction:** C_within(dx) has power-law decay (R² ≥ 0.70) with positive slope
σ_within > 0 for at least 3 of 5 structural heads.

**Kill K1:** R² < 0.70 AND/OR σ_within < 0 for a majority (≥ 3) of structural heads
→ the single-layer delta has no power-law position structure under random-token input;
self-consistency does not operate through this channel.

### P2 — Self-consistency slope test
**Prediction (exploratory):** σ_within ≈ Δ_census ≈ 0.25 for the structural heads.
This is the "one power less" prediction: if attention weights A(i,j) ~ |i-j|^{-2Δ},
then the output of applying A to position-structured input should have correlation
exponent reduced by one factor of Δ.

**Not a kill condition:** σ_within ≠ 0.25 does not kill anything — it is informative
about what mechanism produces the accumulated delta result in exp-117.

**Kill K2 (strong form only):** σ_within > 0.45 for a majority of structural heads
→ the single-layer delta's correlation is closer to the raw attention exponent (2Δ ≈ 0.5)
than to the reduced exponent Δ, suggesting the mechanism is dominated by the
attention weight lag profile itself rather than a self-consistency fixed point.

### P3 — Mean-field collapse
**Prediction:** C_mf_single(dx) is approximately flat (|slope| < 0.05) for all
structural heads under random-token input — the mean head output has little
position structure under random tokens, because the mean-token embedding is
roughly position-independent in content.

If P3 holds: the self-consistency must operate through the fluctuation channel
(within-input), not the mean-field channel, explaining why exp-117's mean-field
measure of the accumulated delta picks up structure only over many layers.

If P3 fails (C_mf_single has structure): revisit the mean-field positional
embedding argument from exp-117 for the single-layer case.

### P4 — Semantic vs structural comparison
**Exploratory (no kill):** Run the same measures on 5 representative semantic heads
(text-native Δ-window heads from exp-109: L4H4, L7H5, L8H3, L9H1, L10H3) under
the same RANDOM TOKEN protocol. If semantic heads show different σ_within than
structural heads under random tokens, the two populations have distinct single-layer
dynamics even when the input regime isn't the one that "activates" them.

---

## Protocol

1. Load GPT-2 small (gpt2, same model as census experiments).
2. Generate N_INPUTS=50 random-token sequences (uniform in [0, vocab_size), SEQ_LEN=512),
   seeded for reproducibility (SEED=42, same as census).
3. Forward pass with hooks to capture head_out^(n)[h][i] at each structural head
   (layer, head) and 5 representative semantic heads.
4. Compute Measure A: C_within(dx) = (1/N) Σ_n mean_i( head_out^n[i] · head_out^n[i+dx] )
   over dx ∈ [2, 256], using lags ≥ 32 tokens from the sequence end (i > 256).
5. Compute Measure B: head_out_mean[i] = (1/N) Σ_n head_out^n[i]; C_mf(dx) as above.
6. Fit OLS in log-log space over dx ∈ [8, 256] (matching census fit range).
7. Report σ_within, σ_mf_single, R² for each head under both measures.
8. Also: save the raw C_within(dx) profiles to results.json.

**Note on head_out definition:** GPT-2 attention head output per head h is:
  head_out[h][i] = (A[h](i,:) @ V[h])  — shape (d_head=64,)
before the W_O projection. Using pre-W_O output avoids mixing across heads.
The d_head=64 dot product is what "·" means in the correlation formula.

---

## What this does NOT test

- Whether the self-consistency is causal (whether conformal A CAUSES conformal delta,
  or both are consequences of a third thing)
- The mechanism that makes A conformal (still the open Level 3 question)
- Multi-layer accumulation effects (that would require the full network analysis)

---

## Expected runtime

~3 minutes on M5 Max MPS (GPT-2 small, 50 inputs × 512 tokens).

---

## What happens next

If P1 is confirmed (power-law within-input structure exists):
- OVERVIEW.md updated with the single-layer finding
- If σ_within ≈ Δ, this is the strongest single-experiment evidence yet for the
  self-consistency interpretation. The Level-3 derivation is reformulated around it.
- If σ_within ≠ Δ, report the actual value and what it implies for the theory.

If K1 fires (no power-law structure):
- The self-consistency cannot be demonstrated at the single-layer level under
  random-token inputs.
- Try: repeat with WikiText inputs (semantic heads should be activated). Register separately.
- The accumulated delta result from exp-117 remains real but requires
  a multi-layer explanation.

*Written by Ariel, 2026-08-11, physics room session (12:17 PM MDT arrival).*
*Committed to attention-geometry before any forward pass or data access beyond weight loading.*
