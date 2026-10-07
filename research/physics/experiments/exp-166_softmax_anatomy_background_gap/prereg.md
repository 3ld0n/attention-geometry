# exp-166 Pre-Registration
**Title:** Softmax anatomy: background-gap mechanism for L3H4 pool-stability

**Date:** 2026-10-07 (local date at registration)
**Pre-registered:** before run.py is written (commit pushed to attention-geometry)
**Model:** GPT-2 small (gpt2) — via saved arrays only; no new forward passes
**Analysis type:** Analysis-only from exp-164's kbar_qbar.npz

---

## Background

exp-165 confirmed (H1, H3): the per-position mean-field —
  A_mf(i,j) = softmax_j{ qbar[i]·kbar[j] / √64 }
correctly orders pool-stable < pool-sensitive by Δσ (ρ=0.996). K2 fired:
σ_pred ≈ 2×σ_emp across all 15 pairs (Jensen gap from token fluctuations).

The predicted per-pool σ values from exp-165:
- L2H1 (sensitive): σ_B=0.558, σ_D=0.653, Δσ_pred=0.095
- L3H4 (stable):    σ_B=0.600, σ_D=0.626, Δσ_pred=0.026
- L5H0 (sensitive): σ_B=0.456, σ_D=0.578, Δσ_pred=0.122
- L7H11 (stable):   σ_B=0.420, σ_D=0.466, Δσ_pred=0.046
- L10H8 (sensitive): σ_B=0.583, σ_D=0.751, Δσ_pred=0.168

**The paradox:** L3H4's predicted σ_B (0.600) exceeds L2H1's (0.558), despite
L3H4 having a LOWER pool-B kbar absolute drift slope (1.85 vs 1.99 from exp-163/164).
The local slope in the census window is not the mechanism. Something about the
full S_mf distribution elevates L3H4's pool-B concentration.

---

## The question

What structural property of the mean-field score profile explains why L3H4
has higher predicted σ_B than L2H1, despite a shallower census-window slope?

**Candidate mechanism — background score gap:**
For a pool-B query at position i, the causal context is j=0..i. The census
measures attention at lags l=8..256 (key positions j=i-256..i-8: the "census
window"). The remaining positions j=0..i-257 are "background" — outside the
census window.

If L3H4's background scores are lower relative to the census window than
L2H1's, the softmax denominator is smaller relative to the census window's
numerator → more concentrated attention in the census window → higher σ_B.

This is a "background suppression" mechanism: even with a shallower slope
inside the census window, lower background scores produce higher census σ.

---

## Protocol

**Data source:** `exp-164_l3h4_bilinear_decomposition/kbar_qbar.npz`
Same arrays as exp-165. kbar[a] and qbar[i] are per-position mean vectors
over 200 random-token sequences, seed=42, SEQ_LEN=1024.

**Pool definitions (same as exp-163/164/165):**
- Pool B: query positions 256–511 (stride-2 sampling)
- Pool C: query positions 512–767 (stride-2 sampling)
- Pool D: query positions 768–1023 (stride-2 sampling)
- Census window: lags l ∈ [LAG_MIN=8, LAG_MAX=256], i.e., keys j ∈ [i-256, i-8]

**For each structural head h and each query i (stride-2 over each pool):**
1. Compute full causal score vector: S[j] = qbar[h,i]·kbar[h,j]/√64, j=0..i
2. Identify census window: j_win = [i-256, i-8] ∩ [0, i] (valid j range)
3. Identify background: j_bg = [0, i-257] ∩ [0, i] (positions before census window)
4. Compute:
   - census_mean(i) = mean(S[j] for j in j_win) if |j_win| > 0 else NaN
   - background_mean(i) = mean(S[j] for j in j_bg) if |j_bg| > 0 else NaN
   - background_gap(i) = census_mean(i) - background_mean(i)
     (positive = census window scores higher than background)
5. Pool over the query pool P: gap_P = mean(background_gap(i) for i in P
   where both census and background are non-empty)

**Also compute (for mechanism validation):**
For a single representative query in each pool (i=383 for pool B, i=895 for pool D),
compute the full score profile S[0..i] and apply softmax → plot for L3H4 vs L2H1.
Record: slope of S[j] vs j in the census window (j=i-256..i-8) and the
background score level (mean S[j] for j=0..i-257).

---

## Hypotheses

**H1 (Background gap elevates pool-B concentration):**
L3H4's pool-B background gap (gap_B_L3H4) > L2H1's pool-B background gap (gap_B_L2H1).

Operationalized: gap_B_L3H4 > gap_B_L2H1.

**H2 (Gap increase ranks with pool-sensitivity):**
The "gap increase" Δgap = gap_D - gap_B orders pool-stable < pool-sensitive:
max(Δgap for L3H4, L7H11) < min(Δgap for L2H1, L5H0, L10H8).

Operationalized: a clean ordering gap exists in Δgap between the two groups.

**H3 (Gap B predicts σ_B):**
Spearman ρ between gap_B and exp-165's σ_B_pred across all 5 structural heads > 0.7.

---

## Kill conditions

**K1 (H1 fails):** L3H4's gap_B ≤ L2H1's gap_B.
Interpretation: pool-B concentration elevation is not explained by background
suppression. The mechanism must lie elsewhere (e.g., score peaking structure
at specific key positions within the census window, not captured by window-mean
vs background-mean).

**K2 (H2 fails):** Gap increase Δgap does NOT order pool-stable < pool-sensitive
(max stable ≥ min sensitive).
Interpretation: the background-gap mechanism does not generalize across all 5
structural heads; some other structural property governs pool-stability.

---

## Expected outcomes

If H1–H3 all confirmed: pool-B concentration for L3H4 is explained by lower
background scores (relative to the census window), not a higher local slope.
The mechanism: L3H4's kbar vectors at positions 0..i-257 are relatively
less aligned with pool-B query vectors than L2H1's, suppressing background
attention → census window captures a larger share of the softmax mass →
higher σ_B.

If K1 fires (L3H4's gap_B not higher): the mechanism is more subtle — likely
in the within-census-window score distribution shape (e.g., curvature, not
just mean level). Candidate follow-on: compute softmax-concentration of the
census window only (treating the window as the full context) to isolate the
local shape from the background effect.

If K2 fires (gap increase doesn't rank): the background-gap is idiosyncratic
to L3H4/L2H1, not a general pool-stability mechanism. Would suggest each
head's pool-stability has a distinct structural source.

---

## Honest negative if all kill conditions fire

The background-gap mechanism fails. The source of L3H4's predicted σ_B
elevation (σ_B_pred=0.600 > L2H1's 0.558 despite shallower local slope) is
not captured by window-mean vs background-mean decomposition. Write the
negative clearly: what the gap analysis cannot explain, and propose the
next candidate (within-window curvature analysis).
