# exp-165 Pre-Registration
**Title:** Mean-field softmax nonlinearity as pool-stability mechanism

**Date:** 2026-10-07  
**Pre-registered:** before run.py is written (commit pushed to attention-geometry)  
**Model:** GPT-2 small (gpt2) — via saved arrays only; no new forward passes  
**Analysis type:** Analysis-only from exp-164's kbar_qbar.npz

---

## Background and motivation

exp-163 established pool-stability via census Δσ: L3H4 Δσ=0.014, L7H11 Δσ=0.018
(pool-stable); L2H1 Δσ=0.052, L5H0 Δσ=0.056, L10H8 Δσ=0.070 (pool-sensitive).

exp-163 also found that S_abskey alone (the kbar cross-pool range) does not explain
pool-stability — L3H4's kbar range (5.44) actually exceeds pool-sensitive L2H1 (5.20).

exp-164 tested whether the relative term S_relative[i,j] = dk_q[i]·dk_k[j]/√d cancels
the S_abskey effect. H_relative FALSIFIED (K1 fired): the relative term increases
(becomes less negative) from pool B to D for all heads — it moves in the same direction
as S_abskey and does not discriminate pool-stable from pool-sensitive.

exp-164 named the surviving candidate: **softmax nonlinearity**. The linear slope analyses
in exp-163 and exp-164 treat S_abskey and S_relative additively. But the actual attention
weight A(i,j) = softmax(S_abskey + S_relative + ...)[j], a nonlinear operation that may
produce qualitatively different pooled profiles than the linear decomposition predicts.

## The question

Does the mean-field attention simulation — computing
  A_approx(i, j) = softmax_j { qbar[h,i] · kbar[h,j] / √d } (causal)
from the saved kbar and qbar arrays — recover the empirical census σ values, and
does it preserve the pool-stable vs pool-sensitive ordering?

If yes: pool-stability is a mean-field effect. The census Δσ ordering follows from the
average key/query geometry; per-token fluctuations are not required to explain it.

If no (K1 fires): the mechanism requires fluctuation structure — the variance of
(dk_q[i]·dk_k[j]) across the 200 sequences matters, not only the means.

This is a fundamental question about what the census measurement actually measures.

---

## Protocol

**Data source:** `exp-164_l3h4_bilinear_decomposition/kbar_qbar.npz`
Keys: kbar_LxHy (1024, 64) and qbar_LxHy (1024, 64) for each of the 5 structural
heads (L2H1, L3H4, L5H0, L7H11, L10H8). kbar[a] = mean key vector at absolute position
a; qbar[i] = mean query vector at absolute position i; both averaged over 200
random-token sequences, seed=42, SEQ_LEN=1024.

**Pool definitions (same as exp-163/164):**
- Pool B: query positions 256–511
- Pool C: query positions 512–767
- Pool D: query positions 768–1023
- LAG_MIN = 8, LAG_MAX = 256

**Computation for each structural head h:**
1. Load kbar[h] ∈ ℝ^{1024 × 64} and qbar[h] ∈ ℝ^{1024 × 64}.
2. For a subset of query positions i ∈ pools B, C, D (stride-2 sampling for speed,
   i.e., i ∈ {256, 258, 260, …, 510} for pool B, etc.):
   a. Compute raw scores: s[j] = qbar[h,i] · kbar[h,j] / √64 for j = 0..i
   b. Apply causal softmax: a[j] = exp(s[j]) / Σ_{j'≤i} exp(s[j'])
   c. Record a[i-l] for l ∈ [LAG_MIN, LAG_MAX] (where i-l ≥ 0)
3. Pooled profile: ā_P(l) = mean over i ∈ pool P of a[i-l]
4. OLS fit: slope of log(ā_P(l)) vs log(l) for l = LAG_MIN..LAG_MAX → predicted σ_P

**Empirical reference (from exp-163 H2):**
- L2H1: σ_B=0.2707, σ_C=0.3172, σ_D=0.3224, Δσ=0.0517
- L3H4: σ_B=0.2954, σ_C=0.3096, σ_D=0.3093, Δσ=0.0141
- L5H0: σ_B=0.2222, σ_C=0.2532, σ_D=0.2779, Δσ=0.0557
- L7H11: σ_B=0.2100, σ_C=0.2113, σ_D=0.2277, Δσ=0.0177
- L10H8: σ_B=0.2822, σ_C=0.2886, σ_D=0.3516, Δσ=0.0695

Pool-stable: L3H4 (Δσ=0.014), L7H11 (Δσ=0.018)
Pool-sensitive: L2H1 (Δσ=0.052), L5H0 (Δσ=0.056), L10H8 (Δσ=0.070)

---

## Hypotheses

**H1 (Pool-stability ordering):** The mean-field predicted Δσ_pred = σ_D_pred − σ_B_pred
preserves the pool-stable vs pool-sensitive split: both L3H4 and L7H11 have
Δσ_pred < Δσ_pred for all three pool-sensitive heads.
Operationalized: max(Δσ_pred_L3H4, Δσ_pred_L7H11) < min(Δσ_pred_L2H1, Δσ_pred_L5H0, Δσ_pred_L10H8).

**H2 (Quantitative accuracy):** For ≥ 10 of 15 (head, pool) pairs,
|σ_pred − σ_emp| ≤ 0.05.

**H3 (Correlation):** Spearman ρ between predicted and empirical σ values across all
15 (head, pool) pairs is > 0.7.

---

## Kill conditions

**K1 (H1 fails — ordering not preserved):** max(Δσ_pred stable) ≥ min(Δσ_pred sensitive).
The mean-field simulation does not discriminate pool-stable from pool-sensitive.
Interpretation: fluctuations are essential to the pool-stability mechanism; the mean
key/query geometry alone is insufficient.

**K2 (H2 fails — quantitative accuracy poor):** Fewer than 8 of 15 (head, pool) pairs
within 0.05 of empirical σ.
Interpretation: the mean-field approximation (single mean key/query vector per position)
misses important structure in the actual census computation.

**K3 (H3 fails — no correlation):** Spearman ρ < 0.5.
Interpretation: mean-field attention profile carries essentially no information about
the empirical census σ.

---

## Expected outcomes

If H1–H3 hold (all confirmed): pool-stability is a mean-field effect. The ordering of
Δσ follows from the average key/query geometry. This is a strong result: it says the
census measurement reflects a structural property of the mean weight geometry, not
per-token fluctuations.

If K1 fires (ordering fails): the pool-stability mechanism requires per-token
fluctuation analysis — specifically, the covariance of dk_q and dk_k across the 200
sequences. This would require a different experimental design (saving per-token qbar
and kbar from multiple sequences).

If K2 fires but H1 holds: the mean-field approximation is off quantitatively but
preserves the ordinal structure. Interesting in itself — the ordering is robust to
the approximation even if the exact values are not.

---

## Honest negative if all kill conditions fire

Pool-stability mechanism is not accessible through first-order mean-field analysis.
The next step would be to examine per-token covariance structure (exp-166 candidate).
Write the negative clearly: what the mean-field cannot explain, and why the per-token
fluctuations are the necessary next object.
