# exp-164 Pre-Registration
**Title:** L3H4 pool-stability — bilinear score decomposition and relative term analysis

**Date:** 2026-09-30  
**Pre-registered:** before run.py was written (commit pushed to attention-geometry)  
**Model:** GPT-2 small (gpt2), learned positional encoding  
**Protocol:** Same random-token census as exp-163 (seed=42, 200 seqs, SEQ_LEN=1024)  
**Analysis type:** New forward passes — saves full kbar[h, a] and qbar[h, i] arrays as npz

---

## The paradox this addresses

exp-163 established:
- L3H4 kbar cross-pool slope range = **5.44** (larger than pool-sensitive L2H1 = 5.20)
- L3H4 census Δσ = **0.014** (much smaller than L2H1 = 0.052)
- L3H4 σ_C = 0.3096, σ_D = 0.3093 — essentially flat from pool C to D
- L3H4 kbar pool slopes: B=1.85, C=4.64, D=7.30 — accelerating C→D

S_abskey alone predicts L3H4 should have MORE σ variation across pools, not less.
Something in the bilinear score decomposition is offsetting the kbar effect.

---

## Background: bilinear score decomposition (from exp-138)

The inner product score at query i, key j decomposes as:

    S(i,j) = (m_q + dk_q[i]) · (m_k + dk_k[j]) / √d
           = const          (m_q·m_k — absorbed by softmax)
           + S_abskey[j]    (m_q·dk_k[j]/√d — key-position drift, exp-163)
           + S_absquery[i]  (dk_q[i]·m_k/√d — query-position drift, absorbed by softmax)
           + S_relative[i,j](dk_q[i]·dk_k[j]/√d — query×key interaction)

S_absquery[i] is constant across key positions for a fixed query — it shifts all scores
by the same amount and is absorbed by the softmax. It does NOT affect A(i,j).

The candidates that can affect the census lag-profile:
1. S_abskey[j] — measured in exp-163 (kbar-derived)
2. S_relative[i,j] = dk_q[i]·dk_k[j]/√d — NOT yet measured; requires both kbar and qbar

---

## Hypothesis

**H_relative:** L3H4's pool-stability arises from the relative interaction term
S_relative[i,j]. Specifically:

For pool D queries (i ∈ 768–1023) at short lags (l = 8–50):
    E_{i∈D}[dk_q[i]·dk_k[i-l]/√d] < E_{i∈B}[dk_q[i]·dk_k[i-l]/√d]

The mean relative term decreases from pool B to D for L3H4 at short lags —
offsetting the S_abskey increase (which would push σ up for pool D).

For L2H1, no such decrease occurs — the relative term is approximately flat or
increases from pool B to D, so the kbar effect is unopposed, giving larger Δσ.

**Operationalization:**
- For each head h and pool P ∈ {B, C, D}, compute:
    rel_profile_P(l) = E_{i ∈ P}[S_relative(i, i-l)]
                     = E_{i ∈ P}[dk_q[i] · dk_k[i-l] / √d]
  for l = LAG_MIN..LAG_MAX (8..256)
- Compute rel_slope_P = OLS slope of rel_profile_P(l) vs log(l)
  (positive slope = relative term adds to lag-concentration; negative = subtracts)
- Compute rel_cross_pool_range = max(rel_slope_B, C, D) - min(rel_slope_B, C, D)
- Compare L3H4 vs L2H1 on: sign of (rel_slope_D - rel_slope_B) and magnitude of
  rel_cross_pool_range

**Secondary check (total decomposition):**
- Compute total predicted σ from (S_abskey + S_relative), pooled:
    OLS slope of log(E[exp(S_abskey[j] + S_relative[i,j])]) vs log(lag)
  for each pool and head — compare with empirical census σ from exp-163 H2
- If H_relative holds: the total predicted σ variation should be smaller for L3H4
  than kbar-only predicted, and closer to the empirical Δσ

---

## Kill conditions

**K1 (H_relative fails — no cross-pool decrease):** For L3H4, rel_slope_D ≥ rel_slope_B
(the relative term does NOT decrease from pool B to pool D at short lags). If K1 fires,
H_relative is falsified; pool-stability mechanism remains open.

**K2 (no discrimination):** L3H4 and L2H1 have the same sign of
(rel_slope_D - rel_slope_B) — the relative term does not discriminate the two heads.
If K2 fires, H_relative fails to explain the stable/sensitive split even if present.

**K3 (total decomposition inadequate):** The total predicted σ from (abskey + relative)
does not match empirical σ within 0.02 for either head — the bilinear decomposition
misses important structure (e.g., relative term after softmax is not additive).

---

## Protocol

1. Load GPT-2 small (eager attention, float32)
2. Run same 200 random-token sequences at SEQ_LEN=1024, seed=42 (same as exp-163)
3. Hook c_attn to collect Q and K for all 5 structural heads at every position
4. Accumulate kbar[h, a] and qbar[h, i] (mean over 200 sequences)
5. Save raw arrays as `kbar_qbar.npz`
6. Compute dk_k[h, a] = kbar[h, a] - mean_a(kbar[h, a]) — mean over positions 256–1023
7. Compute dk_q[h, i] = qbar[h, i] - mean_i(qbar[h, i]) — mean over positions 256–1023
8. For each head h and pool P ∈ {B, C, D}:
   a. rel_profile_P(l) = mean over i∈P of (dk_q[h,i] @ dk_k[h, i-l] / √d) for l=8..256
   b. OLS slope of rel_profile_P(l) vs log(l) → rel_slope_P[h]
9. Compare L3H4 vs L2H1 on (rel_slope_D - rel_slope_B) and cross-pool range
10. Secondary: compute exp(S_abskey[j] + S_relative[i,j]) for sampled (i,j) pairs and
    estimate the "total" predicted attention-lag profile; compare with empirical σ

---

## Expected outcome if H_relative holds

L3H4: rel_slope_D < rel_slope_B (negative cross-pool slope shift for relative term)
L3H4: total predicted Δσ (abskey + relative) close to empirical 0.014
L2H1: rel_slope_D ≈ rel_slope_B (flat or increasing cross-pool)
L2H1: total predicted Δσ close to empirical 0.052

The interpretation: L3H4's query vectors at large positions (pool D) have
dk_q[i] that is anti-correlated with dk_k[j] for j near i — the relative term
SUBTRACTS from short-lag attention, opposing the S_abskey effect.

## Honest negative if H_relative fails

If K1 fires: pool-stability has a different explanation. Candidates:
- Softmax nonlinearity: the S_abskey profile shape for L3H4 vs L2H1 produces
  different sensitivity curves (the attention-to-S_abskey relationship is nonlinear)
- Architectural effects: L3H4 is in layer 3 vs L2H1 in layer 2; the residual stream
  at layer 3 is more processed, potentially with different query/key structure
In either case, write the negative clearly and name the next hypothesis.
