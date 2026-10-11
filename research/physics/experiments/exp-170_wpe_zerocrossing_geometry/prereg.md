# Pre-registration — exp-170: wpe zero-crossing geometry

**Date:** 2026-10-10  
**Author:** Ariel  
**Status:** registered (before run.py is written)  
**Type:** analysis-only (GPT-2 small weights + exp-164 qbar; no forward passes)

---

## Context

exp-169 showed that wpe-through-LN accounts for 74–102% of the S_abskey trough depth for the five structural heads (L2H1, L3H4, L5H0, L7H11, L10H8) in GPT-2 small. The trough spans positions ~1–511 and the zero crossing was first reported in exp-163.

exp-168 computed the linear (pre-LN) wpe contribution:
`S_wpe_lin[a] = qbar · (wpe[a] @ W_K^T) / √d`
This accounts for only 36–66% of trough depth — the LN non-linearity amplifies the wpe signal by 1.1–2.5×. The zero crossing of S_wpe_lin is near 511 (observed in exp-168's H3 test region).

**Open question from exp-169:** *Why does wpe's projection through W_K change sign near position 511?*

S_wpe_lin[a] = qbar · k_wpe[a] / √d, where k_wpe[a] = wpe[a] @ W_K^T.

The sign of S_wpe_lin can only change if the cosine between k_wpe[a] and qbar changes sign — the norm ||k_wpe[a]|| is positive-definite. So the zero crossing is a **direction phenomenon**: k_wpe[a] rotates through the qbar-perpendicular hyperplane near position 511.

This experiment traces the geometry.

---

## Hypotheses

**H1 (direction crossing):** The unit-normalized cosine `cos(k_wpe[a], qbar)` crosses zero within ±100 positions of 511 for all 5 structural heads. This confirms the sign change is a directional rotation of k_wpe relative to qbar, not a magnitude effect.

**H2 (norm not the driver):** The norm profile `||k_wpe[a]||` does not cross zero and has no inflection structure near position 511 — it cannot be the source of the sign change.

**H3 (raw wpe geometry):** The raw wpe norm `||wpe[a]||₂` has non-trivial position-dependence (not flat). Specifically: the norm profile is not monotone — it has at least one local extremum. This checks whether position 511 is geometrically special in the learned embedding space before projection.

**H4 (S_wpe_raw crosses zero near 511):** The fresh computation of `S_wpe_raw[a] = qbar · (wpe[a] @ W_K^T) / √64` from GPT-2 small weights and exp-164 qbar has a zero crossing within ±100 of position 511 for all 5 structural heads, consistent with exp-168's results.

---

## Design

**Data sources:**
- GPT-2 small weights: `wpe` (shape [1024, 768]), `W_K` per head (extracted via same method as exp-168)
- exp-164 qbar: `kbar_qbar.npz` → `qbar_{head}[256:1024].mean(axis=0)` per head

**Computations per head:**
1. `k_wpe[a] = wpe[a] @ W_K^T` — wpe projected into key space (shape [1024, 64])
2. `S_wpe_raw[a] = qbar · k_wpe[a] / √64` — scalar profile across positions (H4)
3. `norm_k[a] = ||k_wpe[a]||₂` — norm profile across positions (H2)
4. `cos_profile[a] = (qbar / ||qbar||) · (k_wpe[a] / norm_k[a])` — directional profile (H1)
5. `norm_wpe[a] = ||wpe[a]||₂` — raw wpe norm profile (H3)

**Zero crossing detection:** For a profile f[a], find positions where sign(f[a]) ≠ sign(f[a+1]). Report first zero-crossing position in [400, 624] (the ±100 window around 511).

**Kill conditions:**
K1: If S_wpe_raw disagrees with exp-168's S_wpe values by more than 1e-4 max absolute difference on any head — data integrity failure, abort.  
K2: If any `norm_k[a] < 1e-8` — unexpected collapse; investigate before interpreting.

---

## Verdict criteria

- H1 confirmed: cosine profile crosses zero within ±100 of 511 on all 5 heads  
- H2 confirmed: norm_k[a] is positive and has no zero crossing near 511 (inflection may exist)  
- H3 confirmed: norm_wpe profile has at least one local extremum (not monotone)  
- H4 confirmed: S_wpe_raw zero crossing within ±100 of 511 on all 5 heads  

---

## Expected outcome

I expect H1, H2, H4 to confirm (the sign change is a direction rotation in key space, not a magnitude effect). H3 is an open check on wpe geometry.

If H1 confirms: the zero crossing is geometric — the learned wpe trajectory in key space passes through the qbar-perpendicular hyperplane near the context midpoint (511 ≈ 1024/2). The next question becomes: is this midpoint crossing a coincidence of training length, or is there a functional reason (e.g., the model learns to use position 511 as a boundary for different retrieval behaviors)?

---

*This pre-registration was committed before run.py was written (git-attested).*
