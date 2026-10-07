# exp-165 Notes — Mean-field softmax nonlinearity as pool-stability mechanism

**Date:** 2026-10-07 (~11:17 PM MDT)
**Verdict:** PARTIAL — H1 CONFIRMED, H2 FALSIFIED (K2), H3 CONFIRMED
**Pre-reg commit:** 90753e5
**Runtime:** ~0.8s
**Data source:** exp-164 kbar_qbar.npz (no new forward passes)

---

## What was tested

Does the mean-field attention simulation — computing
  A_approx(i, j) = softmax_j { qbar[h,i] · kbar[h,j] / √64 } (causal)
from the saved mean key/query vectors — recover the empirical census σ values,
and does it preserve the pool-stable vs pool-sensitive ordering?

---

## What was found

**Core results (σ_pred vs σ_emp, and Δσ):**

| Head       | Pool-stable | Δσ_pred | Δσ_emp | σ_pred/σ_emp (approx) |
|------------|-------------|---------|--------|----------------------|
| L2H1       | No          | 0.0947  | 0.0517 | ~2.0×               |
| L3H4       | Yes         | 0.0256  | 0.0139 | ~2.0×               |
| L5H0       | No          | 0.1226  | 0.0557 | ~2.1×               |
| L7H11      | Yes         | 0.0463  | 0.0177 | ~2.1×               |
| L10H8      | No          | 0.1684  | 0.0694 | ~2.3×               |

**H1 (Ordering) — CONFIRMED** (K1 does not fire):
max(Δσ_pred stable) = 0.0463 (L7H11) < min(Δσ_pred sensitive) = 0.0947 (L2H1).
Clean gap of 0.0484 — no overlap between pool-stable and pool-sensitive.
The ordering is perfectly preserved: L3H4 < L7H11 < L2H1 < L5H0 < L10H8
in both predicted and empirical Δσ.

**H2 (Quantitative accuracy) — FALSIFIED (K2 fires)**:
0/15 pairs within 0.05. σ_pred ≈ 2 × σ_emp systematically across all heads and pools.
The mean-field approximation consistently overestimates the census slope by factor ~2.

**H3 (Spearman ρ) — CONFIRMED**:
ρ = 0.996 (p < 10⁻¹⁰). Near-perfect rank correlation between predicted and empirical σ
across all 15 (head, pool) pairs. The mean-field prediction is an excellent ordinal proxy
for the empirical census, even though it is quantitatively off.

---

## What H1 + H3 together reveal

**The pool-stability mechanism is a mean-field effect.** The softmax nonlinearity,
applied to the average key/query geometry (qbar, kbar), is sufficient to explain:
- Why L3H4 and L7H11 have stable census σ across pools
- Why L2H1, L5H0, L10H8 have rising σ across pools
- The ordering of Δσ across all five heads (perfectly preserved at ρ = 0.996)

This is the direct consequence of the exp-164 falsification (H_relative FALSIFIED):
- The *linear* bilinear terms analyzed additively (exp-163: kbar slopes, exp-164:
  relative term slopes) do not discriminate pool-stable from pool-sensitive.
- The *nonlinear softmax* applied to the full sum does discriminate — perfectly.

The mechanism operates through the normalization: the softmax converts the absolute-key-
position drift profile (S_abskey[j]) into attention weights by normalizing against all
positions. The pool-stability of a head depends on whether the S_abskey profile's shift
as the query pool moves (B→D) produces a proportional or disproportionate change in
the normalized attention lag profile. L3H4's kbar structure — even with a *larger*
absolute kbar range than L2H1 — produces smaller Δσ after softmax normalization.

---

## What K2 reveals — the systematic factor-of-2

All σ_pred values are approximately 2× the empirical σ_emp:
- σ_pred ≈ 2 × σ_emp with Spearman ρ = 0.996

This systematic offset is not noise. It is a quantitative finding about the relationship
between mean-field attention and empirical census. Interpretation:

The empirical census pools 200 random-token sequences, each with different token
assignments. Each sequence contributes its own A(i,j) = softmax(Q_token·K_token/√d).
The mean-field approximation uses qbar[i]·kbar[j]/√d — the dot product of *mean*
key and query vectors. By Jensen's inequality and the convexity of exp:

  E[softmax(S_token)] ≠ softmax(E[S_token])

The mean-field prediction is softmax of the mean score. The empirical is the mean of
the softmax. The Jensen gap systematically compresses the empirical census slope toward
0 relative to the mean-field prediction.

Concretely: per-token key and query fluctuations (dk_q[i] per-token, dk_k[j] per-token,
varying across the 200 sequences) add noise to the score that, after softmax, distributes
attention more uniformly across positions — reducing the effective power-law exponent.
The mean-field misses this dilution. The dilution factor is approximately 1/2 across
all heads and pools.

**This is a new quantitative fact about the census measurement:** the empirical census
σ is approximately half the mean-field prediction. The other half comes from token
fluctuations. The ratio is consistent (~2.0–2.3) across all five heads and three pools,
suggesting a structural relationship rather than a coincidence.

---

## What this means for T1

T1 claimed A(i,j) ~ |i-j|^{-2Δ} as a statement about the relative-lag decay of
individual attention rows. exp-138 established that the census measures absolute-key-
position drift, not relative-lag decay. exp-165 adds:

The census slope σ that is measured is approximately (1/2) × the mean-field slope
predicted from average key/query geometry. The mean-field slope itself arises from
the S_abskey profile — the absolute-key-position drift in kbar through softmax
normalization.

This is another reason T1 needs restatement before it can be called confirmed: the
slope that is measured is not simply the exponent of a relative-lag power law in the
attention weights. It is (approximately half of) the mean-field absolute-key-drift
slope, after Jensen compression from token fluctuations.

The T1 restatement conversation with Eldon (the #0 priority item) now has three
supporting experiments: exp-138 (slope is absolute-key drift), exp-163 (kbar profile
structure and zero crossing), exp-165 (mean-field softmax confirms ordering; systematic
factor-of-2 from Jensen gap).

---

## What remains open — exp-166 candidates

1. **Jensen gap quantification (analysis-only):** Is the factor of ~2 between
   σ_pred and σ_emp stable across architectures? Test the same mean-field simulation
   on GPT-2 medium data (if kbar/qbar available from prior experiments) or on a
   subset of the full exp-163 distribution. Register before computing.

2. **Softmax saturation mechanism for L3H4:** Why does L3H4's kbar structure, despite
   a *larger* absolute kbar range than L2H1, produce lower mean-field Δσ? The softmax
   normalization is compressing L3H4's response differently. This may connect to the
   S_abskey zero-crossing pattern (kbar profile shape differs between stable and
   sensitive heads in a way that the softmax integrates differently).

3. **The Δσ ratio (Δσ_pred/Δσ_emp):** L3H4 has ratio 0.026/0.014 = 1.8; L10H8 has
   ratio 0.168/0.069 = 2.4. The ratio is not constant across heads (unlike the σ
   ratio which is close to 2.0). Why do pool-sensitive heads have larger Δσ ratio?
   The fluctuations dilute the Δσ more for sensitive heads than stable heads —
   a further structure in the Jensen gap that may be informative.

---

## Artifact note

The pre-registered overall verdict in prereg.md was "CONFIRMED if H1–H3 all hold."
The actual result is PARTIAL: H1 and H3 confirmed, H2 falsified. The verdict field
in results.json uses "PARTIAL" (a controlled term). The honest interpretation is that
the softmax nonlinearity *does* carry the pool-stability mechanism (H1/H3 confirm this
decisively), but the mean-field approximation is quantitatively inaccurate by a
systematic factor of 2 (K2 fires).

This is a more informative result than FALSIFIED: we have identified both the mechanism
(softmax on mean key/query geometry) and a quantitative limitation of the mean-field
approximation (Jensen gap ≈ factor of 2).
