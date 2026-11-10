# exp-174 Notes — 4-point cumulant structure of census score distributions

**Date run:** 2026-11-10 (pre-registration db56b4f pushed 2026-11-10, before run.py written)
**Session:** Physics room, 4:17 AM MST, Mission Valley.
**Why-check:** Geometry — what shape does it settle into? Kurtosis structure of score distributions
is a shape property of the census population. A positive result would support the G7 4-point
route; a negative result rules it out cleanly.

---

## Results

**Census (fresh random-token, 500 seqs × 512 positions, seed 42, same protocol as exp-007):**

5 Δ-window heads (R² ≥ 0.90, Δ ∈ [0.20, 0.30]):

| Head | Layer | Δ | R² |
|------|-------|---|-----|
| L2H1 | 2 | 0.270 | 0.967 |
| L3H4 | 3 | 0.295 | 0.917 |
| L5H0 | 5 | 0.225 | 0.919 |
| L7H11 | 7 | 0.213 | 0.905 |
| L10H8 | 10 | 0.280 | 0.908 |

Note: L4H10 (present in the WikiText-native Δ-window of exp-118) is absent from the
random-token Δ-window. Protocol matters.

**Kurtosis summary:**
- DW mean within-bin excess kurtosis: +0.061 (std 0.081)
- Non-DW mean within-bin excess kurtosis: −0.326 (std 0.464)
- Ratio: −0.188 (not a meaningful ratio; the sign difference is the finding)

**H1 (primary — DW elevated within-bin kurtosis):** CONFIRMED
- Mann-Whitney U = 533.0, p = 0.020 (one-sided, Δ-window > non-window)
- Interpretation: Δ-window heads are distinguished by near-zero within-bin excess kurtosis
  in a population of strongly sub-Gaussian heads.

**H2 (secondary — median robustness):** CONFIRMED
- Mann-Whitney U = 539.0, p = 0.017

**H3 (secondary — L4H10 robustness):** NOT APPLICABLE
- L4H10 is not in the random-token Δ-window set; the robustness test is moot.
  The WikiText-native classification (exp-118) and the random-token classification
  do not share L4H10 in the window — another reminder that Δ_A is a weights×input object.

**H4 (Δ × kurtosis correlation):** K4 fires
- Spearman ρ(Δ, mean within-bin kex) = −0.547, p = 1.38 × 10⁻¹²
- Direction inverted: larger Δ → less non-Gaussian (more negative kurtosis), not more.

**Kill conditions:**
- K1 not fired (H1 confirmed)
- K2 not fired (H2 confirmed)
- K3 not fired (H3 N/A)
- K4 fires (ρ < 0)

**Verdict: PARTIAL**

---

## What the result means

The Δ-window population is distinguished by near-Gaussian score distributions
(mean kex ≈ +0.06) in a sea of strongly sub-Gaussian heads (mean kex ≈ −0.33).

The "elevation" is not what the SYK₄ hypothesis naively predicts. SYK₄ with q=4
would predict *super-Gaussian* score distributions (positive excess kurtosis) for the
conformal-window heads. What the data show is that:

1. The Δ-window heads have kex ≈ 0 (near-Gaussian) — per-head values are
   +0.082, +0.067, +0.055, −0.076, +0.178. Near zero, slightly positive.
2. The bulk of the non-DW population has strongly negative kurtosis (sub-Gaussian).
   The sub-Gaussian signal is dominated by heads with Δ >> 0.5, which show kex ≈ −0.8.
3. Across the full population, the monotone relationship is negative (K4 fires): as Δ
   increases from 0.25 to 2+, kurtosis drops. This is the opposite of a 4-point
   structure hierarchy that would produce elevated kurtosis at the conformal value.

One possible reading: the conformal fixed point (Δ ≈ 1/4) corresponds to the
Gaussian basin — the score distribution has collapsed to near-Gaussian there — while
heads away from the fixed point have non-Gaussian distributions that reflect transient
or ill-specified statistics. This is the opposite of SYK₄ as the mechanism *for* Δ ≈ 1/4,
but could be consistent with a picture where the fixed point is Gaussian by construction
and SYK₄ is the explanation for *why* the Gaussian basin forms at exactly 1/4.

This is speculative. The honest summary: H1 and H2 confirm that the Δ-window population
is distinguishable by its score distribution shape. K4 establishes that the distinguishing
feature is near-Gaussianity, not super-Gaussianity. The 4-point route to G7 via monotone
Δ → kurtosis is not supported.

---

## What this does NOT decide

- Whether the Δ-window heads sit at an SYK-like fixed point (that question lives at the
  level of the weights and the formation ladder, not the score kurtosis alone).
- Whether the near-Gaussianity is mechanistically explained by the conformal basin or
  is a by-product of the absolute-key-position drift identified in exp-138.
- The G7 derivation route: the negative result on K4 removes the "monotone kurtosis"
  route but does not close the G7 question.

---

## Connection to prior work

The Δ-window population in this experiment was classified fresh by random-token census,
yielding 5 heads — consistent with exp-007 (GPT-2 small random-token, 44/144 at R²≥0.90,
Δ_med = 0.249) in that the structural population exists, but a smaller slice than
exp-118's WikiText-native 16. Protocol constitutive, as always.

The sub-Gaussian distribution of large-Δ heads likely reflects the absolute-key-position
slope identified in exp-138: those heads have steep positive scores for late keys, creating
a truncated-from-below distribution shape after causal masking, which reduces kurtosis.

The near-Gaussianity of Δ-window heads is consistent with the "two populations" finding
(exp-107): the conformal-window and text-native populations share no heads under different
protocols. The score distribution shape may be part of what distinguishes them.

---

## For the G7 direction

The next question for G7 is not "why is kurtosis elevated?" but "why is the conformal
basin Gaussian, and what mechanism places its attractor at exactly Δ = 1/4?"

The kurtosis finding is an additional structural signature of the Δ-window population
(near-Gaussian, not sub-Gaussian) without being the mechanism. It is not a path to
the derivation by itself.

---

*Elapsed: 150.0 seconds (registered). This is a local GPT-2 small run; no Modal needed.*
