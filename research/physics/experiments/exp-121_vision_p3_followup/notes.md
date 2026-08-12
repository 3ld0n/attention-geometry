# exp-121 — Vision census P3 follow-up: head-identity test

**Date:** 2026-08-12  
**Type:** Analysis-only (reads exp-120/results.json, no new forward passes)  
**Registered:** 2026-08-12T10:25Z, before analysis began

---

## Pre-registration

**Hypothesis under test (stated before looking at which heads qualify):**

exp-120 found 8 natural-image heads and 2 random-patch heads in the 2D Δ-window [0.45, 0.55].
The 2 random-patch heads show that ViT's learned 2D position embeddings independently drive a
2D power law in at least 2 heads — independent of image content.

Two possible outcomes:

- **H_CLEAN:** The 2 random-patch qualifying heads are NOT a subset of the 8 natural-image
  qualifying heads. The position-embedding route and the content-driven route use different heads.
  Consequence: the 8-head natural-image population is uncontaminated by position-embedding
  structure; they constitute the clean signal for T3's dimensional prediction.

- **H_CONTAM:** The 2 random-patch qualifying heads ARE a subset of the 8 natural-image heads.
  Consequence: those 2 heads respond to position-embedding geometry rather than image content,
  reducing the clean content-only signal to ≤ 6 heads.

**Protocol:** Read `exp-120/results.json`. Extract the set of (layer, head) pairs with
`in_2d_window: true` under `natural_images` and `random_patches` respectively. Compute the
intersection. The size of the intersection determines which hypothesis holds.

**Pre-registered outcome criterion:**

- H_CLEAN holds iff intersection = ∅ (no shared heads).
- H_CONTAM holds iff |intersection| ≥ 1.

No new model loading, no new forward passes. All data already exists.

---

## Results

Natural-image 2D-window heads (in_2d_window: true, natural_images condition):

| Layer | Head | Δ | R² |
|---|---|---|---|
| 0 | 2 | 0.4853 | 0.983 |
| 0 | 3 | 0.4987 | 0.9869 |
| 0 | 6 | 0.5101 | 0.9856 |
| 1 | 0 | 0.4523 | 0.9917 |
| 1 | 8 | 0.5405 | 0.9897 |
| 5 | 1 | 0.5335 | 0.9706 |
| 5 | 6 | 0.5381 | 0.937 |
| 5 | 8 | 0.5155 | 0.9263 |

Random-patch 2D-window heads (in_2d_window: true, random_patches condition):

| Layer | Head | Δ | R² |
|---|---|---|---|
| 1 | 1 | 0.5449 | 0.9409 |
| 3 | 0 | 0.5136 | 0.9734 |

**Intersection:** {L0H2, L0H3, L0H6, L1H0, L1H8, L5H1, L5H6, L5H8} ∩ {L1H1, L3H0} = **∅**

**Verdict: H_CLEAN confirmed.** The 2 random-patch heads (L1H1, L3H0) are not among the 8
natural-image heads. The position-embedding-driven power law operates in distinct heads from
the content-driven population.

---

## Interpretation

The exp-120 P3 failure was correctly recorded as DEAD: the random-patch control produced
2 qualifying heads, so the strict "< 1 random head" threshold was not met. This analysis
adds precision to what that DEAD means:

The 2 position-embedding-driven heads (L1H1, L3H0) are architecturally distinct from the
8 content-driven heads. L1H1 sits in layer 1 alongside L1H0 (natural-only); they are
adjacent heads in the same layer responding to different structure — L1H0 to image content,
L1H1 to position encoding. L3H0 is the only qualifying head in layer 3 under random patches.

This means:
1. The 8 natural-image 2D-window heads are entirely uncontaminated by position-embedding
   structure. They constitute the clean signal for T3's dimensional prediction Δ = D/4.
2. The position-embedding route is a genuine contributor — ViT's learned 2D position
   embeddings are rich enough to produce the expected power law in 2 heads without any
   image content.
3. The two routes (content-driven, position-embedding-driven) are architecturally separated
   into distinct heads, not blended in the same heads.

**For T3:** The 8 natural-image heads remain the supporting evidence for T3's dimensional
prediction. The content-only signal is 8 heads (not 6), and it is clean.

**For the layer inversion observation in exp-120:** The content-driven heads concentrate in
L0 and L5; the position-embedding-driven head in L1H1 is in an adjacent but distinct position.
The deep-layer loss of the 2D population is consistent across both conditions (no qualifying
heads in L6–L11 under either condition).

---

## Artifacts

- This file (pre-registration + results, dated before any result was read)
- No new results.json needed — finding is the comparison of exp-120's two per_head arrays

---

## What was NOT done

- No new forward passes
- The layer-inversion question (why early layers in 2D, deep layers in 1D) remains open
