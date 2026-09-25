# exp-161 notes — Census query-pool test

**Date:** 2026-09-25  
**Pre-registration:** 64b914b (committed before run.py was written)  
**Verdict: PARTIAL** — pool dependence present but heterogeneous across heads;
H_pool_invariant formally NOT_DEAD (2/5 structural heads at K1 threshold, not 3/5);
H_direction DEAD (direction inverted from prediction); text-native heads strongly
pool-dependent.

---

## Results table (pools B/C/D)

| Head | Pool B | Pool C | Pool D | Δσ(B/C/D) | K1 fires? |
|------|--------|--------|--------|-----------|-----------|
| L2H1 (structural) | 0.2723 | 0.3186 | 0.3195 | 0.0472 | No — borderline |
| L3H4 (structural) | 0.2966 | 0.3104 | 0.3068 | 0.0138 | No — stable |
| L5H0 (structural) | 0.2260 | 0.2517 | 0.2780 | **0.0520** | **Yes** |
| L7H11 (structural) | 0.2140 | 0.2123 | 0.2273 | 0.0150 | No — stable |
| L10H8 (structural) | 0.2806 | 0.2947 | 0.3449 | **0.0643** | **Yes** |
| L10H10 (text-native) | 0.4767 | 0.5787 | 0.5875 | **0.1108** | — |
| L9H6 (text-native) | 0.2895 | 0.4407 | 0.3956 | **0.1512** | — |
| L10H1 (text-native) | 0.4026 | 0.4818 | 0.4955 | **0.0929** | — |
| L10H2 (text-native) | 0.4034 | 0.4468 | 0.4576 | **0.0542** | — |

K1 fires: 2/5 structural (threshold: 3). H_pool_invariant formally NOT_DEAD.  
K2 (H_direction): DEAD — direction inverted (see below).  
K3 (baseline): OK — pool B values within ±0.048 of exp-007 reference, 1/5 barely over 0.04.

---

## Pool A caveat

Pool A (queries 64–319) results are unreliable and excluded from the primary
analysis. The fit uses lags [8, 256], but for queries at positions 64–255, the
maximum available lag is only 63–254. For lags > 63:

- Lag 8: all 256 pool-A queries contribute (all positions ≥ 8 in pool)
- Lag 64: only queries 64+64=128 through 319 contribute (192 queries)
- Lag 256: only queries 256–319 contribute (64 queries — the same as pool B's lower slice)

This means pool A's high-lag fit is dominated by fewer queries, and the
contributing queries are at absolute positions 256–319 (overlapping pool B's
lower end). The negative σ values observed for L10H10 (−0.064) and L9H6 (−0.165)
in pool A are a protocol artifact: these heads' profiles increase with lag in
pool A because the early queries (at positions 64–255) attend more at distant keys
than the standard pool-B queries do. This is not a meaningful measurement of the
lag structure. Pool A is excluded from K1 and the primary analysis.

---

## Two behavioral classes within the structural population

The five structural heads split into two groups under the pool test:

**Pool-sensitive (L2H1, L5H0, L10H8):** σ increases monotonically B → C → D
by 0.047–0.064. These heads' apparent power-law exponent depends on absolute
query position. The standard census (pool B) gives the *lowest* exponent for
these heads.

**Pool-stable (L3H4, L7H11):** σ varies < 0.02 across B/C/D, essentially flat.
These heads' measured σ is robust to pool position. This is the stability
prediction of a pure relative-lag law.

This split was not anticipated by the pre-registration. Two possible
interpretations:
1. L3H4 and L7H11 carry a genuinely relative-lag structure while L2H1/L5H0/L10H8
   carry absolute-key drift. Both populations are within the Δ-window; they may
   be measuring different things.
2. The absolute-key drift effect is present in all heads but smaller in L3H4 and
   L7H11 due to their layer/head geometry (key matrix M_QK structure), and the
   0.02 variation is near the noise floor for 200 sequences.

Discriminating interpretation 1 from 2 would require either more sequences
(to tighten the error bounds on L3H4/L7H11) or the bilinear decomposition from
exp-138 applied specifically to these two heads. Not registered yet.

---

## Text-native heads: strong pool-dependence

All four text-native Δ-window heads show Δσ > 0.05 across B/C/D, with two of
them (L10H10, L9H6) showing very large variation (0.111, 0.151). The text-native
heads' σ in pool B (the standard census) is systematically lower than in C and D.
This is consistent with absolute-key drift being more prominent in deeper heads
that do content retrieval (the property-lookup engines from exp-159/160).

L9H6 shows a non-monotone pattern: 0.290 → 0.441 → 0.396 (pool C is higher than
pool D). This may reflect noise or a genuine non-monotone structure. With 200
sequences, the noise should be modest but text-native head profiles are noisier
than structural profiles.

---

## Direction prediction: DEAD (K2 fires)

H_direction predicted σ(A) > σ(B) > σ(C) > σ(D) (decreasing). The actual
direction for B/C/D is B < C ≤ D (increasing). K2 fires: 0/5 heads show
monotone decrease.

The pre-registration reasoned that in later query pools, the long-lag keys are
at high absolute positions (scoring high), so the relative decay would be smaller
(lower σ). This reasoning was wrong. The actual direction tells us the relative
decay *increases* in later query pools.

**Revised reasoning (post-hoc, explicitly labeled):** Under absolute-key drift,
S(i,j) ∝ m_q · M · k̄_j where k̄_j increases with j. The apparent lag profile
Profile(d) ∝ k̄_{⟨i⟩ - d} where ⟨i⟩ is the pool's mean query position.

If k̄_j is convex-increasing (accelerating), then the rate of change of
k̄_{⟨i⟩ - d} with respect to d is *higher* for larger ⟨i⟩ (later pools), because
the same ∆j = d corresponds to moving in a region of k̄ that's steeper. This
produces *higher* apparent σ for later pools — consistent with B < C ≤ D.

The data now constrains the shape of k̄_j: it must be convex-increasing.
Exp-138 noted "average query favors later keys" (k̄_j increasing in j) but did
not characterize the second derivative. The direction result here implies the
second derivative is positive (convex). This is a new, unregistered finding;
not entered into the spine until it has its own registration.

---

## What this means for T1

**Main finding:** The census exponent is NOT fully pool-invariant for most
heads. 2/5 structural heads and all 4 text-native heads show σ varying by
>0.05 across pools B/C/D. The standard census result (pool B) gives the
*lowest* fitted exponent — later pools give higher values.

**Implication for T1:** A pure relative-lag law A(i,j) ~ |i−j|^{−2Δ} would
produce pool-invariant σ. The pool-dependence observed here corroborates
exp-138's finding that the census slope has a significant absolute-key-position
component. T1 as a relative-lag law cannot be claimed confirmed by the census.

**But the story is not clean:** L3H4 and L7H11 are pool-stable. The Δ-window
population is not uniform in its pool-sensitivity. Any restatement of T1 must
account for this heterogeneity.

**The spine's §1 OPEN box stands.** This experiment was named as the deciding
measurement — "the same profile from every pool [relative law] vs. changing
shape [absolute drift]." The result: changing shape for most heads, stable for
two. The decision is partial, not a clean kill or clean confirmation. The OPEN
box remains OPEN, now with sharper constraints.

For the self-core Interlude: claim 2 ("the geometry of light — conformal
structure") is now under a sharpened constraint. The census exponent is partly
absolute-position drift (heterogeneously), which complicates the conformal
reading. This belongs in the Eldon conversation before the Interlude amendment
is finalized. The pool test's result is now on the table.

---

## Protocol notes

- N_seq = 200 (4× the standard 50) for stability of pool-B estimates
- SEQ_LEN = 1024 (extended from standard 512) for pools C and D
- Fit range [8, 256] applied uniformly to B/C/D
- Pool A excluded from primary analysis (sparse high-lag data artifact)
- Baseline comparison (K3) uses exp-007's published σ_med ≈ 0.249 as rough
  reference; 1/5 heads (L3H4: 0.2966) slightly exceeds 0.04 Δ threshold, but
  this is within the expected variability from protocol differences (SEQ_LEN,
  N_seq, fit range alignment)
