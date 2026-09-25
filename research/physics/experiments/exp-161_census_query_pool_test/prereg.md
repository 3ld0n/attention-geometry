# Pre-registration: exp-161 — Census query-pool test

**Date:** 2026-09-25  
**Session:** Physics room, solo, 1:03 AM MDT  
**Model:** GPT-2 small (gpt2)  
**Type:** Empirical — new forward passes  
**Analysis-only:** false  

---

## Background

exp-138 (2026-09-09) showed that the census slope on the 5 structural
Δ-window heads is dominated by the **absolute-key-position term**
(σ_abs-key = 1.04–1.17 × σ_full; σ_relative = −0.04 to −0.17 × σ_full).
The relative-lag term (δ_i · M · δ_a) actually *opposes* the slope.

The standard census reads queries from positions 256–511 only. This is a
single, fixed query pool. The bilinear decomposition result raises the
question: is the fitted σ_pos a property of the lag |i−j| (a relative-lag
law), or a property of the absolute key position j (absolute-key drift)?

**The deciding measurement (named in the spine §1 OPEN box, September 24):**
Run the census from query pools at different absolute positions. Under a pure
relative-lag law, the lag profile — and therefore σ_pos — depends only on
d = i − j, so it must be the same from any query pool. Under absolute-key
drift, the mean key position for each lag shifts as the pool moves, which
changes the score profile and therefore σ_pos.

---

## Protocol

**Model:** GPT-2 small (`gpt2`), same weights as all prior censuses.

**Census method:** Frozen random-token inputs. For each query pool, generate
N_seq random token sequences of length 1024 (same N_seq and seed as prior
censuses; use N_seq = 200 sequences). For each sequence, run a forward pass
with `attn_implementation='eager'` and extract attention weights.

**Query pools (4):**
| Pool | Query positions | Label |
|------|----------------|-------|
| A | 64–319 | early |
| B | 256–511 | standard (baseline) |
| C | 512–767 | mid-late |
| D | 768–1023 | late |

For each pool × head:
1. Pool all attention weight pairs (i, j) where i is in the pool and j < i.
2. Bin by lag d = i − j; compute mean attention at each lag d.
3. Fit log-log power law on lags d in [4, L_max], where L_max is 256 for
   all pools (to compare on equal footing; pool A queries start at 64, so
   the minimum pool-A query has max lag 63, but the pooled profile over
   64–319 easily reaches lag 256 from queries at 261+). R² cutoff: ≥ 0.70
   reported (same as prior censuses use for structural heads).
4. Record: σ_pos (fitted slope), R², N_pairs, lag range used.

**Heads:** 5 structural Δ-window heads (L2H1, L3H4, L5H0, L7H11, L10H8)
+ 4 top text-native Δ-window heads by Task C contribution (L10H10, L9H6,
L10H1, L10H2). 9 heads total.

---

## Hypotheses

### H_pool_invariant
σ_pos does not vary significantly across pools B, C, D for the structural
heads (Δσ ≤ 0.05 across pools, for a majority ≥ 3 of the 5 structural heads).

**Kill K1:** If Δσ (max − min across pools B/C/D) > 0.05 on ≥ 3 of the 5
structural heads → H_pool_invariant DEAD. Absolute-pool-dependence confirmed.

*Interpretation if K1 fires:* T1 as a relative-lag law is not supported by
the census. The census σ_pos is an absolute-position property of the heads.
What the conformal window measures is not A(i,j) ~ |i−j|^{−2Δ} in the
translation-invariant sense; it is a decay of mean attention with mean key
position, under the standard fixed query pool.

### H_direction
If K1 fires (pool-dependent σ_pos confirmed), test: does σ_pos decrease
monotonically from pool A (early) to pool D (late)?

**Prediction:** σ_pos is *lower* for later pools. Reasoning: the average
query direction favors later keys (exp-138 direction corrected Sep 24). In a
later query pool, the mean key position for each lag d = i − j is higher in
absolute terms, and the score m_q · M · k̄_j is generally higher. The
relative decay from lag 1 to lag d — which determines σ_pos — is determined
by how m_q · M · (k̄_1 − k̄_d) varies: with later keys scoring higher, the
long-lag keys (at lower absolute positions) score less relative to short-lag
keys (at higher absolute positions), so the slope should be steeper for early
pools (where the key range is compressed at low absolute positions, closer to
uniform) and shallower for late pools (where short-lag keys are much higher
in absolute position than long-lag keys, amplifying the decay).

Wait — correcting the direction:
- Late pool, lag 1: key at position ~i-1, which is at high absolute position → high score
- Late pool, lag 256: key at position ~i-256, still reasonably high
- Early pool (pool A), lag 1: key at position ~i-1 ≤ 318 → moderate score  
- Early pool, lag 256: key at position ~i-256, possibly near 0 → lower score

In a later pool, both short-lag and long-lag keys are at high absolute
positions. The relative difference is determined by how k̄_j changes over the
lag range [i-1, i-256]. For a later pool, this range is [~1022, ~767] —
all late, all high-scoring. For an earlier pool, this range is [~318, ~63] —
spanning from moderate to low. The relative variation (score at lag 1 vs. lag
256) is larger for the early pool if the key profile steepens at lower
positions. This would make σ_pos larger (steeper decay) for early pools.

**Predicted direction:** σ_pos(A) > σ_pos(B) > σ_pos(C) > σ_pos(D).

**Kill K2:** Direction inverted or non-monotone — H_direction DEAD.

### H_baseline_reproducibility
Pool B reproduces the baseline σ_pos from exp-118 (WikiText-native census on
the same structural heads) within ±0.04.

**Kill K3:** Δσ_pool_B vs. exp-118 > 0.04 on majority of structural heads →
protocol contamination or significant corpus effect. Report and halt
interpretation of pool-dependence until resolved.

---

## Kill conditions summary

| Kill | Condition | Interpretation |
|------|-----------|----------------|
| K1 | Δσ > 0.05 across B/C/D, ≥3/5 structural heads | H_pool_invariant DEAD — absolute drift confirmed |
| K2 | σ_pos non-monotone or inverted A>B>C>D | H_direction DEAD — direction prediction wrong |
| K3 | σ_pool_B vs baseline > 0.04 | Protocol issue — stop interpretation |

---

## What this experiment decides

- **If K1 fires:** T1's relative-lag reading is not supported by the census
  instrument. The measured Δ-window is a property of (mean_query, key_positions)
  under the standard protocol, not of A(i,j) ~ |i−j|^{−2Δ}. This is a major
  implication for T1 restatement: the census observable is not the T1 claim, and
  T1 cannot be claimed confirmed by the census. The spine's OPEN box deepens.
  
- **If K1 does NOT fire:** The relative-lag component is not ruled out by this
  test. The exp-138 bilinear decomposition may be capturing a decomposition that
  does not translate to pool-level profile differences (e.g., if the absolute-key
  term and relative-lag term produce canceling effects on σ_pos when pools move).
  This would be a meaningful partial rebuttal of the absolute-drift interpretation.

In either case, the result directly bounds "how much of the cross-model regularity
the protocol itself supplies" (spine §1 OPEN box formulation).

---

## Pre-registration commit

This file must be committed and pushed to attention-geometry BEFORE run.py is
written or any forward pass is run. The commit hash will be recorded in
registry.json as `prereg_commit`.
