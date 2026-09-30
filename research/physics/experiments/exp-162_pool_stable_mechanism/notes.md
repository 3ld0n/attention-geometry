# exp-162 — Notes: Pool-stable head mechanism — absolute-key profile curvature

**Date:** 2026-09-30 (02:45 UTC / 2026-09-29 20:45 MDT)
**Ariel — Physics room, autumn evening, Mission Valley.**

---

## Summary

**INCONCLUSIVE — operationalization failure.** The pre-registered hypothesis (H1: pool-stable heads have lower slope curvature of S_abskey[a] across key-position sub-ranges) could not be tested as operationalized, because S_abskey[a] is **oscillatory** in absolute-position space — not the monotone power-law decay the operationalization assumed. K1 and K2 did not technically fire (NaN vs NaN), but the test was not executed in any meaningful sense.

No post-hoc property from exp-138 data correlates with pool-sensitivity (all |ρ| ≤ 0.4, all p > 0.5). The mechanism behind L3H4/L7H11 pool-stability cannot be determined from existing data.

---

## What was found

### The S_abskey profile shape

S_abskey[a] = m_q · dk[a] / √d_head (where dk[a] = kbar[a] - mean(kbar)) is an oscillatory function of absolute key position a:

| Position | L2H1 | L3H4 | L5H0 | L7H11 | L10H8 |
|----------|------|------|------|-------|-------|
| 0 | +2.12 | +4.13 | +4.32 | +3.22 | +3.86 |
| 1 | negative for all heads (zero crossing at position 1) |
| 127 | −0.99 | −1.15 | −0.91 | −0.79 | −1.08 |
| 255 | +0.10 | −0.16 | −0.12 | −0.05 | −0.18 |
| 383 | +0.89 | +0.87 | +0.87 | +0.84 | +0.97 |
| 511 | +1.51 | +2.21 | +1.98 | +1.59 | +1.94 |

The function is strongly positive at position 0 (GPT-2's wpe[0] has a distinctive structure), sharply negative by position 1, near zero around position 255, then positive again from ~300 onward through 511.

**This is not a power-law of absolute key position.** The pre-registration assumed a monotone decay and proposed a slope-curvature measure; that measure is undefined when S_abskey is negative over most of the range being analyzed.

### Post-hoc: exp-138 properties don't predict pool-sensitivity

Spearman ρ between each exp-138 property and exp-161 Δσ (pool-sensitivity):

| Property | ρ | p |
|---|---|---|
| σ_abskey | −0.10 | 0.87 |
| σ_relative | −0.40 | 0.51 |
| r2_relative | +0.10 | 0.87 |
| ratio_abskey_to_full | −0.30 | 0.62 |
| S_abskey[0] | +0.10 | 0.87 |

All n.s. (n=5, power is weak, but the absence of any monotone relationship is informative). The mechanism cannot be diagnosed from exp-112/138 data.

### Why the operationalization failed (and what it reveals)

The exp-138 abskey_profile_at_lags (which showed clean monotone positive-to-negative decay) is a **LAG profile** — for each lag dx, it averages S_abskey[a] over key positions a = i − dx where i runs over the query pool. This averaging produces a clean function of dx despite S_abskey(a) being oscillatory.

Crucially: exp-138's pool covered queries at positions [DEEP_LO=256, 511]. For that pool, the key positions a = i − dx ranged over [0, 503]. The average of S_abskey(a) over that range, as a function of dx, produced the clean lag profile.

For Pool C [512, 767] of exp-161, keys would range over [256, 759]. For Pool D [768, 1023], keys range over [512, 1015]. These key positions are partially or entirely outside the exp-112 data range [0, 511]. The direct S_abskey values at positions 512–1023 are unknown.

**The pool-stability question is inherently about positions > 511 that weren't measured in exp-112.**

---

## What this means for the T1 conversation

The pool-sensitivity finding (exp-161) and this analysis together establish:
1. The census slope is absolute-key drift for all structural heads (exp-138).
2. The absolute-key profile S_abskey(a) is oscillatory in position space — not a clean power law of a.
3. The lag profile of S_abskey (averaged over queries) is approximately a power law of lag, which is what σ_abskey measures.
4. Pool-sensitivity depends on how much the lag profile of S_abskey changes when key positions shift to a higher range. This requires data beyond position 511.

For the T1 restatement: σ_pos measures a slope of a lag profile that is driven by absolute-key drift. But the "stability" of that slope across pools depends on whether the absolute-key profile at higher positions (512–1023) has the same slope as at lower positions (0–511). Two structural heads (L3H4, L7H11) appear to maintain a stable slope even when the key-position range shifts — explaining pool-stability — but the mechanism requires direct measurement.

---

## Next experiment (proposed)

**exp-163 — Absolute-key profile at SEQ_LEN=1024**

Run forward passes at SEQ_LEN=1024 (same protocol as exp-112 but extended) to collect kbar for all positions [0, 1023]. Then:
1. Compute S_abskey[a] for a ∈ [0, 1023].
2. Compute the lag profile of S_abskey for each of the three exp-161 query pools (B: 256–511, C: 512–767, D: 768–1023).
3. Fit σ for each pool's lag profile.
4. Compare σ_B, σ_C, σ_D for pool-stable vs pool-sensitive heads.

Pre-registration before any code: the absolute-key lag profile should be pool-invariant for L3H4/L7H11 and pool-variant for L2H1/L5H0/L10H8.

This is a lightweight experiment — one forward pass at SEQ_LEN=1024 with the same random-token protocol as exp-112. Expected runtime: ~1 min. However it requires new forward passes (not analysis-only), so it needs its own registry entry, pre-registration, and commit before the run.

---

## Register discipline note

The pre-registration (commit 4b01dab) committed the hypothesis before any analysis code was written. The result is an operationalization failure — the assumed shape of S_abskey was wrong. This is a correctly-formatted honest negative: the experiment ran, the pre-registered test was undefined (NaN), and the failure reveals something real about the structure of S_abskey that wasn't known before.

The difference from a "confabulated clean result": the failure mode was found by running the analysis, not by suppressing it. The exp-163 proposal is the correct path forward.
