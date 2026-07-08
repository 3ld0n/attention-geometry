# exp-082 — Whirlpool/Crystal Robustness: MAX_DX=256 (GPT-2)

*2026-07-06. Analysis-only; GPT-2 cached (no download needed).*
*Direct follow-up to exp-081. Pre-registration committed before running.*

---

## Motivation

exp-081 (2026-07-04) confirmed H_whirlpool narrowly (REAL=10, RAND=4, Δn=+6 > threshold +5)
but noted a critical protocol caveat: the absolute RAND count (4/144) was far below the
exp-007 RAND baseline (44/144), suggesting MAX_DX=64 might be too short.

The queue named this the required follow-up: "re-run exp-081 protocol with larger MAX_DX
(128 or 256) to see if the REAL>RAND n_syk_near finding holds at the exp-007 lag range."

This experiment answers that question.

**Protocol diagnostic note (identified before running):** exp-007 used `cutoff_low=3`,
excluding dx=1 and dx=2 from the fit (to avoid attention-sink contamination at near lags).
exp-081 starts from dx=1. This protocol difference is a plausible contributor to the RAND
count discrepancy (44/144 vs 4/144). However, this experiment keeps exp-081's protocol
(starting from dx=1) to ensure clean comparability with exp-081 — isolating MAX_DX as the
single variable changed. The cutoff_low effect is documented but not adjusted here.

---

## Pre-stated Hypotheses (COMMITTED BEFORE RUNNING)

**H_robust:** The whirlpool finding from exp-081 replicates at MAX_DX=256.
- n_syk_near(REAL) > n_syk_near(RAND) + 5 (same threshold as exp-081)
- Interpretation: the REAL>RAND signal is protocol-robust, not a MAX_DX artifact.

**H_artifact:** The finding collapses or reverses at MAX_DX=256.
- Δn_syk ≤ 0 (REAL ≤ RAND) OR the threshold is not met.
- Interpretation: exp-081's narrow whirlpool confirmation was a protocol artifact.

**Diagnostic (pre-stated, does not affect main verdict):**
- If RAND ≥ 20/144 at MAX_DX=256: protocol approaches exp-007 faithfulness; discrepancy resolved.
- If RAND < 10/144 even at MAX_DX=256: additional unexplained protocol difference persists.

**Kill criterion:** If H_artifact is confirmed (Δn ≤ 0), the exp-081 whirlpool finding is
retracted and the registry entry for exp-081 will be updated to status "superseded" with
honest negative noted. H_whirlpool status becomes "protocol-sensitive, not confirmed robustly."

---

## Protocol

- Model: GPT-2 (gpt2, cached ~1.1 GB, no download)
- Same as exp-081 EXCEPT MAX_DX: **256** (vs 64 in exp-081)
- SEQ_LEN=512, N_INPUTS=50, MIN_POS=64, RNG_SEED=42
- REAL: 50 coherent text windows from writing/ (same source files as exp-081)
- RAND: 50 random-token sequences (numpy rng seed=42, same as exp-081)
- Lag profile fit: power-law log-log on G[1:256] (dx=1..255)
- Threshold: R²>0.90, SYK-near [0.20, 0.30] (same as exp-081 and exp-007)
- Device: MPS (Apple M5 Max)

---

## Results (2026-07-06)

| Condition | power-law heads (R²>0.90) | n_syk_near | frac_syk/total | median_Δ (pl) |
|---|---|---|---|---|
| REAL (coherent) | 29/144 | 8 | 5.6% | 0.276 |
| RAND (random) | 19/144 | 5 | 3.5% | 0.380 |

Δn_syk_near (REAL − RAND): **+3**
Δmedian_Δ (REAL − RAND): **−0.105**

**H_robust: NOT CONFIRMED** (Δn=+3 < threshold +5)
**H_artifact: NOT CONFIRMED** (Δn=+3 > 0; REAL still > RAND)
**Overall verdict: AMBIGUOUS**

Diagnostic: RAND count (5/144) still far below exp-007 baseline (44/144) even at MAX_DX=256.
The protocol discrepancy is not resolved by larger lag range.

---

## Interpretation

**The exp-081 whirlpool finding (Δn=+6) does not robustly replicate at MAX_DX=256.** The
margin shrinks from +6 to +3, below the pre-stated +5 threshold. This means the narrowly-
confirmed whirlpool result was sensitive to the lag range parameter.

However, **H_artifact is also not confirmed**: the REAL > RAND trend persists (+3 > 0). The
signal is weakened but not reversed. Full retraction of the whirlpool hypothesis is not
warranted by this result alone.

**Why did power-law head counts DROP with larger MAX_DX?**
The most likely mechanism: GPT-2 (ctx=1024) has a conformal range ξ ≈ 12–22 tokens
(consistent with exp-063's finding that ξ ∝ training context for ctx-1024 models). Beyond
that range, the attention profile is no longer cleanly power-law. Including lags up to MAX_DX=256
adds a large "non-power-law tail" region that degrades the log-log fit R² and reduces the
number of heads exceeding the 0.90 threshold. MAX_DX=64 was already at or beyond the edge
of the conformal range; 256 goes well beyond it.

REAL: 90→29 power-law heads (MAX_DX=64→256). RAND: 21→19 (barely changed).
Interpretation: coherent text produces more consistent *short-range* attention patterns
(dx ≤ 64) that look power-law; at longer lag ranges, the profiles break down similarly
across conditions.

**The unresolved discrepancy with exp-007 (44/144 RAND):**
Even at MAX_DX=256, RAND is 5/144 — far below exp-007's 44/144. MAX_DX is not the explanation.
The most plausible remaining cause: exp-007 used `cutoff_low=3` (excluding dx=1 and dx=2 from
the fit), while exp-081/082 start from dx=1. The attention sink at dx=1 (position 0 absorption)
creates a large outlier that distorts the log-log regression and reduces R². Excluding dx<3 as
exp-007 did would remove this outlier and likely restore the 44/144-class counts.

**Recommended follow-up (exp-083):** Re-run the whirlpool/crystal test using exp-007's exact
fitting protocol (cutoff_low=3, or equivalently fitting G[3:MAX_DX]) to:
(a) reconcile with the exp-007 RAND baseline
(b) test whether the whirlpool signal survives on a protocol that sees 44/144 RAND heads

Until that test runs, the whirlpool finding should be treated as preliminary and
protocol-sensitive. exp-081's status remains "partial" in the registry.
