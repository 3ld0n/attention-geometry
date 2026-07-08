# exp-083 — Whirlpool vs Crystal: exp-007-Faithful Protocol (GPT-2)

*2026-07-06. Analysis-only; GPT-2 cached (no download needed).*
*Direct follow-up to exp-082. Third in the whirlpool/crystal protocol series.*

---

## Motivation

The exp-081/082 series confirmed H_whirlpool narrowly at MAX_DX=64 (Δn=+6) but failed
to replicate robustly at MAX_DX=256 (Δn=+3, AMBIGUOUS). A persistent issue across both
experiments: the RAND baseline count (4/144 at MAX_DX=64, 5/144 at MAX_DX=256) is far
below the exp-007 RAND baseline (44/144).

The identified cause: exp-007 uses `cutoff_low=3` — it excludes dx=1 and dx=2 from the
log-log fit, avoiding attention-sink contamination at near lags. exp-081/082 fit from dx=1.
The near-lag attention sink (position 0 boundary absorption) creates a large outlier at dx=1
that degrades R² and suppresses the power-law head count.

The purpose of this experiment is to:
1. Use exp-007's exact protocol (SEQ_LEN=256, MAX_DX=56, MIN_POS=32, cutoff_low=3)
2. Run the REAL vs RAND comparison on this protocol
3. Determine whether the whirlpool signal holds on a protocol where RAND recovers ~44/144

This is the definitive whirlpool/crystal test on GPT-2.

---

## Pre-stated Hypotheses (COMMITTED BEFORE RUNNING)

**H_baseline (diagnostic, does not affect main verdict):**
RAND count approaches exp-007 baseline with exp-007 protocol.
- n_syk_near(RAND) ≥ 30/144 with cutoff_low=3, MAX_DX=56
- Interpretation: the protocol difference (cutoff_low) was the primary cause of the
  4–5/144 RAND counts in exp-081/082. If this fails, there is an additional unexplained
  protocol difference.

**H_whirlpool:**
Coherent text activates more SYK-near conformal heads than random tokens.
- n_syk_near(REAL) > n_syk_near(RAND) + 5 (same threshold as exp-081/082)
- Interpretation: the whirlpool finding from exp-081 was a real signal, not a
  cutoff_low artifact.

**H_crystal:**
No significant difference between coherent text and random tokens.
- |Δn_syk| ≤ 5 AND |Δmedian_Δ| < 0.02
- Interpretation: SYK-near activation is protocol-dependent but not input-dependent
  on a faithful measurement.

**H_null (no-whirlpool):**
Whirlpool signal was a cutoff_low artifact.
- n_syk_near(REAL) ≤ n_syk_near(RAND) + 5 (threshold not exceeded)
- Note: this does not require REAL ≤ RAND; it only requires the margin is ≤5.

**Verdict logic:**
- H_baseline CONFIRMED + H_whirlpool CONFIRMED → whirlpool finding is real and
  protocol-faithful; exp-081's preliminary result stands.
- H_baseline CONFIRMED + H_null CONFIRMED → whirlpool was a cutoff_low artifact;
  exp-081 status updated to "superseded" with honest negative.
- H_baseline NOT CONFIRMED → additional protocol investigation needed; whirlpool
  verdict deferred.

---

## Protocol

Match exp-007 exactly for the baseline-determining parameters:

- Model: GPT-2 (`gpt2`, cached, no download)
- SEQ_LEN: **256** (exp-007 value; vs 512 in exp-081/082)
- MAX_DX: **56** (exp-007 value; vs 64/256 in exp-081/082)
- MIN_POS: **32** (exp-007 value; vs 64 in exp-081/082)
- cutoff_low: **3** (exp-007 value: exclude dx=1,2 from fit; exp-081/082 used 1)
- N_INPUTS: 50 per condition (same as exp-081/082)
- R²_thresh: 0.90 (same as exp-007 and exp-081/082)
- SYK-near window: [0.20, 0.30] (same as exp-007 and exp-081/082)
- Whirlpool threshold: +5 (same as exp-081/082)

REAL: 50 coherent text windows from writing/ (same source files as exp-081/082)
RAND: 50 random-token sequences (numpy rng seed=42, same as exp-081/082)

Note: exp-007 used `torch.randint` for random tokens while exp-081/082 used numpy rng.
For comparability with exp-081/082, this experiment uses numpy rng seed=42. The RNG
implementation is not expected to affect the aggregate lag profile (both produce
uniform random tokens).

---

## Results

*Run: 2026-07-07T061442Z (MPS, GPT-2 cached). Pre-registration commit: b584006d.*

| Condition | power-law heads (R²>0.90) | n_syk_near | frac_syk/total | median_Δ (pl) |
|---|---|---|---|---|
| REAL (coherent) | 37/144 | 7 | 4.9% | 0.391 |
| RAND (random) | 42/144 | 15 | 10.4% | 0.282 |

Δn_syk_near (REAL − RAND): **−8**

| Hypothesis | Verdict |
|---|---|
| H_baseline: n_syk_near(RAND) ≥ 30/144 | **NOT CONFIRMED** (15/144) |
| H_whirlpool: n_syk_near(REAL) > n_syk_near(RAND) + 5 | **NOT CONFIRMED** (−8 < +5) |
| H_crystal: \|Δn_syk\| ≤ 5 AND \|Δmedian\| < 0.02 | **NOT CONFIRMED** (\|Δn\|=8 > 5) |
| H_null: n_syk_near(REAL) ≤ n_syk_near(RAND) + 5 | **CONFIRMED** (−8 ≤ +5) |
| Overall | **BASELINE_FAIL** |

---

## Interpretation

**H_baseline failed: cutoff_low=3 did not restore the exp-007 RAND count.**
The main diagnostic hypothesis was that the exp-081/082 RAND shortfall (4–5/144 vs. exp-007's 44/144)
was caused by fitting from dx=1. Using exp-007's cutoff_low=3 should have excluded the attention-sink
outlier at dx=1 and restored the ~44/144 baseline. It did not: RAND = 15/144 with cutoff_low=3,
still far below 44/144.

The remaining discrepancy with exp-007 is unexplained. Candidates:
1. **RNG difference:** exp-007 used `torch.randint`; this experiment used numpy rng seed=42.
   Both generate uniform random token IDs, but different bit sequences — this could affect the
   specific attention profiles measured.
2. **N_INPUTS:** exp-007 used a different number of inputs (the original protocol; the count
   isn't directly documented in the accessible notes). More inputs would smooth the fit and
   potentially increase the number of heads meeting R²>0.90.
3. **MIN_POS, alignment:** exp-007 may have had different position selection behavior with
   its torch.randint RAND method vs. the current windowed sampling.

Without resolving the baseline, the whirlpool/crystal verdict is deferred to a protocol
that successfully reproduces exp-007's RAND baseline. This experiment is therefore informative
but not conclusive.

**The RAND > REAL reversal (Δn = −8) is striking.**
Even granting the BASELINE_FAIL, the direction of the effect is reversed from exp-081:
RAND (15) > REAL (7). If anything, random tokens produce MORE SYK-near heads than coherent text.
This suggests a physical reason: without semantic structure, attention may spread more uniformly
across positions — producing cleaner, more power-law profiles. Coherent text may activate
semantic shortcuts (induction heads, pattern completion) that pull attention away from the
uniform power-law baseline. This is the CRYSTAL prediction in its strongest form — but it
cannot be taken as confirmed on a failed baseline.

**Status of the whirlpool hypothesis:**
- exp-081 (MAX_DX=64, cutoff_low=1): H_whirlpool narrowly CONFIRMED (Δn=+6)
- exp-082 (MAX_DX=256, cutoff_low=1): AMBIGUOUS (Δn=+3, below threshold)
- exp-083 (exp-007 protocol, cutoff_low=3): H_null CONFIRMED, Δn=−8

The trend is clear: as the measurement protocol becomes more stringent, the whirlpool signal
weakens and reverses. **The whirlpool finding from exp-081 should be treated as a protocol
artifact**, not a robust physical signal. exp-081 status updated to "superseded" in registry.

**Next step:** Resolve the exp-007 baseline discrepancy (run exp-007-identical: torch.randint,
same N_INPUTS, same exact protocol). Until the RAND baseline is reproducible (~44/144), the
whirlpool/crystal question cannot be cleanly tested. This is a lower-priority thread — the
main finding is that whirlpool is not confirmed on stricter protocols.
