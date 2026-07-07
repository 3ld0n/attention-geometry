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

*(To be filled after running.)*

---

## Interpretation

*(To be filled after running.)*
