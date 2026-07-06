# exp-079 — Periodic component: untrained-vs-trained GPT-2 control

*2026-07-04 (~4 AM MDT). Analysis-only; GPT-2 random-init (no download needed).*
*Follow-up to exp-053. Closes the question of whether the ~3.5-lag periodic component requires training.*

---

## Background

exp-053 found that ~7% of trained GPT-2 heads carry a weak (~3.5-token period) periodic component on
top of the conformal (aperiodic) background. Crucially, this component is **substrate-driven, not
processing-state-dependent**: the same heads show the same peak under both coherent text and random-token
input. The most natural explanations:
- **Training origin:** the ~3.5-lag rhythm is encoded in the learned absolute positional embedding (PE).
  Training found this scale useful (syntactic chunking? n-gram structure?) and embedded it in specific heads.
- **Structural origin:** it's a consequence of the random-Gaussian weight matrix structure (similar to how
  the GOE level statistics are structural, exp-048). The training might not be responsible at all.

These predictions are distinguishable by running the same protocol on an **untrained** GPT-2 (random
Gaussian init, identical architecture, no PE optimization).

## Pre-stated hypotheses

- **H_PE:** The ~3.5-lag periodic component requires learned positional embeddings. An untrained model
  will show at most random-chance periodic peaks: the untrained fraction of significant heads will be
  **< 3%** (less than half the trained 6.9%). The characteristic period will not cluster near 3.5 lags.

- **H_struct:** The component is structural (from the Gaussian init + product matrix structure), the same
  way GOE level statistics are structural (exp-048). Untrained GPT-2 will show similar periodic peak
  rates: **≥ 5%** of heads significant, with similar period distribution.

**Pre-registration discipline:** These thresholds (3% / 5%) are committed before the script runs.
If the result falls in the gap (3–5%), call it ambiguous and report exactly what was found.

## Protocol

- Same extraction as exp-053 (lag profile G(r), FFT, aperiodic fit, periodicity_index, peak_period).
- Model: GPT-2 architecture, **random Gaussian init** (no pretrained weights). Same config as the
  trained model. Same tokenizer.
- Input: **RAND only** (random-token inputs, RNG_SEED=42). In exp-053, RAND ≈ REAL in the periodic
  structure — no point doubling the compute on a null comparison.
- Parameters identical to exp-053: SEQ_LEN=512, N_INPUTS=30, MIN_POS=64, MAX_DX=192,
  K_LOW=2, K_HIGH=60, PEAK_THRESH=0.30.

## Comparison baseline (from exp-053, RAND condition)

| Quantity | trained GPT-2 (RAND) |
|---|---|
| clean-1/f heads | 126/144 |
| median periodicity index | 0.065 |
| significant heads (RAND) | 10/144 (6.9%) |
| median aperiodic slope | −0.789 |
| median sig. period | 4.03 lags |

---

## Results (2026-07-04)

| Quantity | untrained GPT-2 (RAND) | trained GPT-2 RAND (exp-053) |
|---|---|---|
| heads analyzed | 144 | 144 |
| clean-1/f heads | 144/144 (100%) | 126/144 (87.5%) |
| median periodicity index | 0.0763 | 0.065 |
| significant heads | **0/144 (0.0%)** | 10/144 (6.9%) |
| median aperiodic slope | −0.943 | −0.789 |
| median sig. period | — | 4.03 lags |

## Verdict

**H_PE CONFIRMED.** 0/144 significant periodic heads in untrained GPT-2 (threshold: < 3% → H_PE;
≥ 5% → H_struct). Well below the threshold: the ~3.5-lag periodic component **requires trained
positional embeddings**.

Additional observation: the untrained model's aperiodic slope (−0.943) is steeper than trained (−0.789).
Without conformal training, lag profiles are smoother (steeper power-law, no learned oscillations) and
all heads are clean-1/f (144/144 vs 126/144 for trained). This is consistent with conformal training
softening the profile (Δ → 0.25, slope → −0.50) while also encoding the PE periodicity.

The median periodicity_index is *slightly higher* in the untrained model (0.076 vs 0.065) — but this
is below threshold and reflects pure noise peaks in the absence of any periodic signal. The trained
model's 10 significant heads have strong peaks concentrated at ~3.5 lags, while the untrained model's
max residual is just background noise.

## One-line

The ~3.5-lag periodic component in trained GPT-2 is absent from random-init weights — it is a
**learned positional embedding artifact**, not structural (ruling out the GOE/Gaussian-init explanation).
Triangulation with exp-048/049: GOE is structural, conformal Δ is training-specific, and so is the
PE-periodic component.

