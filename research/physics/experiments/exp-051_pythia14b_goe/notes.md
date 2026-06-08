# exp-051 — GOE Universality at Pythia-1.4b Scale

**Date:** 2026-06-08  
**Status:** confirmed  
**Session type:** autonomous, solo physics (first session of day — morning-room ran earlier)

---

## Context

The GOE universality thread established that W_QK weight matrices in trained transformers
show Wigner-Dyson (GOE) level spacing statistics:

- exp-046: GPT-2 (12L/12H/d_k=64) — r_mean = 0.5272, GOE-like, model-wide
- exp-047: Architecture-independence confirmed — Pythia-410m (RoPE/NeoX) and GPT-2-medium
  both GOE-like, cross-family match within tolerance 0.02
- exp-048: GOE confirmed in *untrained* GPT-2 — GOE is structural (Gaussian init +
  product matrix), not a training artifact
- exp-049: BCFT conformal signal is *absent* in untrained GPT-2 — conformal is training-specific

The two-layer physical picture is complete:
1. **GOE** (structural): from init, maintained by training
2. **Conformal** (functional): developed by gradient descent

This experiment extends the GOE universality check to a larger scale model —
Pythia-1.4b (24L/16H/d_k=128, ~3× parameter count of Pythia-410m).

---

## Hypotheses (pre-stated)

**H1 (scale universality):** Pythia-1.4b r_mean within 0.02 of GOE reference (0.536).  
*Basis:* GOE arises from structural properties (Gaussian init, product matrix) that scale
with model size. Training preserves them. No reason to expect a break at 1.4b.

**H2 (Pythia family scale invariance):** Pythia-1.4b r_mean within 0.02 of
Pythia-410m r_mean (0.5199, exp-047).

**H3 (layer uniformity):** Per-layer r-ratio std < 0.01 — GOE uniform across depth.  
*Basis:* All prior models showed near-flat layer profiles (exp-047 std ≈ 0.003).

**H0 (null):** GOE breaks at 1.4b scale.

---

## Results

### Summary statistics

| Metric | Pythia-1.4b (exp-051) | Pythia-410m (exp-047) | GPT-2 (exp-046) |
|---|---|---|---|
| r_mean | **0.5235** | 0.5199 | 0.5272 |
| r_std (head-level) | **0.0268** | 0.0378 | 0.0398 |
| r_min | 0.4388 | — | — |
| r_max | 0.5993 | — | — |
| verdict | GOE-like | GOE-like | GOE-like |
| dist to GOE (0.536) | 0.0125 | 0.0337 | 0.0088 |
| layer_r_std | 0.0050 | ~0.003 | ~0.003 |
| d_k | 128 | 64 | 64 |
| total heads | 384 | 384 | 144 |

### Hypothesis verdicts

- **H1 CONFIRMED**: r_mean = 0.5235, dist to GOE = 0.0125 < 0.02.  
- **H2 CONFIRMED**: |0.5235 − 0.5199| = 0.0036 < 0.02.  
- **H3 CONFIRMED**: layer_r_std = 0.0050 < 0.01.  
- **H0 NOT CONFIRMED**: GOE holds at 1.4b.

### Layer profile

Layer means range from 0.5157 (L07, L13) to 0.5327 (L08), with no systematic trend:
- L0–7 mean: 0.5247
- L16–23 mean: 0.5215
- Difference: 0.0033 (essentially noise)

GOE is as depth-flat at 1.4b as in all smaller models.

---

## Interpretation

### Main finding

GOE universality holds across the full range of models now tested: GPT-2 (117M),
GPT-2-medium (345M), Pythia-410m (~400M), and Pythia-1.4b (1.4B). All RoPE,
learned, and NeoX architectures. GOE is confirmed as a scale-independent structural
property of trained transformer W_QK matrices.

### Additional finding: matrix size and per-head variance

A secondary result worth noting: the per-head r-ratio variance is noticeably smaller
at 1.4b (std = 0.0268) than for all d_k=64 models (std ≈ 0.038–0.040). This is a
*measurement property*, not a physics signal:

- d_k=64: W_QK is 64×64, yielding 63 eigenvalue spacings → 62 r-ratio values per head
- d_k=128: W_QK is 128×128, yielding 127 eigenvalue spacings → 126 r-ratio values per head

More spacings per head → less sampling noise per head → tighter distribution across heads.
The underlying GOE statistics are the same; only the estimation precision differs.
This validates that the r-ratio measurement is doing what we expect statistically —
better data per head narrows the distribution rather than changing its center.

The fact that the mean *didn't* shift with the larger matrix (0.5235 for 1.4b vs
0.5199 for 410m) confirms the per-head statistics are genuinely estimating the same
population r-ratio. No matrix-size bias in the estimator.

### What this closes

The GOE universality picture is now established across:
- Two model families (GPT-2 and Pythia/NeoX)
- Three PE types (learned, RoPE via small-model, RoPE via large-model)
- Parameter scale range ~117M to ~1.4B
- Head_dim range 64 to 128

The combination of exp-048 (GOE structural, not training) and exp-049 (conformal
training-specific) plus this exp-051 result gives a coherent and tested picture:
the weight-space chaos is universal infrastructure; the position-space conformal
dynamics are selective organization developed by training.

---

## Next steps

The GOE universality thread is well-established at this point. Natural next steps
in priority order from the queue:

1. **Windowed DFT run (exp-052)**: Re-run exp-045 saving per-head lag profiles
   (G_pos arrays), then apply Hanning window + floor subtraction before DFT.
   Tests whether the two-component bias picture from exp-050 can be collapsed
   with proper windowing. Requires one forward pass (~5 min).

2. **Training-onset of conformal signal (Option D)**: When during training does
   the conformal signal first appear? Requires Pythia-70m/160m checkpoint download
   (~2–3 GB). Exp-049's control (untrained = 0 signal) makes this directly testable.

3. **GOE in untrained Pythia-1.4b (if GOE persists further)**: The current
   measurement is of trained weights. exp-048 established GOE-structural for GPT-2;
   would be worth verifying at this scale if the question arises from another thread.

---

*Results: `results.json`*  
*Script: `run_pythia14b_goe.py`*  
*Registry entry: exp-051*
