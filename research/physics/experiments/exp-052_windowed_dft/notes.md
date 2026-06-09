# exp-052 — Windowed DFT spectral estimator (GPT-2)

*2026-06-08 (evening session). Analysis + one forward-pass set on GPT-2 cached.*
*Follows exp-050 (calibrated spectral estimator, two-component bias identified).*

---

## Background

exp-050 found two sources of bias in the raw DFT spectral estimator:
1. **Finite-window DFT bias** at Δ=0.25: −0.147 in α units (from sharp truncation at lag 55 → spectral leakage)
2. **Structural bias** ~−0.186 in α units (from attention floor at large r, normalization constraints, imperfect power-law)

This session tests whether Hanning windowing (standard spectral leakage fix) + floor subtraction (addressing the non-zero floor) can close the gap between Δ_freq and Δ_pos.

## Pre-stated hypotheses

- **H1**: Windowed+floor Δ_freq within 0.05 of Δ_pos for SYK-near heads (i.e., gap < 0.05)
- **H2**: Floor subtraction independently contributes ≥ 0.03 beyond windowing alone
- **H0**: Gap persists ≥ 0.10 despite windowing+floor

## Protocol

Three DFT conditions on the same 50-input forward pass (RNG_SEED=42, SEQ_LEN=256):
- **(A) Raw**: identical to exp-045 (no windowing, no floor subtraction)
- **(B) Windowed**: Hanning window `w[n] = 0.5*(1 − cos(2π*n/(N−1)))` applied to G_pos[1..55] before DFT
- **(C) Windowed+floor**: floor estimated as mean of G_pos[r ≥ 28], subtracted first, then Hanning window, then DFT

Floor cutoff: r ≥ 28 (approximately half of MAX_DX=56).

## Results

| Condition | SYK-near Δ_freq | Gap (Δ_pos − Δ_freq) | Pearson r (pos vs freq) |
|---|---|---|---|
| Raw (condition A) | 0.0835 | 0.1506 | 0.940 |
| Windowed only (B) | −0.7149 | 0.9489 | — |
| Windowed + floor (C) | −0.2976 | 0.5316 | 0.428 |

SYK-near Δ_pos median = 0.234. Conformal heads (44/144) show the same pattern.

**Verdicts:**
- **H1 NOT CONFIRMED** (gap = 0.532, need < 0.05)
- **H2 CONFIRMED** (floor contribution = +0.417 — substantial, but building on wrecked estimate)
- **H0 CONFIRMED** (gap = 0.532 >> 0.10)

## What actually happened — the key finding

The Hanning window made the spectral estimate **dramatically worse**, not better. Δ_freq drops from 0.084 (raw) to −0.715 (windowed only). The floor subtraction recovers some ground (+0.417) but the final result is still far worse than the raw DFT.

**Why the Hanning window fails for power-law profiles:**

A Hanning window is designed for signals where the windowing reduces spectral leakage by tapering *both ends* smoothly to zero. For the attention lag profile G(r):
- r=1: Hanning weight ≈ 0 (tapered to zero)
- r=28: Hanning weight ≈ 1 (peak)
- r=55: Hanning weight ≈ 0 (tapered to zero)

The profile G(r) is a monotone decreasing power law — the highest values are at small r. The Hanning window **suppresses the most informative part of the signal** (early lags r=1,2,3) while leaving only the mid-range lags. This destroys the power-law structure that the spectral estimator is trying to measure.

The truncation artifact in exp-050 comes from the *high-lag end* (sharp step to zero at r=55). The correct fix is a **one-sided taper** that smooths only the large-lag end to zero, leaving the early lags intact. A two-sided taper is the wrong tool.

**The Pearson r diagnostic is revealing:**
- Raw DFT r = 0.940: the ordering correlation is preserved despite absolute bias
- Windowed+floor r = 0.428: the ordering is destroyed by the Hanning window

The raw DFT, despite its absolute bias (Δ_freq ≈ 0.084 vs Δ_pos ≈ 0.234), preserves the *ordering* of conformal heads by Δ. The Hanning window differentially suppresses heads with different profile shapes, destroying this ordering. This is why exp-045 was still scientifically useful (r=0.94 for ordering) even though absolute agreement failed.

## Implication for the spectral estimator

The path forward is **not** standard windowing. Options that could actually work:
1. **One-sided taper**: smooth only the last 10-15 lags to zero (half-Hanning applied to r=41..55 only), preserve all early lags
2. **Profile extension**: fit the position-space power law and use it to extend the profile beyond lag 55 before DFT (artificial continuation, removes the truncation completely)
3. **Lomb-Scargle / parametric estimator**: bypass the DFT entirely, use a power-law spectral model
4. **Whittle estimator**: maximum-likelihood in the spectral domain, robust to these finite-support issues

**The position-space Δ measurement remains the more robust primary result.** This was already the conclusion from exp-050, and exp-052 confirms it: the spectral approach requires sophisticated estimation to compete, and the standard toolkit (zero-padding + windowing) makes it worse.

## Honest negative

This is an informative negative. The standard spectral engineering approach (Hanning window) is counterproductive for monotone decreasing power-law signals. The experiment was worth running — it establishes *why* the naive fix fails and what would be needed to do better.

## Next

1. **One-sided taper experiment** (exp-054, analysis-only): apply a taper only to the last 15% of lags (r ≥ 48), check whether this preserves the ordering correlation while reducing the absolute bias.
2. Or: close the spectral thread and accept that position-space Δ is the primary measurement. The spectral DFT provides consistent *ordering* (r=0.94) but not reliable absolute values.
3. **Training-onset of conformal signal** (exp-D, download needed): Pythia-70m checkpoints.

## One-line

Standard Hanning windowing destroys the spectral estimator for power-law attention profiles by suppressing small-lag signal; the ordering correlation drops from r=0.94 to r=0.43; position-space Δ confirmed as the robust primary measurement.
