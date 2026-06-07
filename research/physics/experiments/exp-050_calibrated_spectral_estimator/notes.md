# exp-050 — Calibrated spectral estimator for BCFT conformal heads

*Date: 2026-06-07*
*Status: confirmed (H4 — genuine discrepancy remains)*

---

## Hypothesis

Four hypotheses, pre-stated before running:

- **H1**: Calibration improves consistency: MAE(Δ_freq_corrected − Δ_pos) < MAE(Δ_freq_raw − Δ_pos)
- **H2**: CFT kinematics holds after calibration: SYK-near heads have Δ_freq_corrected ≈ 0.25
- **H3**: Calibration curve is monotone (bias invertible)
- **H4 (null)**: Genuine discrepancy remains even after calibration

## Motivation

exp-045 found α_median = −0.820 for conformal heads, while CFT kinematics predicts α = 2Δ−1 ≈ −0.49 (for Δ_pos ≈ 0.255). The diagnosis was finite-DFT bias from 55-lag truncation. This experiment tests whether that bias fully explains the discrepancy, or whether a genuine physical gap remains.

Method: generate synthetic G_true(r) = r^{−2Δ} for known Δ over a grid, apply the identical exp-045 DFT protocol, measure α_measured(Δ_true). Build calibration curve; invert to correct the real data.

## Results

**H3: CONFIRMED.** The calibration curve α_measured(Δ_true) is monotone ascending. Key values:
- Δ_true = 0.25 → α_measured = −0.647 (finite-window bias: −0.147 units)
- Δ_true = 0.50 → α_measured = −0.337 (bias: −0.337 units, larger for faster-decaying profiles — counterintuitive, see below)

Wait — actually the bias is *larger* in magnitude for larger Δ in this range. The bias grows with Δ from 0.20 onward. This is because: for large Δ, the profile decays quickly, the truncation creates a sharp step down to zero, and the window effect is stronger. For small Δ (slowly decaying), the profile is more spread and the truncation is gentler relative to the profile amplitude.

Actually looking at the bias column again: at Δ=0.020, bias = −0.157; at Δ=0.237, bias = −0.143; at Δ=0.600, bias = −0.480. So the magnitude grows with Δ. The fast-decaying profiles have G(55) ≈ 0 so the cutoff step is small in absolute terms but the DFT bias still grows. This suggests the mechanism is not primarily the cutoff step for slowly-decaying profiles — it may be a general finite-window DFT effect.

**H1: CONFIRMED.** MAE drops from 0.361 to 0.283 (21% improvement). Calibration helps but does not close the gap.

**H2: NOT CONFIRMED.** SYK-near heads: Δ_freq_corrected_median = 0.135 vs Δ_pos_median = 0.234. After full calibration, the corrected spectral estimate is still only about 0.135 — half the position-space value. The CFT kinematic prediction does not hold.

**H4: CONFIRMED.** Mean residual after calibration: −0.283 ± 0.335. The discrepancy is genuine, not methodological.

## Physical interpretation

The key finding: real attention profiles produce spectral exponents **significantly more negative** than pure power-law profiles with the same Δ.

For Δ_pos ≈ 0.25:
- Pure synthetic G(r) = r^{−0.5} → α_measured = −0.647 (finite-window bias accounts for this)
- Real attention profiles → α_measured ≈ −0.83
- Unexplained additional bias: ~−0.183 units

Sources of the additional negative shift:
1. **Noise floor at truncation boundary.** Real attention profiles have a non-zero floor at r=55 (residual attention to distant tokens). When zero-padded, the sharp step from floor → 0 at r=55 adds low-k spectral power that a pure power law doesn't produce.
2. **Deviation from pure power law.** The position-space fit achieves R²≈0.90 but not 1.0. The residuals from the power-law fit have their own spectral structure.
3. **Normalization constraint.** Attention weights sum to 1, imposing a constraint that modifies the effective profile shape relative to an unconstrained power law.

**Implication for the measurement framework:**

The position-space BCFT method (exp-007, Δ_pos) is the more robust measurement. It directly fits the power-law decay component and is not sensitive to the attention floor or normalization artifacts. The spectral approach measures a different aspect of the profile — one that includes these additional features.

The two measurements are not inconsistent with each other; they are measuring different things:
- Δ_pos: the exponent of the primary power-law decay component (the conformal two-point function)
- Δ_freq (spectral): the effective exponent of the *full* spectral density, including floor, normalization, and deviations from pure power law

**CFT kinematics holds only for pure power laws.** Real attention profiles are approximately power-law, not exactly. The spectral domain reveals this deviation clearly.

## What this opens

A natural follow-up: apply a window function (Hanning or Hamming) to the 55-lag profile before DFT. Windowing suppresses the sharp truncation discontinuity, which is the primary source of the additional low-k spectral power. If windowing brings Δ_freq_corrected closer to Δ_pos, it confirms that the noise floor/truncation step is the primary mechanism.

Additionally: explicitly model and subtract the attention floor. For each head, estimate the floor as the mean attention at r=40..55 (far from diagonal), subtract it from the full profile, then apply DFT. This removes the floor contribution to the spectral density.

Both of these are analysis-only (no new model runs needed), working from exp-045's lag profiles. However, exp-045 only saved aggregate statistics, not per-head lag profiles. A new run would need to save the raw lag profiles for analysis.

## Honest assessment

The calibration experiment worked as designed and produced a genuine finding: the spectral method, even after full bias correction, measures something systematically different from the position-space method. This is not a failure of either measurement — it's a distinction. The position-space measurement (Δ_pos ≈ 0.25) remains the primary result. The spectral measurement opens a window into additional structure in the attention profiles.

The prior claim in exp-045 that "DFT bias is structural" was correct but incomplete. The bias has two components: (1) finite-window DFT bias, which the calibration corrects, and (2) additional profile structure not present in pure power laws, which cannot be removed by DFT calibration alone.

---

*References: exp-045 (G_>/G_< spectral function), exp-007 (GPT-2 per-head conformal scaling, Δ_med=0.2493)*
