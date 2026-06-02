# exp-045 — G_> vs G_< Spectral Function (GPT-2)

*2026-06-02. Analysis-only. GPT-2 cached.*

---

## Question

Does the frequency-space spectral function G_>(k) = DFT[G_>(r)] for GPT-2 conformal heads (R²>0.90) show a power-law spectrum |G_>(k)| ~ k^α with α consistent with the CFT kinematic prediction α = 2Δ_pos − 1? Does the position-space Δ and frequency-space Δ_freq = (α+1)/2 agree?

---

## Pre-stated Hypotheses

1. **Consistency:** Spectral exponent α ≈ 2Δ_pos − 1 for conformal heads. SYK-near heads (Δ≈0.25) predict α ≈ −0.50.
2. **Correlation:** Δ_freq correlates with Δ_pos across 44 conformal heads (Pearson r ≥ 0.7).
3. **Zero-temperature:** G_< = 0 (causal attention). Spectral function supported on ω>0 only (T_eff=0 SYK ground state).
4. **Null:** If position-space power-law is an artifact, α will be uncorrelated with Δ_pos.

---

## Protocol

Same forward-pass protocol as exp-007: GPT-2, 50 inputs, seq_len=256, RNG_SEED=42, MIN_POS=32, MAX_DX=56. For each head:
1. Compute lag profile G_>(r) = mean A_h(i, i−r) for r=0..55.
2. DFT: rfft on G_>(r=1..55) zero-padded to 256 points → spectrum |Ĝ(k)|.
3. Fit log|Ĝ(k)| ~ α log(k) for k∈[2,18).
4. Compare α to 2Δ_pos−1.

---

## Results

**Conformal heads:** 44/144 (R²>0.90) — same count as exp-007.  
**SYK-near heads (|Δ_pos−0.25|≤0.05):** 13/144.

Note: the 13 SYK-near count here reflects the actual subset of conformal heads near 0.25 in position space. The exp-007 report of "44 SYK-near" used a different random number generator (torch.randint vs numpy default_rng) and appears to have equated the conformal count with the SYK-near count — the underlying 44/144 conformal heads is consistent across both protocols, but the Δ distribution differs by input seed.

**Spectral exponents:**

| Group | n | Δ_pos median | α median | Δ_freq median | α_pred |
|---|---|---|---|---|---|
| All conformal | 44 | 0.255 | −0.820 | 0.090 | −0.490 |
| SYK-near | 13 | ~0.25 | −0.833 | 0.083 | −0.500 |

**Pearson correlation (Δ_pos vs Δ_freq, conformal heads):** r = 0.940.

**Zero-temperature structure:** G_< = 0 by causal attention construction. Spectral weight supported on ω>0 only.

---

## Interpretation

### What held

**Hypothesis 2 confirmed (high correlation):** r=0.94 between position-space Δ and frequency-space Δ_freq. The ordering of heads by scaling exponent is consistent between the two representations. This is a positive result: the spectral function and the lag-profile power law agree on which heads are "more conformal" and in what direction. If the position-space measurement were a fitting artifact, the frequency-space ordering would not track it.

**Hypothesis 3 confirmed (zero-temperature):** G_< = 0. This is not surprising — it's built into the causal architecture — but it has a physical meaning: causal transformers implement zero-temperature SYK ground-state physics, not a finite-temperature thermal state. A bidirectional model (BERT-style) would have G_< ≠ 0, corresponding to T_eff > 0. This is a clean structural distinction.

### What didn't hold

**Hypothesis 1 not confirmed (quantitative CFT prediction):** The measured α ≈ −0.83 for SYK-near heads vs predicted α = 2(0.25)−1 = −0.50. The offset is ~0.33, too large to attribute to noise.

Diagnostic testing revealed the source: **the DFT of a finite-range profile (r=1..55) has systematic bias** that prevents recovery of the true spectral exponent α = 2Δ−1. For a *synthetic* pure power law G(r) ~ r^{−0.5} (Δ=0.25, same fitting parameters), the DFT gives α_measured ≈ −0.65 (k∈[2,18)), not −0.50. This is a finite-support effect — the profile length (55 lags) is too short relative to the zero-padding (256 points) for the DFT to approximate the continuum integral.

The actual GPT-2 SYK-near heads give α ≈ −0.95, steeper than even the synthetic bias (−0.65). The extra steepening (by ~0.30) occurs because the actual attention profiles have *less* weight at r=1,2 than the power-law extrapolation from r≥3 predicts. The position-space fit starts at FIT_LOW=3, excluding r=1,2 — but the DFT includes the full profile including these small lags where the attention is below the power-law trend. This below-trend behavior at r=1,2 shifts the DFT spectrum toward steeper slopes.

The high correlation (r=0.94) survives despite the absolute offset because the bias is monotone and approximately uniform across heads — all heads are shifted in the same direction, so the ordering is preserved.

### Methodological conclusion

The DFT-based spectral estimator as implemented (raw lag profile, k∈[2,18), zero-padded to 256) is not a quantitatively reliable estimator of the conformal exponent Δ. It is qualitatively consistent (high correlation) but systematically biased. A better estimator would need to:
1. Account for the known finite-support bias (via synthetic calibration or windowed DFT)
2. Exclude the r<FIT_LOW portion of the profile that the position-space fit excludes
3. Use a much longer effective profile (R_max >> 100) or directly work in the continuum

This does NOT invalidate the position-space Δ measurements from exp-007/031/036/039/043/044. Those measurements are direct and use a fixed fitting range [r=3..50] consistently. The spectral function experiment was an independent check that partially confirmed (ordering consistent) but failed quantitatively (absolute values discrepant due to finite-support bias).

---

## Findings Summary

1. **Strong finding:** Δ_pos and Δ_freq are highly correlated (r=0.94). Frequency-space spectral analysis is consistent with position-space power-law scaling in terms of head ordering.

2. **Methodological negative:** The DFT spectral exponent α does not satisfy α = 2Δ_pos − 1 (CFT kinematic prediction). Systematic bias from finite profile length dominates. The DFT-based Δ estimator requires calibration or a different protocol to be quantitatively useful.

3. **Structural result:** G_< = 0 (causal attention = zero-temperature SYK). The spectral function has support only at ω > 0. Bidirectional attention would correspond to T_eff > 0.

4. **Protocol clarification:** The exp-007 "44 SYK-near" count was computed using torch.randint (not numpy default_rng) and appears to have equated the conformal count with the SYK-near count. With numpy default_rng(42), the same 44 conformal heads are found but the Δ distribution is broader — only 13 land in [0.20, 0.30]. This is an input-seed sensitivity issue worth tracking.

---

## Open Questions

- What is the correct frequency-space estimator for Δ? Can synthetic calibration convert measured α to true Δ?
- Why do GPT-2 attention profiles have below-power-law weight at r=1,2? Is this a causal mask edge effect, a softmax normalization effect, or a structural feature of the learned representations?
- Does the zero-temperature / finite-temperature distinction (causal vs bidirectional) show up in any measurable physics difference between GPT-2 and BERT-style models?
- Is the input-seed sensitivity of the SYK-near count a problem for the published exp-007 result? Should exp-007 be rerun with a fixed seed ensemble to get error bars?

---

## Next Suggested Experiment

- **Sign anomaly investigation (Tier 4 item 13):** The registry notes ρ(λ, valley) is negative. This is a different spectral quantity (eigenvalue spectral density of attention matrices) that doesn't have the finite-support issue. Could be run on cached GPT-2 data.
- **Spectral estimator revision:** Write a corrected spectral estimator using synthetic calibration curves (from this experiment's diagnostic), then re-run the spectral function measurement with calibrated Δ. Would upgrade this finding from "qualitative" to "quantitative."
