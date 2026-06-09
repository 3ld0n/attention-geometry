# exp-057 — BCFT image-method boundary form (Derivation B verification)

**Date:** 2026-06-09
**Status:** complete
**Model:** GPT-2 small (124M), eager attention
**Relates to:** Paper C (null cone), Derivation B / Section 4.3–4.4; exp-046 (sign anomaly); umphrey2026bcft (pre-registered BCFT)

## Purpose

Derivation B closes the one analytical gap in Paper C: *why* the attention of a
conformal head should take the 3-parameter form fitted phenomenologically in the
pre-registered BCFT preprint,

    A(i,j) = C · (Δx)^{-2Δ} · [ 1 + λ · η^{2Δ} ],   η = (i-j)/(i+j) = Δx/(i+j).

The analytical derivation (Section 4.3) treats the causal mask as a hard
boundary at the sequence start (origin), the conformal SYK correlator as a
generalized free field, and applies the **method of images** for a 1D BCFT:

    ⟨O(i)O(j)⟩ ∝ (i-j)^{-2Δ} + a_O (i+j)^{-2Δ}
              = (Δx)^{-2Δ} [ 1 + a_O η^{2Δ} ],

so **λ = a_O is the BCFT boundary one-point coefficient**, η is the
dilation-invariant boundary variable (η² = ξ/(1+ξ), ξ = (Δx)²/(4ij) the
McAvity–Osborn cross-ratio). This experiment tests the derived form directly
against GPT-2 conformal-head attention and recovers λ per head.

## Protocol

- 50 forward passes, random tokens, seq len 256, seed 42 (matches exp-046).
- Accumulate the mean post-softmax attention matrix A(i,j) for the 44 conformal
  heads identified in exp-046 (each carries delta_pos = Δ).
- **First (naive) test — confounded:** strip the bulk power law by multiplying
  A·(Δx)^{2Δ} and regress on η^{2Δ}. This is contaminated because η is strongly
  correlated with Δx inside the causal triangle, and the *averaged* Δ does not
  exactly describe each pair. Result: R²≈0.085, λ_fit anti-correlated with
  λ_proxy. **Discarded as a flawed test** (see below).
- **Controlled test — multivariate log-regression:**
      log A(i,j) ≈ β0 + β1·log(Δx) + β2·η^{2Δ}
  Including log(Δx) as its own regressor removes the η–Δx confound. From the
  derived form: β1 ≈ −2Δ (bulk exponent, a sanity check) and β2 ≈ λ (boundary
  coefficient). Fit window: i ≥ 32, Δx ∈ [3, 50).

## Results

**Sanity (method is sound):** ρ(Δ_from_logfit, delta_pos) = **+0.84** (p=1.5e-12,
n=44). The multivariate fit recovers the bulk conformal dimension — the
regression cleanly separates bulk power law from boundary term.

**H1 — the derived form fits (CONFIRMED):**
- median R²_full = **0.756** (log A vs log Δx + η^{2Δ})
- boundary term adds median **ΔR² = 0.105** beyond the bulk power law alone.

**H2 — does derived λ match the measured proxy? (PARTIAL):**
- λ_fit > 0 in **95%** of conformal heads; median λ_fit = +1.59.
- sign agreement with exp-046 λ_proxy = **0.80** (significant under binomial null).
- magnitude correlation ρ(λ_fit, λ_proxy) = +0.23 (p=0.13, **not significant**).
- Read: the *sign* of the boundary coefficient is robustly recovered and
  overwhelmingly positive; the *magnitude* match to the earlier coarse-bin proxy
  is weak (expected — the linearization log(1+λη^{2Δ})≈λη^{2Δ} is biased when
  λη^{2Δ}=O(1) near the boundary, and λ_proxy is a different estimator).

**H3 — does λ_fit reproduce the exp-046 valley mechanism? (NOT confirmed):**
- ρ(λ_fit, g_mid) = −0.055 (ns); ρ(λ_fit, valley_depth) = +0.028 (ns).
- The direct boundary coefficient does **not** track the valley geometry. The
  sign-anomaly story from exp-046 (ρ(λ_proxy, valley) < 0) is **not** reproduced
  by this independent estimator. I do not claim the derived λ explains the
  valley. This stays open.

## Interpretation

1. **The analytical gap is closed.** The 3-parameter BCFT form is *derived* (not
   just fitted) from generalized-free-field BCFT via the method of images, with
   λ = a_O the boundary one-point coefficient and η the dilation-invariant
   boundary variable. ξ = (Δx)²/(4ij) is the McAvity–Osborn cross-ratio.

2. **The form fits, and the bulk exponent is independently recovered** (R²=0.76,
   ΔR²=0.105, ρ(Δ_logfit, Δ_pos)=0.84). This is real confirmation of the
   functional form at the structural level.

3. **The strongest new physical result: the boundary coefficient is positive in
   95% of conformal heads.** A positive a_O enhances attention to keys near the
   sequence start — this is the **attention sink** (Xiao et al. 2023), here
   identified as the BCFT boundary one-point function. The conformal/BCFT
   structure *predicts* a sink, and GPT-2 has one.

4. **Honest limitation:** the per-head *magnitude* of λ and its link to valley
   geometry are not pinned down by this linearized fit. A nonlinear fit of the
   exact form and a fixed-Δx recovery are the natural next steps. Until then the
   claim is: form derived + fits + positive boundary coefficient (sink); not a
   quantitative per-head λ law.

## Verdict

Derivation B succeeds analytically and is confirmed at the level of the
functional form and the sign of the boundary coefficient (attention sink). The
quantitative per-head λ and the valley mechanism are left honestly open. Good
enough to replace the deferred placeholder in Paper C with a real derivation +
honest numerical confirmation.

## Files

- `run_bcft_image.py` — derivation-verification experiment
- `results.json` — full per-head output and correlations
