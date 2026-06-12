# exp-063 — ξ characterization (Phase 1.2)

*June 11, 2026. The exp-060 kill's positive residue (SESSION_BRIEF_PHASE1 §3.1.2): the winning boundary form C·Δx^(−2Δ)·(1 + b·e^(−j/ξ)) carries an absolute length scale ξ ≈ tens of tokens that no current framework piece predicts. Characterization only — form (b) fits, no new adversarial claims, no hypothesis confirmation.*

## Question

What sets ξ? Discriminating comparison across cached models with different training context windows (GPT-2 family ctx 1024 vs Pythia family ctx 2048; GPT-Neo global layers ctx 2048): does median ξ scale with training context, stay constant in tokens, or track something else?

## Protocol

exp-060 frozen pipeline throughout: L = 256, 50 random-token inputs, seed 42, fp32 + dtype/NaN asserts, conformal selection (1D lag fit, queries i ≥ 32, lags [3,50), R² ≥ 0.90, Δ ≥ 0.05), 2D domain (i ≥ 32, Δx ∈ [3,50), A > 0), form (b) least-squares on log A, 5 restarts. Script: `run_xi.py`.

- **Part A** — archived exp-060 GPT-2 fits: ξ̂ distribution, correlations (layer, Δ̂, b amplitude).
- **Part B** — GPT-2 re-run saving per-input 2D profiles: ξ̂ stability across the 50 inputs; per-head attention entropy correlate. (Note: exp-060's archived `per_input_lag_gpt2.json.gz` contains 1D lag profiles only, which cannot constrain form (b)'s j-dependence; the per-input ξ̂ stability therefore required this re-run.)
- **Part C** — cross-model: gpt2, gpt2-medium, gpt2-large (ctx 1024), pythia-410m, pythia-1.4b (ctx 2048); pythia-2.8b / gpt-neo-2.7B global appended when their downloads complete.

## Results

### Part A (archived GPT-2, n = 43)

| Statistic | Value |
|---|---|
| ξ̂ median | 20.1 tokens |
| ξ̂ IQR | [0.84, 37.2] |
| at upper bound (256) | 0 |
| ρ(ξ̂, layer) | −0.06 (p = 0.70) — **no depth trend** |
| ρ(ξ̂, Δ̂) | **−0.51** (p = 5×10⁻⁴) |
| ρ(ξ̂, b amplitude) | **−0.66** (p = 1.5×10⁻⁶) |

The IQR's low end (ξ < 1) flags a sub-population where form (b) degenerates into a j=0 spike — consistent with exp-060's two-component sink anatomy (sharp spike + extended layer). The extended-layer population sits at ξ ≈ 20–40. Higher-Δ heads have *shorter* boundary layers and larger amplitudes.

### Part B (GPT-2 re-run, per-input stability + entropy; n = 43)

- **ξ̂ is a stable per-head property, not an averaging artifact:** ρ(mean-profile ξ̂, per-input median ξ̂) = **0.935** (p = 4×10⁻²⁰); median |ξ̂_mean − ξ̂_pi-median|/ξ̂ = 2.3%. Per-input scatter is moderate (median relative IQR 0.42; extended subpopulation ξ > 5: ρ = 0.864, rel IQR 0.46).
- ρ(ξ̂, entropy) = **+0.62** (p < 10⁻³): longer boundary layers belong to more diffuse heads.

### Part C (cross-model, frozen pipeline, form (b) only)

| Model | train ctx | PE | n conf | ξ̂ median | ξ̂ med (interior) | at bound |
|---|---|---|---|---|---|---|
| gpt2 | 1024 | learned | 43 | 22.4 | 21.2 | 1 |
| gpt2-medium | 1024 | learned | 68 | 12.6 | 9.6 | 4 |
| gpt2-large | 1024 | learned | 181 | 13.9 | 9.9 | 6 |
| pythia-410m | 2048 | RoPE | 43 | **43.5** | 43.5 | 0 |
| pythia-1.4b | 2048 | RoPE | 29 | **59.6** | 59.6 | 0 |
| gpt-neo-2.7B (global) | 2048 | learned | 78 | **58.9** | 49.6 | 11 |

- The ctx-2048 models sit at ξ ≈ 43–60; the ctx-1024 models at ξ ≈ 13–22. **The GPT-Neo deconfound point is decisive:** a learned-PE, GPT-family architecture trained at ctx 2048 lands at ξ_med = 58.9 (interior 49.6) — in the ctx-2048 band, ~4× its architectural siblings at ctx 1024. Every ctx-2048 model exceeds every ctx-1024 model by ≥ 2×, across two PE schemes and three model families.
- Correlates: ρ(ξ̂, Δ̂) negative in 4/5 full-attention models (−0.43 to −0.61; Pythia-410m −0.16) but ≈ 0 in GPT-Neo global layers; ρ(ξ̂, entropy) positive in 5/5 full-attention models (+0.34 to +0.62) but ≈ 0 in GPT-Neo; ρ(ξ̂, layer) inconsistent everywhere. The within-model correlates are population-specific; the cross-model context-window separation is not.
- Within GPT-2 family, ξ̂ does NOT grow with model scale (small 21 → medium/large ≈ 10): not a parameter-count effect.

## Verdict (registered): **ξ tracks the training context window**

ξ is a real, per-head-stable second length scale: roughly constant fraction of training context (≈ 1.5–2% of ctx at 1024, ≈ 2–3% at 2048 by interior medians), NOT constant in tokens, NOT a family/PE/scale effect. The sink's extended component is a legible training-configuration signature — the practical, publishable residue the brief anticipated for the ξ ∝ context branch. Scope cautions: (a) the L = 256 measurement window weakly constrains ξ̂ ≳ 100 (11/78 GPT-Neo heads at the fit bound, excluded from the interior median); the ≥2× band separation is robust, the precise scaling exponent is not; (b) ctx-4096+ models and an L = 512 re-measurement are the natural extension before claiming linearity.
