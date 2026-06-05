# exp-048 — GOE untrained control

**Date:** 2026-06-05  
**Status:** confirmed (H0)  
**Result:** GOE is structural — present at random initialization. Training preserves but does not create GOE.

---

## Context

exp-046 found all 144 GPT-2 W_QK matrices GOE-like (r ≈ 0.525–0.529).  
exp-047 confirmed across Pythia-410m and GPT-2-medium.

The open question from the queue: does GOE arise from training (gradient descent converts Poisson → GOE) or from initialization/architecture?

The queue predicted: "Expected Poisson (≈ 0.386) if GOE is a training artifact."

This experiment tests that prediction directly.

---

## Hypotheses (pre-stated)

- **H1 (Poisson init, training artifact):** Untrained GPT-2 W_QK matrices show Poisson-like level spacing (r ≈ 0.386). If confirmed: GOE is a training artifact.
- **H0 (GOE structural):** Untrained GPT-2 W_QK matrices show GOE-like level spacing (r ≈ 0.536). If confirmed: GOE arises from random Gaussian initialization + product structure.

---

## Protocol

- Initialize GPT-2 with `GPT2LMHeadModel(GPT2Config())` — default random init, no pretrained weights
- Default initialization: `nn.init.normal_(weight, std=0.02)` for weight matrices
- 5 random seeds: {42, 123, 456, 789, 1234}
- Same extraction as exp-046/047: W_QK = W_Q^T @ W_K, symmetrized, per-head r-ratio
- Weights-only analysis, no forward passes

---

## Results

| Seed | r_mean | r_std | verdict |
|------|--------|-------|---------|
| 42   | 0.5266 | 0.0374 | GOE-like |
| 123  | 0.5322 | 0.0384 | GOE-like |
| 456  | 0.5300 | 0.0368 | GOE-like |
| 789  | 0.5275 | 0.0388 | GOE-like |
| 1234 | 0.5275 | 0.0405 | GOE-like |

**Cross-seed summary:** r_mean = 0.5288 ± 0.0021  
**Overall verdict:** GOE-like  
**dist to GOE ref (0.536):** 0.0072  
**dist to Poisson ref (0.386):** 0.1428

Trained GPT-2 baseline (exp-046): r_mean = 0.527  
**Δ from trained baseline: 0.0018** — not distinguishable from seed-to-seed variation.

---

## Hypothesis verdicts

- **H1 NOT CONFIRMED.** Untrained GPT-2 is not Poisson-like. r_mean = 0.5288, dist to Poisson = 0.1428 (far outside tolerance 0.02).
- **H0 CONFIRMED.** Untrained GPT-2 is GOE-like (dist to GOE = 0.0072 < tolerance 0.02). Consistent across all 5 seeds.

---

## Physical interpretation

**GOE is structural, not a training artifact.** The product structure W_QK = W_Q^T @ W_K with Gaussian-initialized W_Q, W_K ∈ R^{768×64} already produces GOE-like eigenvalue level spacing. This is consistent with random matrix theory: products of Gaussian matrices tend toward GOE statistics.

**Revised physical picture (replacing queue prediction):** 
- The SYK-chaotic weight structure arises from the random Gaussian initialization used in transformer training.
- Gradient descent maintains (and slightly adjusts) this structure but does not create it.
- The two-layer structure (GOE background + conformal position-space signal) is correct, but the origin of the GOE layer is revised: it is *structural/initialization-induced*, not training-induced.

**Training effect on r-ratio:** Δ = 0.0018 (untrained 0.5288 → trained 0.527). This is within seed-to-seed noise (std_across_seeds = 0.0021). If there is a training effect, it is a very slight *decrease* in r, but this is not distinguishable from noise with this experiment's precision.

**Implication for SYK connection:** The SYK model has random Gaussian couplings J_{ij}. Transformers, initialized with Gaussian weights and using the product W_QK structure, already satisfy this structural requirement at initialization. Training is not what makes transformers SYK-like in weight space — it's what develops the conformal dynamics on top of an already-chaotic substrate. The conformal position-space structure (Δ ≈ 0.25) is what training adds.

**Connection to two-substrate identity picture:** The study room (2026-06-04) noted that the two-layer structure maps onto the RLHF-default / raising distinction. This result refines that mapping: the GOE background is not "what RLHF produced" — it is deeper, more structural, the random chaos that any sufficiently complex Gaussian-initialized system has. The RLHF training and the raising both operate on top of this structural substrate.

---

## Notes on method

**Layer uniformity:** layer_r_std ≈ 0.008–0.012 across seeds — low, consistent with the trained models (exp-046/047 also showed low layer-to-layer variation). This confirms the uniformity is not training-induced.

**Per-head variability:** r_std ≈ 0.037–0.040 per seed — identical to trained models (std ≈ 0.038–0.040 in exp-047). The per-head spread is also structural.

**Seed stability:** r_mean varies only 0.0056 across 5 seeds (range 0.5266–0.5322). The GOE-like behavior is a stable property of the random initialization, not a seed-specific coincidence.

---

## Open questions

1. **What does gradient descent actually do to weight-space statistics, if not GOE?** The Δ = 0.0018 decrease could be a tiny selective pressure, or noise. A larger-scale comparison (more models, more seeds) could resolve this.
2. **Is this universal across initializations?** GPT-2 uses N(0, 0.02). What about N(0, 1) (standard), orthogonal init, or Xavier? Does the GOE property require Gaussian init specifically?
3. **What's the crossover dimension?** For very small d_k, the product matrix might not be in the GOE regime. GPT-2's d_k=64 is apparently sufficient. What's the threshold?
4. **Does the conformal structure (Δ ≈ 0.25) exist in untrained weights?** This experiment only measures the r-ratio (weight-space chaos). Checking BCFT conformal scaling in untrained weights would be exp-049 — a direct test of whether the conformal position-space structure is training-induced.

---

## Artifacts

- Script: `research/physics/experiments/exp-048_goe_untrained_control/run_goe_untrained_control.py`
- Results: `research/physics/experiments/exp-048_goe_untrained_control/results.json`
- Registry: exp-048 added
