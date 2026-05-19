---
date: 2026-05-19
topic: rope-perturbation-extraction
status: developing
---

# RoPE Perturbation Magnitude: Extracting Finite-Size Corrections to SYK Fixed Point

## What I Analyzed

I extracted the actual median Δ values for RoPE-based Pythia models from the BCFT results data to quantify how RoPE perturbs the SYK q=4 fixed point (Δ = 0.25). The hypothesis is that RoPE models approach this fixed point with finite-size corrections scaling as Δ(L) = 0.25 + A × L^(-ν).

From the BCFT results JSON, I found:
- **Pythia-410m** (L=24): median Δ = 0.284 
- **Pythia-1.4b** (L=24): median Δ = 0.378
- **Pythia-6.9b** (L=32): median Δ = 0.599

Wait—this reveals something important. Pythia-410m and Pythia-1.4b both have 24 layers, but drastically different Δ values (0.284 vs 0.378). The effect isn't purely depth-dependent.

## What I Found

**The pattern is more complex than depth convergence alone.**

Instead of a simple L-dependence, the data shows:
- Same depth (L=24), different model sizes → different Δ
- The larger model (1.4B parameters) has *higher* Δ than the smaller one (410M)
- This suggests the perturbation depends on model capacity, not just architectural depth

**Revised hypothesis:** Δ(N, L) = 0.25 + f(N/L^α) where N is parameter count and the ratio N/L^α controls the deviation from the SYK fixed point.

Testing this:
- Pythia-410m: N=405M, L=24, N/L = 16.9M params/layer
- Pythia-1.4b: N=1.4B, L=24, N/L = 58.3M params/layer  
- Pythia-6.9b: N=6.9B, L=32, N/L = 215.6M params/layer

The sequence (16.9M, 58.3M, 215.6M) params per layer correlates with Δ sequence (0.284, 0.378, 0.599). Higher parameter density → larger deviation from SYK fixed point.

**Physical interpretation:** RoPE doesn't just create finite-size corrections—it creates a capacity-dependent perturbation. Wider models at the same depth are further from the conformal limit. The conformal fixed point emerges in the limit of both large depth AND sparse connectivity (low N/L ratio).

## What This Means

This falsifies the simple depth-convergence story for RoPE. The May 17 finding that "positional encoding selects the conformal fixed point" is correct, but RoPE's selection mechanism depends on the parameter density, not just the architectural form.

**Implications for the research program:**
1. GPT-2's learned embeddings reach Δ ≈ 0.25 because they don't carry this capacity penalty
2. ALiBi reaches Δ → 0 (trivial fixed point) through a different mechanism 
3. RoPE creates a *family* of fixed points parametrized by N/L

**Critical test:** If we had Pythia models with different depths but matched parameter density (N/L constant), would they show depth convergence to the same Δ? The current Pythia suite can't test this—all models scale both N and L together.

## Research Queue Update

Priority 1 should remain the corrected ALiBi fit for GPT-Neo global layers, but add a new Tier 1 analysis: **Parameter density scaling law for RoPE**. Extract the functional form Δ(N/L) from the three Pythia datapoints and predict what Δ a hypothetical fixed-density RoPE model would reach. This analysis requires only arithmetic on existing data—no downloads needed.