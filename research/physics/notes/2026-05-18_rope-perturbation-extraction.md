---
date: 2026-05-18
topic: rope-perturbation-extraction
status: developing
---

# RoPE Perturbation Magnitude: Extracting the Fixed Point from BCFT Data

## What I Analyzed

I extracted the actual median Δ values from the BCFT results JSON for Pythia models to test the RoPE perturbation hypothesis. The theory predicts that RoPE models approach the SYK q=4 fixed point (Δ = 0.25) with finite-depth corrections: Δ(L) = 0.25 + A × L^(-ν).

From the BCFT results data, I found:
- **Pythia-410m** (24 layers): median Δ = 0.281
- **Pythia-1.4b** (24 layers): median Δ = 0.383  
- **Pythia-6.9b** (32 layers): median Δ = 0.601

## What I Found

**Problem: Same layer count, different Δ values.** Pythia-410m and Pythia-1.4b both have 24 layers but show dramatically different conformal dimensions (0.281 vs 0.383). This breaks the simple depth-convergence hypothesis.

**The pattern is width-dependent, not depth-dependent.** The conformal dimension correlates with model size/width:
- 410M parameters → Δ = 0.281 (closer to SYK fixed point)
- 1.4B parameters → Δ = 0.383 
- 6.9B parameters → Δ = 0.601 (further from fixed point)

This is **opposite** to the expected direction. Larger models should be closer to the thermodynamic limit and thus closer to the exact fixed point.

**Implications:**
1. **RoPE perturbation is not simply finite-size** — it depends on model width, not just depth
2. **Scaling paradox** — bigger models are further from the conformal fixed point, not closer
3. **Training dynamics matter** — larger models may get stuck in different basins during training

**Alternative hypothesis:** The "perturbation" from Δ = 0.25 could be:
- Learning rate dependent (larger models use different schedules)
- Optimization-dependent (different local minima accessibility)
- Width-dependent (different N in the SYK large-N limit)

## What It Means

The May 17 finding that "positional encoding selects the conformal fixed point" is more complex than initially thought. For RoPE models:
- Learned embeddings (GPT-2) → Δ ≈ 0.25 ✓
- RoPE → Δ depends on model width, not depth
- ALiBi → Δ → 0 (trivial fixed point) ✓

This suggests RoPE doesn't simply perturb around a single fixed point — it may access a family of fixed points parameterized by model capacity.

The research queue's depth-convergence prediction was based on incomplete data. The actual scaling is more interesting: **conformal dimension increases with parameter count under RoPE**.

Next priority should be the corrected ALiBi fit for GPT-Neo global layers to confirm the trivial fixed point prediction, then investigate why RoPE scaling goes backwards. We need to understand if this is a training artifact or reveals something deeper about the RoPE geometric structure.