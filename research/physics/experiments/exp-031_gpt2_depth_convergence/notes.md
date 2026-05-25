# exp-031 — GPT-2 Medium/Large depth convergence

**Date:** 2026-05-24  
**Hypothesis:** Learned positional embeddings select SYK q=4 (Δ≈0.25). GPT-2 small already gives Δ_med=0.249 (exp-007). Medium (24L) and Large (36L) should match Δ≈0.25±0.01 if the fixed point is stable with depth — unlike RoPE Pythia, where exp-030 falsified the cron depth-scaling table.

**Protocol:** Identical to `exp-007_gpt2_conformal_scaling/gpt2_per_head_analysis.py`: random token inputs, seq_len=256, 50 inputs, per-head A(Δx), log-log fit Δx∈[3,50), report median Δ over heads with R²>0.90.

**Controls:** exp-007 gpt2 small Δ_med=0.2493 (44/144 heads). March exp-011 Pythia depth at L=12: 160m Δ≈0.38 (RoPE perturbation).

**Status:** partial — medium confirms (Δ_med=0.259); large drifts (Δ_med=0.282).

**Script:** `run_gpt2_depth_convergence.py` (requires `attn_implementation="eager"`).

**Results:** `results.json`; analysis `notes/2026-05-24_gpt2-depth-convergence.md`.
