# GPT-2 depth convergence (exp-031)

**Date:** 2026-05-24  
**Question:** After exp-030 falsified the RoPE three-point Δ table, does the learned-embedding control (GPT-2 family) hold SYK q=4 (Δ≈0.25) as depth increases?

## Protocol

Same as exp-007: random tokens, seq_len=256, 50 inputs, per-head A(Δx), log-log fit on Δx∈[3,50), median Δ over heads with R²>0.90. `transformers` 5.8.1 requires `attn_implementation="eager"` on `GPT2LMHeadModel` or attentions are not returned.

## Results

| Model | Layers | Heads (R²>0.90) | Median Δ | \|Δ−0.25\| | Verdict |
|-------|--------|-----------------|----------|-----------|---------|
| gpt2 (exp-007) | 12 | 44/144 | **0.2493** | 0.001 | SYK q=4 |
| gpt2-medium | 24 | 68/384 | **0.2589** | 0.009 | within 0.01 |
| gpt2-large | 36 | 176/720 | **0.2819** | 0.032 | near, not within 0.01 |

Raw: `experiments/exp-031_gpt2_depth_convergence/results.json`, archive `results/exp-031_gpt2_depth_convergence_*.json`.

## Interpretation

**Partial confirmation of the May 17 learned-embedding classification.** Medium (24L) stays on the SYK q=4 attractor within measurement precision — the fixed point is stable at 2× depth. Large (36L) drifts upward to Δ_med≈0.28: still far from RoPE’s shallow perturbation (Pythia-160m L=12 ≈0.38 in exp-011), but **not** the pre-registered “tighter convergence with depth.” Capacity at the largest GPT-2 scale adds a modest positive Δ bias, analogous in magnitude to the RoPE perturbation band (0.28–0.38) but on learned embeddings.

**Layer structure (large):** Global median is pulled up by mid-network heads with Δ≫0.25; deep-layer medians in the top quartile of layers are lower (~0.16). The SYK-near head count scales with total heads (16/384 medium, 39/720 large) — more conformal heads, not a cleaner fixed point.

## Comparison to RoPE at same depth

Under exp-030’s SYK-near filter (R²≥0.85, \|Δ−0.25\|≤0.05), Pythia-410m and 1.4b at L=24 gave medians 0.253 vs 0.248 — already near 0.25, not 0.28–0.38. The cron table was wrong; **same-depth RoPE vs learned is not decided by this experiment alone.** What exp-031 establishes: learned embeddings do not require depth to *reach* 0.25 (medium already there); large scale introduces a **parameter-count** perturbation, not a depth monotonic flow.

## Next experiment

1. **Per-layer deep median** on large — test whether layers 25–36 alone match 0.25±0.01 (subset median), or whether the 0.28 global median is intrinsic.
2. **Apple sigmoid** (queue Tier 1 #3) — falsification test for softmax-independence.
3. Do **not** refit Δ(N/L) across GPT-2 sizes until a single run script logs all three in one JSON (avoid overwrite bug).
