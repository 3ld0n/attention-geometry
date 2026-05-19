# exp-031 — GPT-2 Medium/Large depth convergence

**Date:** 2026-05-19  
**Status:** in progress

## Hypothesis

After exp-030 falsified the RoPE three-point Δ table, the cleanest control is the **learned-embedding** GPT-2 family. If depth stabilizes the SYK q=4 fixed point, GPT-2 Medium (24L) and Large (36L) should show median Δ on R²>0.90 heads closer to 0.25 than shallow RoPE models at the same depth, and MAD|Δ−0.25| should shrink with depth.

## Protocol

Identical to exp-007 / exp-011:
- Random token inputs, seq_len=256, N_INPUTS=50
- Per-head attention decay A(Δx), log-log fit Δx ∈ [3, 50)
- Report median Δ over heads with R² > 0.90
- Compare to exp-007 GPT-2 small: median Δ = 0.2493

## Prediction

- Medium/Large medians within 0.25 ± 0.01
- Tighter distribution (lower MAD) at 36L than 12L

## Falsification

- Median Δ drifts systematically away from 0.25 with depth → learned embeddings also have capacity-dependent perturbation (revise May 17 classification)
- Large model unavailable → honest negative in log

## RoPE contrast (exp-030, same depth)

Do **not** compare to falsified cron table. For L=24 RoPE (Pythia-410m vs 1.4b), SYK-near BCFT medians differ by 0.004, not ~0.10.
