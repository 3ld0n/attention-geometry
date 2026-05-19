# exp-030 — RoPE Δ BCFT verification

**Date:** 2026-05-19  
**Hypothesis tested:** May 18–19 cron notes correctly extracted Pythia median Δ from BCFT pre-registered JSON; same-depth Pythia-410m and 1.4b have substantially different Δ.

**Data:** `exp-025` BCFT JSONs (`bcft_pre_registered_run_2026-04-17T092056Z.json` for 410m; `092239Z` for 1.4b). Script: `verify_pythia_delta_bcft.py`. Output: `results.json`.

## Verdict

**Falsified** (cron extraction narrative). The claimed same-depth split and 6.9b number are not supported by the files the queue named.

## Key numbers (BCFT head-level Δ)

| Model | Filter | n heads | median Δ |
|-------|--------|---------|----------|
| pythia-410m | R2>0.90 (March-style) | 36 | 0.361 |
| pythia-410m | SYK-near (R2≥0.85, \|Δ−0.25\|≤0.05) | 17 | 0.253 |
| pythia-1.4b | R2>0.90 | 46 | 0.327 |
| pythia-1.4b | SYK-near | 22 | 0.248 |

Cron asserted 410m ≈ 0.28, 1.4b ≈ 0.38, 6.9b ≈ 0.60.

## Findings

1. **Pythia-6.9b is not in BCFT JSON.** Δ≈0.60 is March exp-011 **Pythia-70m** (6 layers), not a 32-layer model.
2. **410m cron value ≈ March 410m (0.28), not BCFT file (≈0.36 at R2>0.90).** Mixed protocols.
3. **1.4b cron value ≈ March 160m (0.38 at 12 layers), not 1.4b.** Likely 160m ↔ 1.4b name confusion.
4. **410m BCFT file incomplete:** 239/384 heads, layers 0–14 only.
5. **Same-depth different-Δ:** Under SYK-near filter, 410m and 1.4b medians differ by **0.004**, not ~0.10. The dramatic split was an extraction artifact.

## What stays true

- May 17 positional-encoding-selects-attractor pattern (learned / RoPE / ALiBi) is unchanged.
- March exp-011 depth convergence (70m→160m→410m: 0.60→0.38→0.28) remains valid **for that protocol** — do not merge with BCFT medians without relabeling filters.
- RoPE models show Δ > 0.25 on BCFT conformal subset; magnitude and scaling law remain open, need clean re-measurement.

## Next

Tier 1: GPT-2 Medium/Large depth test (learned-embedding control). Do not fit Δ(N/L) from the three cron numbers — they are not a coherent dataset.
