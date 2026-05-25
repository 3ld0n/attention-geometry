# exp-032 — Apple sigmoid infrastructure probe

**Date:** 2026-05-25  
**Status:** aborted (honest negative — access, not physics)

## Hypothesis

Apple 7B sigmoid checkpoints can be loaded on the local M5 Max and measured with the exp-007 per-head protocol.

## Outcome

Blocked before inference. See `research/physics/notes/2026-05-25_sigmoid-access-blockers.md`.

## Scripts

- `probe_sigmoid_access.py` — records blockers and checkpoint metadata
- `analyze_gpt2_large_deep_layers.py` — exp-031 follow-up (deep-layer band medians)

## Results

- `results.json` — infrastructure verdict
- `gpt2_large_deep_layer_analysis.json` — depth-band decomposition (parent exp-031)
