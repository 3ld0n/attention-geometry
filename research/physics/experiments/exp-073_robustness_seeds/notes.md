# exp-073 — Robustness of the QK-slope behavioral handle (seed sweep)

*Status: SPECCED, not yet run. Opened June 16, 2026 (Cursor session, Ariel + Eldon, evening).*
*Program: `research/physics/PROGRAM_BRIEF_LITM_CAUSAL_HANDLE.md` (Phase 1).*

## Question
How much of the exp-072 registered verdict is seed noise? exp-072's bidirectional claim turned on
a **single item out of 40** (κ=0.5 shallowing: middle 0.325→0.350, V 0.594→0.576). This puts error
bars on both legs by repeating the κ-sweep across multiple task seeds.

## Design (reuses exp-072's LOCKED config — no new census)
- Model + heads + projectors + n_doc=40 + positions: read verbatim from
  `../exp-072_cloud_powered_slope_editing/prereg_locked.json` (commit e6fbaecf).
- Task seeds: **{42, 7, 123, 2024, 99}**. Seed 42 must reproduce exp-072 exactly (numerical-
  equivalence check on the batching change).
- Conditions per seed: κ=1.0 (T-D ref), κ=0.5, κ=1.5, κ=2.0 on target heads; κ=1.5 on sham heads.
- fp32, comparable to exp-072. **Batched** forward passes (left-pad + attention_mask) — the cost
  fix. If batched seed-42 diverges from exp-072, fall back to unbatched for the check and document.

## Outputs
- Per-condition V_task and per-bin accuracy as **mean ± SE across seeds**.
- Distribution of the κ=0.5 shallowing delta (stable sign vs. within seed noise).
- T-C Spearman ρ distribution.

## Verdict logic
Shallowing leg *robust* iff V_task(κ=0.5) < V_task(κ=1.0) across a clear majority of seeds (ideally
the delta CI excludes 0). Deepening leg expected comfortably stable. Report honestly either way.

## Run
```
cd research/physics/experiments/exp-073_robustness_seeds
.venv/bin/python -m modal run robustness_sweep.py
```
Reads the exp-072 lock automatically. One A100 job; watch the bill (batching should make it cheaper
than exp-072's ~$5–7). Writes `results.json`.

## Result
*(pending)*
