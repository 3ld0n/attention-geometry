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

## Run log (June 16, 2026 evening)

First launch attempts hit a recurring set of infrastructure issues, none of them
scientific. Documented so the next session does not rediscover them:

1. **CUDA OOM at batch 8** (twice). vicuna-13b fp32 (~52 GB weights) + eager
   attention on ~1945-token prompts: at batch 8 the per-layer activations +
   attention matmul peak hit ~74 GB allocated and the 4.84 GB attn-weights alloc
   overflowed the 80 GB card. Fixes applied to the script: set
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (kills the ~6 GB
   fragmentation waste), `torch.cuda.empty_cache()` after the 50 projector passes,
   and **batch_size=4 is the proven-safe ceiling** (batch 8 OOMs, batch 4 fits with
   ~16 GB headroom). bf16/TF32 would be far faster but were rejected: they change
   numerics and would confound the seed-42 reproduction gate, which must match
   exp-072's fp32 result. fp32 is slow (~20+ min/seed) but correct and comparable.
2. **Orphan ephemeral apps** spawned on several launches (2–3 apps per `modal run`).
   Root cause: flaky local→Modal client connection causing app-creation retries.
   Mitigation: after each launch, `modal app list | grep ephemeral` and stop any
   orphan beyond the tracked app ID (from the terminal "View run" URL).
3. **Client heartbeat ConnectionError** ("Deadline exceeded") mid-run — the local
   client lost its connection to Modal's control plane. The remote container kept
   running, but a sustained drop on a ~2 h run would risk losing everything.

**Robustness fixes for the real run:** function `timeout=14400` (4 h); results
written **server-side to the volume** (`/cache/exp073_results.json` full,
`/cache/exp073_partial.json` after each seed) and `vol.commit()`-ed, so completed
seeds survive any client disconnect; per-condition `V_task` printed to stdout as a
second reconstruction path; launched with `modal run --detach` so the app finishes
server-side independent of the client.

Cost note: Eldon explicitly relaxed the cost constraint ("focus on the data you
want; a few dollars on a wrong turn is fine"). The earlier compromise to 3 seeds
was reverted — the full {42,7,123,2024,99} sweep is the data worth having for a CI
on the shallowing leg.

**Detached run launched:** app `ap-uY09mhL4OOLv2fgNulYtxC`, fp32, batch 4, 5 seeds,
A100-80GB. Retrieve when done: `modal volume get hf-model-cache /exp073_results.json results.json`.

## Result

**Seed-42 reproduction gate: PASSED** (the numerical-equivalence check on the
batched-inference change). Batched fp32 reproduces exp-072's unbatched fp32 to the
4th decimal:

| condition | exp-073 batched | exp-072 locked |
|-----------|-----------------|----------------|
| base acc-by-pos | [0.775, 0.65, 0.325, 0.525, 0.8] | [0.775, 0.65, 0.325, 0.525, 0.8] |
| target κ=1.0 (T-D ref) | V=0.5938 | 0.594 |
| target κ=0.5 (shallow) | V=0.5758 | 0.576 |
| target κ=1.5 (deepen)  | V=0.6562 | 0.656 |

→ batched inference is faithful; the seed sweep is trustworthy. Seed 42 also shows
both legs in the predicted direction (κ=0.5 shallows V by −0.018, κ=1.5 deepens by
+0.062), as expected since seed 42 *is* exp-072. Error bars from seeds 7/123/2024/99
*(pending — ~13 min/condition fp32; expect ~3 seeds complete within the 4 h window,
banked incrementally to volume:/exp073_partial.json)*.
