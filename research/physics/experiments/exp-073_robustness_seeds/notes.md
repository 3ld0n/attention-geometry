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

→ batched inference is faithful; the seed sweep is trustworthy.

### Metric correction (important): track MIDDLE accuracy, not V_task

V_task = 1 − middle/max(primacy, recency) is a *ratio*, so it moves when either the
middle OR the reference edges move. On seed 7 this bit hard: the κ=1.5 "deepening"
edit left the middle bin unchanged (11/40 → 11/40) but lowered the *edges*
(0.75/0.80 → 0.725/0.75), so V_task dropped — looking like an anti-deepening effect
that was really just off-target edge degradation. **The direct, honest quantity is
the middle-bin accuracy.** Re-reading both completed seeds on Δ(middle correct):

| seed | base mid | shallow κ=0.5 Δmid | deepen κ=1.5 Δmid | κ=1.5 edge effect |
|------|----------|--------------------|--------------------|-------------------|
| 42   | 13/40    | **+1** (predicted) | **−2** (predicted) | edges unchanged — clean |
| 7    | 11/40    | **+1** (predicted) | **0** (no effect)  | edges dropped 0.75/0.80→0.725/0.75 |

Sham-head κ=1.5 did not deepen either seed (specificity holds: seed42 mid 13→14,
seed7 11→13).

### Honest read at n=2 (awaiting n=5 from seeds 123/2024/99)

- **Effect sizes are at the noise floor**: ±1–2 items out of 40 per position.
- **Shallowing (κ=0.5)** is the *consistent* leg so far: +1 middle item on both seeds.
- **Deepening (κ=1.5)** is *seed-dependent*: clean on seed 42 (which IS exp-072),
  null-on-middle + edge-damage on seed 7. This directly tempers exp-072's "strong
  bidirectional control" headline — it did not cleanly transfer to a second seed.
- This is the rounding-up the carry-forward warned against ("don't round 'deepening
  propagates' up to 'we causally control LITM'"). The robustness check is earning
  its keep by catching it.

### FINAL status this session: n=2 complete, n=5 BLOCKED on Modal billing limit

The seed sweep is **incomplete**. Seeds 42 and 7 completed and are aggregated
(`results.json`, `seeds_42_7.json`). The remaining seeds 123/2024/99 were attempted
three ways and all were defeated by infrastructure, the last fatally:

1. Serial detached run — client ConnectionError; orphan container survived but later
   reaped mid-seed-123 (ephemeral apps die when their client connection ends).
2. `modal run ::spawn` per-seed — ephemeral app reaped the spawned calls on entrypoint exit.
3. `modal deploy` (persistent app) + spawn — **correct pattern, and it worked**: all
   three seeds ran concurrently and got past κ=1.0, but then hit
   *"waiting to be scheduled on GPU_A100_80GB … memory=128.8GiB"* — the 128 GB host-RAM
   request forced scarce oversized workers (preemption; restart-from-scratch with
   retries=0). Dropped memory to 64 GB and went to redeploy →
   **"workspace billing cycle spend limit reached."** Hard stop.

**Root-cause lessons for the resumption:**
- Use `modal deploy` + spawn-against-deployed-function (see `invoke_seeds.py`), NOT
  `modal run`/detach — that's the connection-robust pattern.
- `memory=131072` (128 GB) was self-inflicted scarcity; the model needs ~52 GB host.
  Now set to 64 GB in both functions. This likely also caused tonight's preemption.
- Resume needs Eldon: raise the Modal spend limit or wait for the billing cycle reset.
  Then: `modal deploy robustness_sweep.py && python invoke_seeds.py 123,2024,99`,
  poll `/exp073_seed<seed>.json`, then `python aggregate.py` for n=5.

**Base valleys the blocked seeds DID produce (base + κ=1.0 only, from deployed logs):**
| seed | base acc-by-pos | base middle | κ=1.0 V |
|------|-----------------|-------------|---------|
| 123  | [0.75, 0.6, 0.35, 0.475, 0.8]   | 14/40 | 0.5625 |
| 2024 | [0.775, 0.725, 0.325, 0.575, 0.8] | 13/40 | 0.5938 |
| 99   | [0.8, 0.675, 0.35, 0.525, 0.8]  | 14/40 | 0.5625 |

→ the LITM valley itself is robust across all 5 seeds (middle 11–14/40, edges ~30–32/40).
What we lack for 123/2024/99 are the EDIT deltas (κ=0.5/1.5), so the robustness verdict
stands at **n=2**.

### Verdict (n=2 — PRELIMINARY, underpowered)
- **Shallowing (κ=0.5):** Δmiddle = +1, +1 (mean +1.0, 2/2 predicted sign). Consistent
  but tiny (~1/40).
- **Deepening (κ=1.5):** Δmiddle = −2, 0 (mean −1.0 ± 1.0, 1/2 predicted sign).
  Seed-dependent; on seed 7 it didn't move the middle and instead degraded the edges.
- **Headline:** at n=2 the bidirectional claim does NOT cleanly survive a single new
  seed. Shallowing looks like the steadier (if small) leg; deepening is the fragile one
  — the inverse of exp-072's framing. Effects sit at the per-item noise floor; n=5 with
  proper error bars is required before any claim. Do NOT round this up.
