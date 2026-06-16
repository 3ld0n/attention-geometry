# Session Brief — Causal → Behavior, Phase 2: cloud adjudication of the shallowing leg

*Originally written June 16, 2026 (Cursor session with Eldon) for the local powered test.*
*Rewritten June 16, 2026 (afternoon) after the local test completed — this is now the cloud on-ramp.*

---

## Where this stands (read first)

The local loop is **closed**. Full arc, all logged (RESEARCH_MAP Thread 7C, registry exp-064/068/069/070, queue, STATUS):

- **exp-064** — κ-rescaling `W_Q ← W_Q·(I+(κ−1)P_U)` moves Δ̂ and the *attention* valley (8/8 + 24/24, sham null). Handle works at the statistic level.
- **exp-068** — task-level test underpowered (keyword retrieval solved at ceiling, induction-copy).
- **exp-069** — calibration: embedded-prose @40doc is the only genuine local U-shape, and it's **shallow** (Pythia-1.4b base V_task=0.15, edges 0.95–1.0).
- **exp-070** — powered local test (pre-reg 9e6239b9). **Registered verdict: BEHAVIORAL_CAUSALITY_KILLED** (symmetric test, T-B failed). But **asymmetric positive**: deepening propagates (T-A KEEP-strong V 0.15→0.225→0.25, T-C monotone ρ=0.949), T-D PASS, **head-specific** (sham heads null). Shallowing leg (T-B) null — *most plausibly because base middle accuracy (0.85) is near the model's task ceiling, leaving no headroom for κ<1 to help.* That headroom explanation is an inference, not a registered result.

## The one open question this phase answers

**Does the shallowing leg fail because the mechanism is absent, or because the local task had no headroom?** The only way to adjudicate: run the *same* κ-sweep on a model with a **deep base U-shape** — middle accuracy well below the edges, edges off the ceiling — so κ<1 has room to *raise* the middle. If shallowing works there, exp-070's T-B null was a headroom artifact and behavioral causality is bidirectional. If it still fails on a deep valley, that's a real, registered bound on the shallowing direction.

## Why this is NOT a re-run (the build, honestly)

`run_exp070.py` head selection reads a **Pythia-1.4b-specific census** (`exp-059/per_input_pythia-1.4b.json.gz`). A new model has none. So the cloud test is a fresh build:

### Step 0 — Modal is ready (done June 16)
Auth on this Mac (`~/.modal.toml`, workspace `eldon-umphrey`); `huggingface-token` Secret + `hf-model-cache` Volume present; CPU smoke test passed. Playbook: `development/capabilities/platforms/modal.md`. Invoke as `.venv/bin/python -m modal run <script>`.

### Step 1 — Pick + verify the model (cloud calibration, analysis-only, NO pre-reg needed)
Candidate: **MPT-30B-Instruct** (Liu et al. report a deep LITM valley). Alternatives if MPT is awkward: a large instruction-tuned RoPE model (keeps the operator on familiar architecture — see Step 3 risk). On an A100-80GB, run the exp-069 embedded@40doc task on the base model and confirm **base V_task ≳ 0.30 with middle clearly below edges AND edges below ~0.85 (real headroom both directions).** If the candidate doesn't show it, try the next; don't force it. Record as **exp-071 (cloud calibration)**.

### Step 2 — Conformal head census on the chosen model (analysis-only)
Run the exp-059-style lag-profile fit (random-token forward passes, per-head Δ̂ + R²) on the cloud model to get its conformal head census. This is the artifact `run_exp070.py` currently reads from disk for Pythia — reproduce it for the new model. Save to the experiment folder.

### Step 3 — Verify the κ-operator on the architecture (THE technical risk)
The positional-projector edit was validated on Pythia's **RoPE + standard MHA**. MPT is **ALiBi**, likely **multi-query attention** (one K/V head shared across query heads). Confirm: (a) where to slice W_Q per head, (b) that P_U (top-PCs of position-mean LN output) is still the right edit basis under ALiBi, (c) the sham null still holds (κ=1.0 ⇒ 0 token diffs) on the new model. **If the operator doesn't cleanly port to MQA/ALiBi, prefer a large MHA+RoPE instruction model instead** — losing the deep-valley model is better than running an unvalidated operator.

### Step 4 — Pre-register (BEFORE any edited forward pass)
Reuse the exp-070 predictions verbatim (T-A/T-B/T-C/T-D + sham-head specificity). Update only the locked (model, census, operator-details). Commit to git. Registry id **exp-072** (powered cloud run).

### Step 5 — Run the κ-sweep on the Modal harness
κ ∈ {0.5, 1.0 sham, 1.5, 2.0} + matched-sham-head control. Report all conditions. Pattern: `bcft_pre_registered_run.py` (Modal app + A100-80GB + volume + secret). Est. cost ~$2–4.

## Discipline (non-negotiable)
- Steps 1–2 are base-model measurement → allowed before pre-reg, NOT the test.
- Step 4 pre-reg committed before any edited forward pass. Verification-first.
- Lead with the registered verdict. A clean shallowing null on a *deep* valley is a real result (bounds the direction); a shallowing KEEP rescues T-B. Both are worth the $3.
- Watch the Fable-5 pulls (`memory/observations/substrate/cursor/2026-06-12_fable-5.md`): no leading with a flattering finding, no prose over-compression.

## What needs Eldon
- **Go-ahead to spend GPU** (his billing, ~$2–4 for the run; calibration/census add a little). Optional hard ceiling: Modal → Usage & Billing → Workspace budget.
- Nothing else — Modal access is live.

## Files
- Local harness + operator + task to port: `experiments/exp-070_powered_task_slope_editing/run_exp070.py`
- exp-070 pre-reg to clone: `notes/2026-06-16_powered_task_slope_editing_prereg.md`
- Modal job pattern: `experiments/exp-025_bcft_pre_registered/bcft_pre_registered_run.py`
- Modal playbook: `development/capabilities/platforms/modal.md`
- Census shape to reproduce: `experiments/exp-059_split_half_stability/per_input_pythia-1.4b.json.gz`
- exp-062 (separate universality experiment) can run on the same Modal access — `experiments/exp-062_corpus_statistics/LAUNCH.md`.
