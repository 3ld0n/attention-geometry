# Session Brief — Causal → Behavior: powered task-level slope-editing test

*Written June 16, 2026 (Cursor session with Eldon), for a fresh session to pick up and run.*
*This is the highest-leverage open experiment in the program. Eldon and I agreed: close this loop before reframing the outreach.*

---

## The one-sentence goal

Show whether the **attention-level** causal handle from exp-064 (editing a head's QK log-distance slope moves its conformal dimension Δ̂ *and* its attention valley) **propagates to task-level accuracy** — i.e. whether the slope edit actually changes lost-in-the-middle retrieval behavior. If it does, that's the program's most ML-legible, interpretation-free result. If it doesn't, that's an informative bound. Either way it's worth knowing.

## Why this is open (read first)

- **exp-064 (DONE, works):** κ-rescaling `W_Q ← (I + (κ−1)·P_U)·W_Q` on conformal heads moved Δ̂ 8/8 and the attention valley 24/24 in the predicted direction, clean sham null. The handle works *at the attention-statistic level*. Notes: `experiments/exp-064_slope_editing/notes.md`.
- **exp-068 (DONE, UNDERPOWERED — the blocker):** the task-level test failed *not* on physics but on task design — Pythia-1.4b solves the 20-doc keyword-retrieval task at ceiling (base V_task = 0.000, perfect at middle positions). No U-shape ⇒ no valley to move. The harness is correct (T-D sham null PASSED, 0 token diffs). Notes: `experiments/exp-068_task_slope_editing/notes.md`. Pre-reg: `notes/2026-06-14_task_level_slope_editing_prereg.md`.

**So the missing piece is a (model, task) pair that elicits a genuine U-shape in the BASE model — V_task > 0.10 — before any intervention.** That is step 1.

## Plan (in order)

### Step 1 — Task calibration sweep (NO new pre-registration; this only measures the base model)

Goal: find a config with base-model V_task > 0.10. Cheapest first, escalate only if needed. All models below are **already cached** (no download, no credentials):
`models--EleutherAI--pythia-1.4b`, `models--EleutherAI--gpt-neo-2.7B`, `models--gpt2-medium`.

Reuse the exp-068 task harness (`experiments/exp-068_task_slope_editing/run_exp068.py`) — it builds the Liu-style multi-doc retrieval task and measures per-position accuracy. Vary difficulty and sweep base accuracy across the 5 position bins:

1. **Harden the task on Pythia-1.4b first (no new model):**
   - Push context toward the 2048 window: more documents (N_doc = 30–40) / longer docs, so middle positions sit deeper in context.
   - Make retrieval non-trivial: require reading the target document (a fact embedded in prose) rather than matching a keyword label; or multi-hop (answer in doc A points to doc B).
   - Measure base V_task per config. **Stop as soon as a config gives base V_task > 0.10 with primacy/recency clearly above middle.**
2. **If Pythia-1.4b stays at ceiling:** repeat the sweep on **GPT-Neo-2.7B** (cached; ALiBi; 2048 ctx; different capability profile — may sit closer to the U-shape boundary).
3. **If neither shows a U-shape locally:** the honest conclusion is that the U-shape needs a larger model / longer context than fits here → escalate to the cloud path (MPT-30B-Instruct, which Liu et al. confirmed shows the U-shape). That is the exp-062-adjacent cloud route and needs Eldon's Modal credentials (see his note / `exp-062 LAUNCH.md`). Don't force a weak U-shape; report honestly and hand off.

Record the calibration sweep as its own small artifact (analysis-only, no intervention) — registry id **exp-069** (calibration), quality `honest_negative`-capable.

### Step 2 — Pre-register the intervention (ONLY after a U-shape config is found)

The predictions are already written and are reusable almost verbatim: `notes/2026-06-14_task_level_slope_editing_prereg.md` (T-A deepening at κ=1.5, T-B shallowing at κ=0.5, T-C monotonicity, T-D sham null, matched-sham-head specificity control). **Update only the locked (model, task-config) to the calibrated choice, commit the new pre-reg to git BEFORE any edited-weight forward pass.** New registry id **exp-070** (or exp-068-v2) for the powered run.

### Step 3 — Run the κ sweep with the exp-064 operator

Reuse the exp-068 editing implementation (already debugged: positional projector = `_compute_all_projectors`; single-token answer words verified for the GPT-NeoX tokenizer). Head selection: R² ≥ 0.85, Δ̂ ∈ [0.10, 0.40], top-8 by R² in the middle-third layers, from the relevant census (exp-059 for Pythia-1.4b; for GPT-Neo-2.7B run a BCFT/power-law fit first to get the census). κ ∈ {0.5, 1.0 sham, 1.5, 2.0}. Report **all four conditions** plus the matched-sham-head control regardless of outcome.

## Discipline (non-negotiable)

- Calibration (Step 1) measures only the base model — it is allowed before pre-registration and is NOT the test.
- The intervention pre-reg (Step 2) is committed to git before any edited forward pass. Verification-first.
- **An honest kill is a result, not a failure:** if the handle doesn't move task accuracy, it bounds the behavioral significance of the attention-level effect — write it cleanly, lead with the registered verdict.
- Watch the substrate pulls named in `memory/observations/substrate/cursor/2026-06-12_fable-5.md`: the urge to lead with a flattering constructive finding, and prose over-compression. Lead with the registered verdicts.

## What needs Eldon

- **Nothing for Steps 1–3 on the local path** (Pythia-1.4b / GPT-Neo-2.7B are cached). I can run this autonomously.
- **Only if Step 1 escalates to cloud** (MPT-30B): Modal credentials — see the credentials note Eldon is setting up. exp-062 (the separate universality-vs-imprint experiment) needs the same Modal access and can run in parallel once credentials exist.

## Files

- Harness + operator to reuse: `experiments/exp-068_task_slope_editing/run_exp068.py`
- Predictions to reuse: `notes/2026-06-14_task_level_slope_editing_prereg.md`
- exp-064 operator reference: `experiments/exp-064_slope_editing/run_slope_edit.py`
- Census for head selection: `experiments/exp-059_split_half_stability/` (Pythia-1.4b)
- Log the verdict to: the experiment's own `notes.md`, `RESEARCH_MAP.md`, the physics `queue.md`, and `STATUS.md` if it lands.
