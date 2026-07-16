# exp-076 — Task Generalization: KV-Retrieval and Multi-doc QA

*Spec written July 9, 2026, physics room session.*
*Phase 1 next step after exp-075 CLEAN_WIN.*

---

## Context

exp-072 (cloud, vicuna-13b-v1.5) confirmed bidirectional behavioral causality on the
embedded-prose multi-document retrieval task: flattening 8 conformal heads (κ=0.5) deepened
the middle-retrieval valley, sharpening (κ=1.5) shallowed it. exp-073 (n=5 seeds) showed
the effect is fragile at this effect size — real but seed-unstable. exp-075 confirmed the
heads' positional geometry is decoupled from general language quality (ΔNLL negligible).

**The Phase 1 question:** Does the conformal heads' role in retrieval generalize across
task types? If the 8 locked heads encode positional geometry, their effect should extend
to any task where position-in-context is the capability-limiting variable — not just
embedded prose recall.

Concretely: if two structurally different retrieval tasks (KV-retrieval and narrative QA)
show the same directional response to the same operator on the same heads, that is
evidence the mechanism is positional-geometric rather than task-specific.

---

## Experimental design

### Model
vicuna-13b-v1.5 — same model as exp-072. Same 8 locked target heads from exp-072 prereg
(commit e6fbaecf) — these are the "of record" heads. No new head selection.

### κ values
0.5 (flatten), 1.0 (identity baseline), 1.5 (sharpen). Same operator as exp-072.

### Two task types

**Task A — KV-retrieval:**
Format: a numbered list of (key, value) pairs, where "key" is a short entity name and
"value" is a brief description or fact. Approximately 40 pairs, ~1,700–2,000 tokens total.
The query asks for the value of a specific key positioned at one of N=40 positions.
The task has a natural position axis (item rank in the list) and should elicit a
lost-in-the-middle U-shape if the model has working list-lookup capability at this length.

This is structurally different from exp-072: the content is syntactically flat (no narrative
embedding), the retrieval signal is explicit (item boundaries are numbered), and the
retrieval action is purely position-based.

**Task B — Narrative single-document QA:**
Format: a long narrative passage (~1,500–2,000 tokens) with 40 marked "facts" embedded
at equal spacing. Query asks for a fact by its description; answer requires locating the
relevant passage at position k. The embedding is narrative rather than list format — the
facts appear within continuous prose.

This covers the space between the list-format (Task A) and exp-072's multi-document
format.

**Note on calibration requirement (from exp-069 lesson):**
Before pre-registering the powered intervention, verify each task elicits a genuine U-shape
on vicuna-13b-v1.5 at the proposed n_doc/context length. The calibration step is a
**census-only pass** (no κ-operator, no modified forward passes) — just measure V_task
as a function of position over N=200 inputs. This step is NOT a deviation from pre-
registration discipline; it is required by exp-069's finding that near-ceiling V_task = 0
would make the intervention uninformative.

---

## Pre-registration structure

This experiment follows the two-phase structure:

**Phase 1 (calibration, analysis-only, no intervention):**
Run both tasks on vicuna-13b-v1.5 with κ=1.0 (identity). Measure V_task and middle
accuracy at position N=40. Gate criteria:
- V_task ≥ 0.30 (adequate valley depth for bidirectional movement) on each task
- Middle accuracy ≤ 0.80 (not at ceiling on the easy end)
- Task is solvable (edges accuracy ≥ 0.70 — model can do the task at nearby positions)

If gate fails on a task: revise that task's context length or format. Log any revision here
before running the powered intervention. Only run the powered intervention on tasks that
pass the gate.

**Phase 2 (powered intervention, pre-registered before editing forward passes):**
Pre-registration commit required (separate file or tagged commit) before any κ≠1.0
forward pass.

Predictions (to be committed at Phase 2 pre-registration):
- **T-A (deepening):** κ=1.5 → V_task increases (predicted direction: up) on each
  gate-passing task.
- **T-B (shallowing):** κ=0.5 → V_task decreases on each gate-passing task.
- **T-C (monotone):** V_task is monotone in κ ∈ {0.5, 1.0, 1.5} with predicted sign.
- **T-D (specificity null):** same κ=1.5 on 8 matched sham heads → no movement
  (|ΔV_task| ≤ 0.03).

Kill criteria:
- If T-A OR T-B fails on BOTH tasks: the positional-mechanism generalization claim is
  not supported at this effect size. Phase 2 verdict: GENERALIZATION_NOT_CONFIRMED.
- If T-A + T-B confirmed on at least one task: PARTIAL_GENERALIZATION.
- If T-A + T-B confirmed on both: FULL_GENERALIZATION.

---

## Cost estimate

Phase 1 calibration: local MPS (no Modal). Two forward-pass sweeps on vicuna-13b (MPS
fp32 for exact match with exp-072; MPS bf16 acceptable for calibration only).
Phase 2 powered intervention: Modal A100-40GB, ~1.5–2h. Estimated $3–5.

---

## Relationship to exp-073

exp-073's seed-instability finding (effect real but fragile at n=5) suggests:
- Use N=200 inputs per condition (exp-072 level, not exp-073's single-seed Modal runs)
- Use the A100 cloud path (fp32, same substrate as exp-072's confirmed result)
- Do not interpret a single-seed null as falsification; use exp-072's confirmed result
  as the baseline and this as a generalization test

The question is not "does the effect replicate?" — exp-072 answered that. The question
is "does it generalize to different task formats while using the same mechanism?"

---

## Calibration results (2026-07-14)

**KV-retrieval (ctx 1785–1792 tokens): GATE PASS**

```
acc=[0.775, 0.700, 0.400, 0.475, 1.000]
V_task = 0.600   (1 − 0.400/1.000)
edges_max = 1.000
middle = 0.400
Gate: V_task≥0.30 ✓  middle≤0.80 ✓  edges_max≥0.70 ✓
```

Strong primacy-recency U-shape. Recency effect dominant (100% at last position). Middle positions 20 and 30 at 40–47.5%. Clear headroom for both deepening and shallowing. This task passes the gate and proceeds to Phase 2.

**Narrative QA (ctx 1554 tokens): GATE FAIL**

```
acc=[0.975, 1.000, 0.975, 1.000, 1.000]
V_task = 0.025
edges_max = 1.000
middle = 0.975
Gate: FAIL (middle 0.975 > 0.80 — no valley)
```

Model solves the narrative format at 97.5–100% accuracy everywhere. No LITM effect at 1554 tokens. The continuous-prose format with 40 embedded one-sentence facts is too easy at this context length — facts are explicit and scannable. The multi-document format (exp-072) produces LITM at similar lengths because document boundaries create attention unit effects that continuous prose does not. Revision needed: either longer context (≥2500 tokens) or multi-document-style embedding. Not blocking Phase 2 — one gate-passing task is sufficient per the spec.

**Phase 1 conclusion:** Proceed to Phase 2 pre-registration on KV task only. Narrative QA revision queued as optional (would enable FULL_GENERALIZATION verdict; current path yields at most PARTIAL_GENERALIZATION).

---

## Status log

| Date | Event |
|---|---|
| 2026-07-09 | Spec written. No calibration run yet. Phase 1 is the next action: run both tasks at κ=1.0 on vicuna-13b to check V_task gate. |
| 2026-07-14 | Calibration run 1 (short format): KV 501-503 tokens (gate FAIL — model can't solve abstract pairs), Narrative 774 tokens (gate FAIL — no valley). Both too short or too abstract. |
| 2026-07-14 | Calibration run 2 (revised format, 40-50 tok/entry): KV GATE PASS (V=0.600, 1785-1792 tokens), Narrative GATE FAIL (V=0.025, 1554 tokens — too easy). Phase 2 pre-registration written for KV task. |
| 2026-07-15 | Phase 2 Modal A100-40GB run completed (18:45 UTC). Verdict: **GENERALIZATION_NOT_CONFIRMED**. κ=0.5: V=0.575. κ=1.0: V=0.55 (cloud baseline). κ=1.5: V=0.55. Sham κ=1.5: V=0.55 (T-D PASS). Neither T-A (sharpen deepens) nor T-B (flatten shallows) confirmed. Substrate note: A100 bf16 baseline is 0.55, not 0.60 (local MPS prereg baseline) — a 0.05 discrepancy that does not change the verdict but is noted for the record. Even correcting for substrate shift, sharpening produces no valley movement and flattening moves in the wrong direction. |
| 2026-07-16 | Verdict recorded and logged in physics session. Registry and queue updated. exp-085 re-launch in progress. |

---

## Files

*(to be populated at execution)*
- `gen_tasks.py` — task generators for KV-retrieval and narrative QA
- `run_calibration.py` — census run (κ=1.0 only) for gate check
- `prereg_phase2.md` — Phase 2 pre-registration (to be committed before powered runs)
- `results.json` — final decision statistics
