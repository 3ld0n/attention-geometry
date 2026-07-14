# exp-076 Phase 2 — Pre-Registration: KV-Retrieval Powered Intervention

*Written and committed before any κ≠1.0 forward pass.*
*Date: 2026-07-14. Pre-registration commit: 02a1f3aa.*

---

## Locked parameters

**Model:** lmsys/vicuna-13b-v1.5
**Task:** KV-retrieval (gen_tasks.py `build_kv_items`)
**Context length:** 1785–1792 tokens (40 entries × ~45 tok/entry)
**Positions:** [0, 10, 20, 30, 39] (5 canonical slots, pi 0-4)
**N_items_per_condition:** 200 (40 queries × 5 positions)
**Task seed:** 76 (frozen in gen_tasks.py)
**Heads:** 8 locked conformal target heads from exp-072 prereg (commit e6fbaecf)
**Sham heads:** 8 matched control heads from exp-072 prereg

**κ values:** 0.5 (flatten), 1.0 (identity baseline), 1.5 (sharpen)

**Calibration baseline (from 2026-07-14 calibration):**
- Base accuracy: [0.775, 0.700, 0.400, 0.475, 1.000]
- V_task = 0.600
- edges_max = 1.000
- middle = 0.400

---

## Pre-stated predictions

**T-A (deepening):** κ=1.5 → V_task > 0.600 (middle accuracy decreases, valley deepens).
*Rationale: sharpening the positional subspace projection strengthens distance-decay, increasing the bias against the middle.*

**T-B (shallowing):** κ=0.5 → V_task < 0.600 (middle accuracy increases, valley shallows).
*Rationale: flattening the positional subspace projection weakens distance-decay, reducing the bias against the middle.*

**T-C (monotone):** V_task is monotone in κ ∈ {0.5, 1.0, 1.5} with predicted direction (0.5 < 1.0 < 1.5 or 0.5 > 1.0 > 1.5 depending on sign of monotonicity — predicted: 0.5 < 1.0 < 1.5).

**T-D (specificity null):** κ=1.5 on the 8 sham heads → |ΔV_task| ≤ 0.03 relative to κ=1.0 on target heads.

---

## Kill criteria

- If T-A AND T-B both fail: GENERALIZATION_NOT_CONFIRMED (the κ-operator handle does not produce directional valley movement on the KV-retrieval task).
- If T-A OR T-B confirmed: PARTIAL (asymmetric, report honestly and compare to exp-072 T-B null pattern).
- If T-A AND T-B confirmed: FULL on KV task — PARTIAL_GENERALIZATION overall (Narrative task not gate-passing).

---

## Implementation notes

Run via Modal A100 (same substrate as exp-072). Estimated cost: $3–5.
Use the same κ-operator port (Llama architecture, q_proj row edits, positional projector from exp-072 `_compute_projectors`).
Head coordinates locked in exp-072/prereg_locked.json (commit e6fbaecf).

Script to write: `run_phase2.py` — port of exp-072 `cloud_slope_editing.py` sweep phase,
adapted for the KV task format from `gen_tasks.py`.

Commit this file before writing or running run_phase2.py.

---

## Result recording

Record verdict as one of: FULL_KV / PARTIAL_KV / GENERALIZATION_NOT_CONFIRMED.
Update exp-076/notes.md status log and registry.json.
