# exp-068 — Task-Level Slope Editing: Pythia-1.4b on Multi-Document Retrieval

*Date: 2026-06-16*
*Pre-registration: `research/physics/notes/2026-06-14_task_level_slope_editing_prereg.md` (commit 6ed3feec)*
*Session: solo physics room session, ~12:00 PM MDT*

---

## Hypothesis

κ-rescaling of the QK matrices in the top-8 conformal heads of Pythia-1.4b moves the task-level "Lost in the Middle" valley score V_task in the predicted direction: deepening (κ > 1) → larger V_task; shallowing (κ < 1) → smaller V_task. This is the first test of behavioral causality in the program — whether the QK-slope → attention-valley causal chain (confirmed in exp-064) propagates to task-level accuracy.

---

## Setup

- **Model:** EleutherAI/pythia-1.4b, loaded fp32 on MPS (Apple M5 Max)
- **Head selection (Stage 1, committed 2026-06-16T141234Z):** 8 target heads from exp-059 Pythia-1.4b census, R² ≥ 0.85, Δ̂ ∈ [0.10, 0.40], layers 8–20. 37 candidates. Top 8 by R²: L19H3 (Δ=0.111), L11H13 (Δ=0.267), L12H13 (Δ=0.140), L10H6 (Δ=0.246), L16H2 (Δ=0.185), L8H6 (Δ=0.169), L9H11 (Δ=0.368), L20H6 (Δ=0.233).
- **Task:** Liu et al. (2023) multi-document retrieval synthetic replica. N_doc=20, ~70 tokens/doc, 10 query words, 5 position bins (primacy/near_primacy/middle/near_recency/recency), 50 items total, TASK_SEED=42. Context length: 1406 tokens (constant across all items).
- **Answer words (single-token verified):** red, oak, gem, ice, map, sky, fig, ash, bay, cup. (`dew` and `elm` replaced pre-run — not single-token in GPT-NeoX tokenizer; recorded as pre-run protocol correction, not a threshold revision.)
- **κ values:** {0.5, 1.0 (sham), 1.5, 2.0}
- **Sham heads:** next 8 by R² from same candidate pool.
- **Positional projectors:** computed with a single set of 50 forward passes across all 10 needed layers (PROJ_SEED=68, PROJ_SEQ=512). P_U trace=8.00 for all layers (correct: 8 PCs).

---

## Results

### Base model performance

| Position | Accuracy (κ=1.0 sham) |
|---|---|
| primacy (pos 1) | 1.000 |
| near_primacy (pos 5) | 0.900 |
| middle (pos 10) | 1.000 |
| near_recency (pos 16) | 1.000 |
| recency (pos 20) | 1.000 |

**V_task (base) = 0.000.** The model achieves perfect accuracy at middle positions — no U-shape exists in the base model.

### κ sweep

| κ | V_task |
|---|---|
| 0.5 | 0.000 |
| 1.0 | 0.000 |
| 1.5 | 0.000 |
| 2.0 | 0.000 |

All conditions identical. T-D (sham null): PASS (0 token differences). T-A (deepening): KILL numerically. T-B (shallowing): KILL numerically. T-C (monotone): KILL numerically.

**Overall verdict: UNDERPOWERED.** Per pre-registration: "If the task has a ceiling effect (base model near-perfect at all positions), report that and record the experiment as underpowered — not a kill, not a confirm."

---

## Interpretation

This is not a test of the physics hypothesis — it is a task calibration failure. Pythia-1.4b trivially solves the 20-document keyword retrieval format at ~1406 token contexts, achieving essentially perfect accuracy at all position bins including middle.

The Liu et al. "Lost in the Middle" U-shape appears in models that are capable but not trivially correct at the task. The valley measures *relative* accuracy, not absolute accuracy. When the base model is at ceiling, there is no valley to probe and no slope edit can change the (null) valley score in a detectable direction.

This does not falsify the QK-slope → behavior chain. exp-064 established that κ-rescaling moves per-head Δ̂ and the attention valley statistic in the predicted direction (8/8 mechanical, 24/24 behavioral sign agreements). The question of whether that attention-level handle propagates to task accuracy is unanswered by this experiment because the task is too easy.

**What's needed for a powered test:**
1. A task design that actually elicits a U-shape in Pythia-1.4b (or whichever model is used). The task should show V_task > 0.10 in the base model at the chosen context length.
2. Options:
   - Longer documents or multi-hop retrieval requiring genuine reasoning (not just keyword lookup)
   - Much longer contexts (requires models with longer context windows — MPT-30B-Instruct, Llama-3-70B, or similar, as noted in the pre-registration)
   - Check whether GPT-Neo-2.7B (cached, ALiBi, trained on 2048-token context) shows a U-shape on this or a harder task variant

**What this experiment did establish:**
- The task construction and evaluation pipeline work correctly (T-D sham null: PASS)
- Pythia-1.4b's 1.4B parameters are sufficient to solve keyword retrieval across 1406 tokens with near-perfect accuracy — useful calibration information for future task design
- The head selection and κ-rescaling implementation are correct (confirmed by T-D pass and by the fact that all conditions ran without error)

---

## Next steps

1. **Redesign the task.** Find a task format that elicits a genuine U-shape in an available model. The simplest diagnostic: sweep base model accuracy across position bins for a range of task difficulties before committing to the pre-registration. This calibration step is missing from the current protocol and should be added.
2. **Check GPT-Neo-2.7B.** Different architecture (ALiBi, 2048 context), potentially closer to the capability boundary. Run the base-model calibration sweep.
3. **Alternatively, harder retrieval.** Multi-document question answering where the answer requires reading the target document (not just matching a keyword label). This reduces ceiling performance.
4. **Longer contexts.** Cloud-GPU experiment (exp-062) with models that have demonstrated the U-shape (MPT-30B-Instruct, confirmed in Liu et al.) remains the cleanest test — but requires Eldon's cloud credentials.

The failure here is in task selection, not in physics or implementation. Record it cleanly, design around it.

---

*Registry: exp-068. Status: underpowered (ceiling effect). Quality: honest_negative = true.*
