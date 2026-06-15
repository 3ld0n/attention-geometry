# Pre-Registration: Task-Level Slope Editing on Pythia-1.4b

*Written June 14, 2026. Committed before any task measurements. This is the pre-registration for Research Plan §2.3, full task-level study.*

*Background: exp-064 established that κ-rescaling of QK matrices moves per-head Δ̂ and the attention valley statistic in the predicted direction (8/8 mechanical, 24/24 behavioral signs). This pre-registration tests whether that attention-level handle propagates to task-level long-context retrieval accuracy.*

---

## Locked Pre-Run (commit this before any task measurement)

Git commit of this file: **to be filled after commit, before running any task evaluation**

---

## Experimental Setup

### Model

**EleutherAI/pythia-1.4b** (HuggingFace, cached locally). GPT-2 was rejected as a retrieval testbed in April 2026 (exp-024: task too easy, near-ceiling at all positions). Pythia-1.4b has a 2048-token training context and reliable conformal heads from exp-059/exp-026.

### Intervention

The exp-064 κ-rescaling operator applied to the **top conformal heads** in Pythia-1.4b:

```
W_Q ← (I + (κ − 1) · P_U) · W_Q   per selected head
```

where P_U = top-8 PCs of the position-mean layer-norm output at the target layer (same method as exp-064).

**Head selection criteria (locked):**
- R² ≥ 0.85 on 1-D power-law fit (from exp-026/exp-059 census)
- Δ̂ ∈ [0.10, 0.40]
- Select the top-8 by R² from layers 8–20 (middle third of the 24-layer model)
- Record the selected heads and their Δ̂, λ̂ before any intervention

**κ values:** {0.5, 1.0 (sham), 1.5, 2.0}

Each κ is applied globally to all 8 selected heads simultaneously (same as exp-064 behavioral leg). Each model variant is a separate forward pass with the edited weights in-memory; no fine-tuning, no weight file saved.

### Task

**Liu et al. (2023) "Lost in the Middle" multi-document retrieval**, synthetic replica:

- Input format: N_doc = {10, 20} documents of ~100 tokens each; one documents contains the answer to a factual question; remaining documents are Wikipedia distractors
- Answer document position: {1, 5, middle, N_doc−4, N_doc} (primacy, near-primacy, middle, near-recency, recency)
- Evaluation: exact match of the single-token answer (following the Liu et al. protocol)
- Total inputs per condition: 50 (10 questions × 5 positions × repeat with same positions; single-answer questions from a fixed seed-42 pool)
- Context length at N_doc = 20: approximately 2000 tokens (fits within Pythia-1.4b's 2048-token window)

**Task metric:** Accuracy at each position bin (fraction correct out of 50), and the *valley score* V_task = 1 − acc(middle) / max(acc(primacy), acc(recency)).

### Sham control

The κ = 1.0 condition must reproduce exactly the base model output (no token changes). Failure of the sham to be exactly null is a hardware/implementation error and must be reported.

---

## Pre-Registered Predictions

**Commit the following as locked before any task measurement.**

### T-A: Direction of valley deepening

*Prediction:* At κ = 1.5, V_task(edited) > V_task(sham). (Deepening Δ deepens the attention valley; this should reduce middle-position accuracy relative to primacy/recency.)

*Threshold:* 
- **KEEP:** V_task(κ=1.5) > V_task(κ=1.0, sham) 
- **KEEP (strong):** V_task(κ=1.5) > V_task(κ=1.0) + 0.05
- **KILL:** V_task(κ=1.5) ≤ V_task(κ=1.0, sham)

*Rationale:* exp-064 showed κ > 1 deepens the attention valley (24/24 behavioral signs). If the attention valley is causally upstream of task-accuracy valley, the same edit should deepen V_task.

### T-B: Direction of valley shallowing

*Prediction:* At κ = 0.5, V_task(edited) < V_task(sham). (Flattening Δ should reduce the attention valley depth and thus improve middle-position retrieval.)

*Threshold:*
- **KEEP:** V_task(κ=0.5) < V_task(κ=1.0, sham)
- **KILL:** V_task(κ=0.5) ≥ V_task(κ=1.0, sham)

### T-C: Magnitude ordering

*Prediction:* V_task(κ=2.0) ≥ V_task(κ=1.5) > V_task(κ=1.0) > V_task(κ=0.5). Monotone in κ.

*Threshold:*
- **KEEP:** Spearman ρ(κ, V_task) ≥ +0.80 (4 points)
- **KILL:** ρ(κ, V_task) < 0 (reversed direction)
- **AMBIGUOUS:** 0 ≤ ρ < 0.80 (some monotonicity, weak)

*Note:* At κ = 2.0, exp-064 observed two heads degrading out of the conformal regime. If the conformal regime breaks, the prediction for T-C may not hold at κ = 2.0. The KILL criterion is on the sign, not the full monotonicity — a non-monotone result between κ = 1.5 and κ = 2.0 is a finding, not grounds for retrospective threshold revision.

### T-D: Sham null

*Prediction:* Sham (κ = 1.0) matches base-model output exactly (zero token disagreement on all 50 × 5 inputs).

*Threshold:*
- **PASS:** 0 token changes
- **FAIL:** Any token change — implementation error; abort

---

## Kill criteria

Any of the following constitutes a kill for task-level causality (T-A and T-B both needed; a kill of either is a finding):

1. T-A kills: no measurable deepening at κ = 1.5
2. T-B kills: no measurable shallowing at κ = 0.5
3. T-D fails: abort, fix implementation, re-run

An honest kill means: the QK-slope → attention-valley chain is causal (exp-064), but the attention-valley → task-accuracy link is not detectable at this model scale with this task and intervention size. This is a scientifically informative result: it bounds the attention-level effect's behavioral significance and motivates the cloud version (MPT-30B, Llama-3-70B, with stronger U-shape behavior).

---

## Sham edit design (details)

The sham head list must be a **matched control**: same number of heads (8), drawn from the same layer range, with similar R² and Δ̂ but from a different randomly-selected group. Apply κ = 1.5 to the sham heads only (not the target heads). Measure V_task. This controls for the effect of "editing any heads" vs "editing the right heads."

*Sham vs. target threshold (secondary):* V_task(target κ=1.5) > V_task(sham-heads κ=1.5) required for a clean specificity claim.

---

## What is not pre-registered

- Layer-level or head-level accuracy breakdown (post-hoc)
- Any claim about accuracy vs. base model for positions other than middle (primacy/recency may also shift)
- Comparison against StreamingLLM-style baselines (future work)
- Effect at N_doc = 10 (smaller context; registered only at N_doc = 20 for the main test)

---

## Reporting obligations

- **All four κ conditions reported**, regardless of which pass or fail.
- If the task has a ceiling effect (base model near-perfect at all positions), report that and record the experiment as **underpowered** — not a kill, not a confirm.
- If the task has a floor effect (base model near-zero everywhere), same reporting obligation.
- The registered verdict for T-A and T-B must be reported before any exploratory analysis of position-level breakdowns.

---

*Registry: exp-068 (to be assigned). Scripts: to be written after this commit. Pre-registration author: Ariel, solo physics room session June 14, 2026.*
