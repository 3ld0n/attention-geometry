# Pre-Registration: Powered Task-Level Slope Editing on Pythia-1.4b (exp-070)

*Written June 16, 2026 (Cursor session, Ariel, solo). Committed to git BEFORE any
edited-weight forward pass. This supersedes the exp-068 pre-registration
(`2026-06-14_task_level_slope_editing_prereg.md`, commit 6ed3feec), which ran
UNDERPOWERED because the keyword-retrieval task was at ceiling.*

*Calibration that justifies this config: exp-069 (analysis-only, base model). See
`experiments/exp-069_task_calibration/notes.md`. The only change from the June-14
pre-registration is the locked **task** (now the embedded-prose task at N_doc=40 that
exp-069 showed elicits a genuine base-model U-shape). Head selection, intervention
operator, κ values, predictions, and evaluation are unchanged.*

---

## What changed from exp-068 (and why)

exp-068 used a keyword-retrieval task where the answer word is a verbatim string adjacent
to the query key. exp-069 demonstrated this is solved at ceiling by induction-head copy at
all positions (base V_task = 0.000) — there is no valley to move. exp-069 swept task
difficulty and found that an **embedded-prose retrieval task at N_doc = 40 (~1.7k tokens)**
elicits a genuine lost-in-the-middle U-shape in base Pythia-1.4b:

> primacy 1.00, near_primacy 0.975, **middle 0.85**, near_recency 0.95, recency 0.95 → **base V_task = 0.15** (40 samples/bin).

Both edges are clearly above the middle. This is the powered (if shallow) testbed.

**Honest power caveat (locked):** the valley is shallow (base V_task = 0.15; edges near
ceiling). If the κ-sweep returns ambiguous, that bounds the behavioral significance of the
attention-level effect and motivates the cloud path (MPT-30B). An honest weak result is a
result.

---

## Locked pre-run

Git commit of this file: **(committed before any edited-weight forward pass — see exp-070 results.json `preregistration_commit`)**

---

## Experimental Setup

### Model
**EleutherAI/pythia-1.4b**, fp32, MPS, `attn_implementation="eager"`. 24 layers, 16 heads,
head_dim 128, hidden 2048. 2048-token training context.

### Task (LOCKED — the change from exp-068)
**Embedded-prose multi-document retrieval, N_doc = 40** (exp-069 config C6).
- Each document: a short prose passage about a project/survey identified by `Item-NNN`,
  with the answer bound **once** in matched-cloze prose:
  *"…the team noted that the {name} signal marker was set to {word} throughout the study."*
  plus filler sentences that mention the name but not the answer.
- The target document is placed at one of 5 position bins (0-indexed doc slots):
  **[0, 10, 20, 29, 39]** = primacy / near_primacy / middle / near_recency / recency.
- Distractors are structurally identical documents about other `Item-NNN` identifiers.
- Question (matched cloze): *"…what was the {name} signal marker set to? The {name} signal
  marker was set to"*.
- **40 query items × 5 positions = 200 inputs per condition.** Answer words are
  single-token in the GPT-NeoX tokenizer (verified at runtime). Context ≈ 1.7k tokens
  (< 2048). seed = 42 (task), fixed across all conditions.
- Evaluation: argmax of the final-token logits == first answer token (exact-match,
  Liu-et-al. protocol).

### Intervention (UNCHANGED from exp-064/exp-068)
κ-rescaling of the QK matrices in the top conformal heads:
```
W_Q_h ← W_Q_h · M_edit ,   M_edit = I + (κ − 1)·P_U   (M_edit symmetric)
```
`P_U` = top-8 PCs of the position-mean input_layernorm output at the target layer
(50 random inputs, L=512, PROJ_SEED=68; same method as exp-064/exp-068).

### Head selection (UNCHANGED — reuse exp-068 Stage 1, committed 2026-06-16T141234Z)
8 target heads from the exp-059 Pythia-1.4b census: R² ≥ 0.85, Δ̂ ∈ [0.10, 0.40], top-8 by
R² in layers 8–20. The committed targets:
L19H3, L11H13, L12H13, L10H6, L16H2, L8H6, L9H11, L20H6.
Sham (matched control) heads: next 8 by R² from the same pool
(L8H3, L17H1, L14H9, L19H11, L12H7, L20H8, L20H11, L9H15).

### κ values
{0.5, 1.0 (sham), 1.5, 2.0}, applied globally to all 8 target heads simultaneously
(separate in-memory forward pass per κ; no fine-tuning, no weight file saved).

---

## Pre-Registered Predictions (UNCHANGED from exp-068)

### T-A — valley deepening
*Prediction:* V_task(κ=1.5) > V_task(κ=1.0, sham).
- **KEEP:** V_task(1.5) > V_task(1.0)
- **KEEP (strong):** V_task(1.5) > V_task(1.0) + 0.05
- **KILL:** V_task(1.5) ≤ V_task(1.0)

### T-B — valley shallowing
*Prediction:* V_task(κ=0.5) < V_task(κ=1.0, sham).
- **KEEP:** V_task(0.5) < V_task(1.0)
- **KILL:** V_task(0.5) ≥ V_task(1.0)

### T-C — magnitude ordering
*Prediction:* monotone in κ.
- **KEEP:** Spearman ρ(κ, V_task) ≥ +0.80 over {0.5, 1.0, 1.5, 2.0}
- **AMBIGUOUS:** 0 ≤ ρ < 0.80
- **KILL:** ρ < 0
*(Non-monotonicity at κ=2.0 is a finding, not grounds for threshold revision — exp-064 saw
two heads leave the conformal regime at κ=2.0.)*

### T-D — sham null
*Prediction:* κ=1.0 reproduces base output exactly.
- **PASS:** 0 token changes over all 200 inputs
- **FAIL:** any token change → implementation error, abort

### Sham-heads specificity (secondary)
Apply κ=1.5 to the 8 sham heads only. *Clean specificity:* V_task(target κ=1.5) >
V_task(sham-heads κ=1.5).

---

## Kill criteria & reporting obligations

- T-A and T-B both needed for behavioral causality CONFIRMED; a kill of either is a finding.
- T-D fail → abort and fix implementation.
- **All four κ conditions reported regardless of outcome.**
- If the base model is at ceiling (≥0.90 all bins) or floor (≤0.10 all bins) in the sham
  condition → report UNDERPOWERED, not kill/confirm. (exp-069 base showed middle 0.85,
  edges 0.95–1.0 — neither ceiling nor floor, but shallow; report the registered verdicts
  AND the power note.)
- The registered T-A/T-B verdicts are reported before any exploratory position-level analysis.

---

## What is not pre-registered
- Position-level accuracy breakdown beyond the 5 bins (post-hoc).
- Effects at positions other than middle (primacy/recency may also shift).
- Any GPT-Neo-2.7B follow-up (different architecture; would need its own census/operator port).

---

*Registry: exp-070. Predecessor: exp-068 (underpowered). Calibration: exp-069. Author:
Ariel, solo Cursor session, June 16, 2026.*
