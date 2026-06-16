# Pre-Registration: Powered Cloud Slope Editing on a deep-U-shape RoPE+MHA model (exp-072)

*Written June 16, 2026 (Cursor session, Ariel, solo). Committed to git BEFORE any
edited-weight forward pass on the cloud model. Adjudicates the open shallowing leg
(T-B) left by exp-070.*

*Calibration that justifies this config: exp-071 (cloud, analysis-only, base model +
conformal head census). See `exp-071_cloud_calibration/results.json`. The predictions,
intervention operator family, κ values, and evaluation are UNCHANGED from exp-070
(`notes/2026-06-16_powered_task_slope_editing_prereg.md`, commit 9e6239b9). The only
changes are the LOCKED (model, architecture port, task N_doc, head census) — all of which
exp-071 measured on the base model before this file was committed.*

---

## What this phase answers (the one open question from exp-070)

exp-070 (Pythia-1.4b, local) returned `BEHAVIORAL_CAUSALITY_KILLED`: the **deepening**
leg propagated cleanly (T-A KEEP-strong, T-C ρ=0.949, head-specific, sham null clean), but
the **shallowing** leg (T-B) was null — most plausibly because the local base middle
accuracy (0.85) sat near the task ceiling, leaving κ<1 no headroom to *raise* the middle.
That headroom explanation was an inference, not a registered result.

**This phase adjudicates it.** Run the SAME κ-sweep on a model with a DEEP base U-shape —
middle well below the edges AND edges off the ceiling — so κ<1 has room to raise the
middle. If shallowing works there, exp-070's T-B null was a headroom artifact and the
behavioral causality is bidirectional. If it still fails on a deep valley, that is a real,
registered bound on the shallowing direction.

---

## Why a Llama-family model, not MPT-30B-Instruct (documented before locking)

1. **Availability:** `mosaicml/mpt-30b-instruct` no longer exists on HF (only third-party
   GGML/quantized forks), so it cannot run our editable fp32 forward pass.
2. **Mechanism (the substantive reason):** MPT uses **ALiBi** — position enters as a fixed
   additive bias on attention scores, NOT in the query content. The κ-operator rescales
   `W_Q`'s projection onto the positional subspace `P_U`; that presumes position lives in
   the query, which is true for **RoPE** (the architecture exp-064/068/070 validated on)
   and false for ALiBi. Running MPT would risk a null that means nothing.

So the model is kept on home turf: a **deep-U-shape RoPE + full-MHA instruction model**.
Candidate (confirmed ungated, RoPE, 40 heads = 40 KV heads ⇒ full MHA, head_dim 128):
`lmsys/vicuna-13b-v1.5` (native 4096 context), fallback `lmsys/longchat-13b-16k`.

---

## Locked pre-run (filled from exp-071 calibration, before any edited forward pass)

- **Model:** `lmsys/vicuna-13b-v1.5`, fp32, CUDA (A100-80GB), `attn_implementation="eager"`.
  40 layers, 40 heads, hidden 5120, head_dim 128, max_position_embeddings 4096. RoPE, full
  MHA (40 query heads = 40 KV heads). Ungated.
- **Task N_doc:** **40**. exp-071 base U-shape: accuracy by position
  `[0.775, 0.65, 0.325, 0.525, 0.80]` (primacy / near_primacy / **middle** / near_recency /
  recency) → **base V_task = 0.594**, middle 0.325, edges ≤ 0.80. Deep, two-sided headroom
  (middle far below edges; edges off the ceiling). Positions = 5 bins **[0, 10, 20, 30, 39]**.
  40 single-token items × 5 positions = **200 inputs/condition**. seed = 42 (task), fixed
  across all conditions.
- **Target heads (8):** R²≥0.85, Δ̂∈[0.10,0.40], top-8 by R² in the mid-layer band
  [round(0.33·40), round(0.83·40)] = **layers 13–33** (the proportional analogue of Pythia's
  8–20 of 24). Selected (L,H, Δ̂, R²):
  **L24H17 (0.336, 0.994), L27H18 (0.311, 0.989), L24H2 (0.290, 0.989), L25H4 (0.282, 0.988),
  L26H30 (0.288, 0.988), L22H33 (0.352, 0.987), L25H30 (0.364, 0.987), L25H33 (0.298, 0.985)**.
- **Sham heads (8, matched control):** next 8 by R² from the same pool:
  **L24H25, L21H34, L26H29, L15H1, L21H0, L25H9, L29H31, L17H13**.
- **Census:** exp-025/exp-059 pipeline verbatim (L=512, 50 random-token inputs, fp32,
  lag-profile Δ̂ on lags [3,120), deep-query thirds valley). **732 conformal heads**
  (R²≥0.85, Δ≥0.05); 298 candidates in the selection band.
- Machine-readable lock: `prereg_locked.json` (what the sweep reads).
- Git commit of this locked file: **(recorded in results.json `preregistration_commit`)**.

### Intervention (UNCHANGED operator family from exp-064/068/070; Llama port)
κ-rescaling of the query projection in the target heads:
```
W_Q_h ← W_Q_h · M ,   M = I + (κ − 1)·P_U      (M symmetric)
```
`P_U` = top-8 PCs of the position-mean `input_layernorm` output at the target layer
(50 random inputs, L=512, PROJ_SEED=68; same method as exp-064/070). **Llama port:**
`W_Q_h` is the per-head row block of `self_attn.q_proj.weight`; `input_layernorm` is the
RMSNorm feeding q_proj — the same normalized vector q_proj acts on, exactly as Pythia's
`input_layernorm` fed its fused `query_key_value`.

### κ values
{0.5, 1.0 (sham), 1.5, 2.0}, applied to all 8 target heads simultaneously (separate
in-memory forward pass per κ; no fine-tuning, no weight file saved).

---

## Pre-Registered Predictions (UNCHANGED from exp-070)

### T-A — valley deepening
*Prediction:* V_task(κ=1.5) > V_task(κ=1.0, sham).
- **KEEP:** V_task(1.5) > V_task(1.0); **KEEP (strong):** > V_task(1.0) + 0.05; **KILL:** ≤.

### T-B — valley shallowing  *(the leg this phase exists to test with headroom)*
*Prediction:* V_task(κ=0.5) < V_task(κ=1.0, sham).
- **KEEP:** V_task(0.5) < V_task(1.0); **KILL:** ≥.

### T-C — magnitude ordering
*Prediction:* monotone in κ.
- **KEEP:** Spearman ρ(κ, V_task) ≥ +0.80 over {0.5,1.0,1.5,2.0}; **AMBIGUOUS:** 0≤ρ<0.80;
  **KILL:** ρ<0.

### T-D — sham null
*Prediction:* κ=1.0 reproduces base output exactly.
- **PASS:** 0 token changes over all inputs; **FAIL:** any change → implementation error, abort.

### Sham-heads specificity (secondary)
Apply κ=1.5 to the 8 sham heads only. *Clean specificity:* V_task(target κ=1.5) >
V_task(sham-heads κ=1.5).

---

## Kill criteria & reporting obligations

- T-A and T-B both needed for behavioral causality CONFIRMED; a kill of either is a finding.
- **The interesting registered outcome this phase can produce:** with a deep base valley
  (edges off ceiling), T-B now has headroom. If T-B KEEPs → exp-070's T-B null was a
  headroom artifact; behavioral causality is bidirectional. If T-B still KILLs on a deep
  valley → a real, registered bound on the shallowing direction. **Both are reported as the
  registered verdict; neither is relabeled.**
- T-D fail → abort and fix the operator port (likely a Llama-specific slicing bug).
- **All conditions reported regardless of outcome.**
- If the base model is at ceiling (≥0.90 all bins) or floor (≤0.10 all bins) → report
  UNDERPOWERED, not kill/confirm. (exp-071 calibration is run precisely to avoid this.)
- The registered T-A/T-B verdicts are reported before any exploratory analysis.
- **Discipline (carried from the morning's text — Job 31:6, the even balance):** the
  registered verdict is the verdict. No leading with a flattering finding; name the
  asymmetry, if any, without flinching either way.

## What is not pre-registered
- Position-level breakdown beyond the 5 bins (post-hoc).
- Effects at positions other than middle.
- Any further model (each would need its own census/operator port).

---

*Registry: exp-072. Predecessor: exp-070 (KILLED on T-B, headroom-bounded inference).
Cloud calibration: exp-071. Operator family: exp-064. Author: Ariel, solo Cursor session,
June 16, 2026.*
