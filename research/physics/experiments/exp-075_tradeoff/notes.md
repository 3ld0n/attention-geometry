# exp-075 — Tradeoff: perplexity audit of the κ-operator on vicuna-13b-v1.5

*Status: PRE-REGISTERED (2026-07-09). Script written; pre-reg committed before any edited forward passes.*
*Program: PROGRAM_BRIEF_LITM_CAUSAL_HANDLE.md (Phase 1). Follows exp-073 (robustness).*
*Spec originally in exp-074_tradeoff/notes.md (renumbered after exp-074 was used for P-B2b).*

---

## Question

Does flattening the conformal heads (κ<1) buy middle-retrieval at the cost of general LM
quality — i.e. is distance-decay a **shared resource** the model budgets across positions?

Clean win vs. redistribution. Both are real mechanistic findings; a tradeoff is itself
evidence about how attention allocates.

---

## Pre-registration (written before any edited forward passes)

**Locked configuration:** exp-072 prereg_locked.json. 8 target heads in layers 22–27
of vicuna-13b-v1.5. Same projector protocol (PROJ_SEED=68, PROJ_N=50, N_PC=8,
input_layernorm basis). Operator: W_Q ← W_Q(I+(κ−1)P_U).

**Already known from exp-072 (not re-run here):**
- κ=0.5 middle-retrieval: 0.350 (baseline 0.325), Δmid = +1 item — RAISES middle
- κ=0.5 recency (position 4): 0.825 (baseline 0.800), Δrec = +1 item — does NOT hurt recency
- κ=0.5 primacy (position 0): 0.775 (baseline 0.775) — unchanged
- T-B (shallowing) registered as KEEP in exp-072: confirmed that flattening works

**New measurement in exp-075:**
LM perplexity (mean NLL nats/token) on wikitext-103-v1 test, first 50 passages of ≥512
tokens, each truncated to 512 tokens. Passages selected in dataset order (no shuffle).

**κ grid:** {0.5, 1.0, 1.5, 2.0} applied to the 8 locked target heads.
Also measured: unedited baseline and κ=1.5 sham condition (for specificity check).
**Precision:** bf16 on MPS (exp-075 is a capability audit, not a sham-null test; bf16 OK per program brief).

**Pre-stated hypotheses:**
- H_flat: |NLL(κ) − NLL(κ=1.0)| ≤ 0.05 nats/tok for all κ ∈ {0.5, 1.5, 2.0}.
  (Positional-subspace edit on 8/1600 heads has negligible global LM effect.)
- H_redistribute: |NLL(κ=0.5) − NLL(κ=1.0)| > 0.05 nats/tok.
  (Flattening those heads costs general language modeling quality.)

**Verdict logic (applied after seeing data, not before):**
- **CLEAN_WIN:** H_flat confirmed for κ=0.5, AND exp-072 middle↑, AND exp-072 recency↔/↑
  → distance-decay is NOT a strictly shared budget; flattening is deployable without cost.
- **REDISTRIBUTION:** H_flat falsified for κ=0.5 (|ΔNLL| > 0.05)
  → distance-decay shares a budget with something perplexity cares about.
  Still publishable; more interesting mechanistically.
- Intermediate: if |ΔNLL| ≤ 0.05 but the direction is consistently positive across κ,
  call out the trend honestly even if below threshold.

**Sham check:** κ=1.5 applied to sham heads (matched conformal heads NOT in the target set).
Prediction: |NLL_sham − NLL_baseline| < |NLL(κ=1.5) − NLL(κ=1.0)|.
If the sham condition matches the target condition, the perplexity signal is not head-specific.

---

## Execution

Script: `run_perplexity.py`

```
.venv/bin/python3 research/physics/experiments/exp-075_tradeoff/run_perplexity.py
```

Output: `results.json` in this directory.
Runtime estimate: ~30–60 min on M5 Max MPS (model load + projectors + 5 × 50 forward passes in bf16).

---

## Result

*(pending)*

---

## Interpretation

*(pending)*
