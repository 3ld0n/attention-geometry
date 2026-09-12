# exp-141 Pre-Registration — World-Model Battery: Corrected ln_1 Hook Protocol

**Date:** 2026-09-12
**Author:** Ariel (Cursor, solo)
**Registered before:** any code written; this file pushed to the public repo first.

---

## Context and motivation

exp-140 (pre-reg 10b7bb4, September 12 2026) ran the world-model battery with
a protocol deviation: the positional field used to compute κ̃ was captured on
the raw residual stream `h` at the attention block input, **before** `ln_1` is
applied internally. GPT-2's attention block applies `ln_1(h)` before projecting
to Q, K, V — so the heads' actual read maps see `ln_1(h)`, not `h`. The
captured field was therefore the wrong tensor.

**Consequence:** κ̃ baselines in exp-140 (0.4–3.5) are inconsistent with
exp-137 (0.05–0.33), which computed the field correctly. The selectivity
finding (Task B 19/20, Task A 10/20) is real regardless, but the intervention
was not precisely targeting the subspace the heads see.

**Fix:** Hook the output of `model.transformer.h[layer].ln_1` using a forward
hook. This captures `ln_1(h)` directly, the tensor the Q/K/V projections
receive. With this fix, κ̃ values should be consistent with exp-137's 0.05–0.33
range for the Δ-window heads.

---

## Hypothesis

**H1 (selectivity preserved):** The directional signal from exp-140 — κ-amplification
on 5 Δ-window heads selectively moves positional retrieval (Task B) without
touching entity-state tracking (Task A) — survives the corrected protocol.

**H2 (effect size larger):** With the amplification precisely targeting `ln_1(h)`,
the Task B effect size ΔP_B may be larger than exp-140's +0.26 nats (direction
preserved; magnitude free to be smaller or larger — not a kill if smaller, only
informative).

---

## Decision criteria (same as exp-140 registration, 10b7bb4)

| Label | Criterion | Meaning |
|---|---|---|
| **P1** (coherence degrades) | ΔP_A < −1.0 nats AND ΔP_A < ΔP_B − 1.5 nats | Coherence hypothesis confirmed |
| **P2** (retrieval improves) | ΔP_B > +1.0 nats AND ΔP_B > ΔP_A + 1.5 nats | Retrieval hypothesis confirmed |
| **P3** (null) | Both ΔP_A and ΔP_B within 0.5 nats of sham | No detectable effect |
| **K1** (sham > amp) | \|sham_A\| > \|amp_A\| OR \|sham_B\| > \|amp_B\| | Intervention weaker than sham — protocol failure |
| **K2** (floor) | baseline < −10 nats on either task | Floor effect — task too hard |
| **K3** (κ failure) | κ̃(amp) < 0.5 on ≥ 2/5 heads after amplification | Amplification failed |

**Direction expected (from exp-140 signal):** P2, not P1. Outcome must be judged
by the registered criteria regardless.

---

## Protocol

**Model:** GPT-2 small (gpt2), same as exp-140.

**Heads targeted:** Same 5 Δ-window structural heads as exp-140:
- L2H1, L3H4, L5H0, L7H11, L10H8

**Hook correction:** Register a **forward hook** on `model.transformer.h[layer].ln_1`
to capture the output of that layer-norm module. The hook receives `(module, input, output)`;
use `output` — this is `ln_1(h)`, the tensor the Q/K/V linear projections see.
Store as `hook_cache[(layer, 'ln1_out')]`.

**Positional field computation:** Use the same frozen random-token census protocol
as exp-112 and exp-137:
- SEQ_LEN = 512, N_INPUTS = 50, SEED = 42
- `x̄_i = mean over inputs of ln1_out[i]` (position-mean vector)
- `δ_i = x̄_i − mean_i(x̄_i)` (centered deviation)
- Positional subspace P_U: top-8 PCA components of the stack [δ_i for i=0..511]
- κ̃_K for head (ℓ,h): `‖W_K P_U P_U^T‖_F / ‖W_K‖_F` (using ln_1 output subspace)

**Amplification:** Same formula as exp-140:
- W_Q_modified = W_Q + (γ−1) · W_Q · P_U · P_U^T, with γ = 2 (unchanged)
- Sham: W_Q_modified = W_Q + (γ−1) · W_Q · P_V · P_V^T, where P_V is the
  complementary (non-positional) subspace (bottom components)

**Task battery:** Same as exp-140 — 20 items each.
- Task A: entity-state tracking (cloze, consistent with entity across context)
- Task B: positional retrieval (list-lookup at stated numerical position)

**Scoring:** Same as exp-140. Log-probability of the correct token. Report
median ΔP across items; binomial item-level test against p=0.5 null.

**Expected κ̃ baseline:** With ln_1(h) hook, should be consistent with exp-137
range 0.05–0.33. If baseline κ̃ > 0.5 on any structural head, re-examine the
hook placement before proceeding.

---

## Kill conditions and ambiguity resolution

- If κ̃ baseline is still in the 0.4–3.5 range (exp-140 territory), the hook
  correction failed — abort, diagnose, do not record as the corrected result.
- If the hook correction is confirmed (κ̃ ∈ [0.05, 0.40]) and P3 fires (null),
  this is evidence the exp-140 selectivity was real but the threshold was tight.
  Report as P3, not as failure of the correction.
- The direction of any effect is informative regardless of threshold.

---

## Connection to exp-140

- exp-140 pre-reg: commit 10b7bb4
- exp-141 is not a new hypothesis — it is a corrected execution of the same
  hypothesis. The selectivity direction (P2) found in exp-140 is the prior;
  this experiment tests it at the correct targeting precision.

*This file is the pre-registration. No run.py has been written; no results.json exists.
Commit and push before writing any analysis code.*
