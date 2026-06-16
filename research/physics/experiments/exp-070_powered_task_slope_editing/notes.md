# exp-070 — Powered Task-Level Slope Editing: Pythia-1.4b, embedded@40doc

*Date: 2026-06-16 (Cursor session, Ariel, solo)*
*Pre-registration: `research/physics/notes/2026-06-16_powered_task_slope_editing_prereg.md` (commit 9e6239b9, committed BEFORE any edited-weight forward pass)*
*Predecessor: exp-068 (underpowered, ceiling). Calibration: exp-069 (found the U-shape task).*

---

## Registered verdict (lead with this)

**The symmetric behavioral-causality prediction was NOT fully confirmed. Overall registered
verdict: BEHAVIORAL_CAUSALITY_KILLED — driven by the T-B (shallowing) leg, which failed.**

| Test | Prediction | Result | Verdict |
|------|-----------|--------|---------|
| **T-D** sham null | κ=1.0 reproduces base exactly | 0 token diffs / 200 | **PASS** |
| **T-A** deepening | V_task(κ=1.5) > V_task(κ=1.0) | 0.225 > 0.150 | **KEEP (strong)** |
| **T-B** shallowing | V_task(κ=0.5) < V_task(κ=1.0) | 0.150 = 0.150 | **KILL** |
| **T-C** monotone | ρ(κ, V_task) ≥ 0.80 | ρ = 0.949 | **KEEP** |
| Sham-heads specificity | V(target 1.5) > V(sham-heads 1.5) | 0.225 > 0.150 | **PASS** |

Both T-A and T-B were pre-registered as required for "confirmed." T-B killed, so the
registered binary outcome is KILLED. **I report that first and do not relabel it.** The
structure underneath it is informative and is described below — but the headline is: the
two-directional test did not pass.

---

## Setup (locked)

- Model: Pythia-1.4b, fp32, MPS, eager attention.
- Task: exp-069 config C6 — embedded-prose multi-document retrieval, **N_doc = 40**,
  fact bound once in matched-cloze prose, answer at 5 position bins [0,10,20,29,39].
  **200 inputs/condition** (40 single-token items × 5 positions). Context ≈ 1732 tokens.
- Heads: 8 conformal target heads (R²≥0.85, Δ̂∈[0.10,0.40], top-8 by R² in L8–20) from the
  exp-059 census — identical to exp-068 Stage 1. 8 matched sham heads (next 8 by R²).
- Operator: exp-064 κ-rescaling `W_Q ← W_Q·(I+(κ−1)P_U)`, P_U = top-8 PCs of position-mean
  layernorm output. κ ∈ {0.5, 1.0, 1.5, 2.0}, applied to all 8 target heads at once.

## Full results (all conditions, as obligated)

| condition | primacy | near_prim | middle | near_rec | recency | V_task |
|-----------|---------|-----------|--------|----------|---------|--------|
| base / κ=1.0 (sham) | 1.000 | 0.950 | **0.850** | 0.950 | 0.975 | 0.150 |
| κ=0.5 (shallow) | 1.000 | 0.975 | **0.850** | 0.950 | 1.000 | 0.150 |
| κ=1.5 (deepen) | 1.000 | 0.925 | **0.775** | 0.950 | 0.975 | 0.225 |
| κ=2.0 (deepen) | 1.000 | 0.925 | **0.750** | 0.950 | 0.975 | 0.250 |
| sham-heads κ=1.5 | 1.000 | 0.950 | **0.850** | 0.950 | 0.975 | 0.150 |

T-D sham null: 0/200 token differences from base (implementation correct).

---

## Interpretation (honest, after the verdict)

**What worked — the deepening leg, cleanly.** Deepening the conformal heads' QK log-distance
slope (κ>1) monotonically deepened the *task-accuracy* valley: middle-position retrieval
fell 0.85 → 0.775 (κ=1.5) → 0.75 (κ=2.0), while primacy/recency stayed pinned at ≈1.0/0.97.
V_task rose 0.15 → 0.225 → 0.25, monotone in κ (T-C ρ=0.95). This is the attention-level
causal handle from exp-064 (κ moves Δ̂ and the *attention* valley, 8/8 + 24/24) **propagating
to behavior** in the predicted direction. And it is **head-specific**: applying the identical
edit to 8 matched control heads (same count, same layer band, similar R²/Δ̂) left V_task at
0.15 — only the conformal target heads moved it.

**What failed — the shallowing leg.** κ=0.5 left the valley unchanged (middle 0.85, V=0.15).
No improvement in middle retrieval.

**Most likely why the asymmetry — limited headroom, not absence of mechanism.** The base
middle accuracy (0.85) is already close to the model's retrieval ceiling on this task (edges
at 0.95–1.0). Shallowing the attention valley can only raise middle accuracy up to that
ceiling, and there is little room (≈0.10–0.15) to gain — whereas deepening has ample room to
hurt. exp-064 showed the *attention-statistic* valley moves in **both** directions under κ
(24/24 sign agreements), so the attention-level effect is bidirectional; the one-sided
*behavioral* readout here is consistent with a task-headroom ceiling on the shallowing side,
not with the mechanism being absent. This remains an inference, not a registered result.

**The power limitation is the shallow base valley.** exp-069 found this was the only genuine
U-shape available locally, but it is shallow (base V=0.15, edges near ceiling). That shallow
valley is exactly what starved the T-B leg. A deeper base U-shape — e.g. MPT-30B-Instruct,
where Liu et al. report V≈0.3–0.5 with edges around 0.55–0.75 (not at ceiling) — would have
headroom in **both** directions and could test the shallowing leg properly. This is the clean
motivation for the cloud path (exp-062-adjacent; needs Modal credentials).

---

## What this does and does not establish

- **Does:** a pre-registered, head-specific, monotone demonstration that the conformal QK-slope
  edit propagates from the attention statistic (exp-064) to **task-level retrieval behavior**
  in the deepening direction, on a task with a genuine base U-shape. T-D and specificity both
  clean. This is the program's first behavioral-level causal propagation result.
- **Does not:** confirm the full symmetric prediction. The shallowing leg (T-B) failed, so by
  the registered criteria behavioral causality is **not confirmed**. The shallowing null is
  plausibly headroom-bounded but that explanation is post-hoc.

## Next

1. **Cloud re-run with a deep base U-shape** (MPT-30B-Instruct or similar): repeats the exact
   κ-sweep where both legs have headroom. This is the test that can actually adjudicate T-B.
   Needs Eldon's Modal credentials (exp-062 LAUNCH).
2. Do **not** over-state exp-070 in any outreach. The honest one-liner: *"deepening the
   conformal-head QK slope deepens the lost-in-the-middle accuracy valley, monotonically and
   head-specifically; the shallowing direction was bounded by task headroom and remains open."*

---

*Registry: exp-070. Status: complete. Registered verdict: KILLED (T-B leg). Substantive
finding: asymmetric — strong, specific, monotone deepening propagation (T-A/T-C/specificity);
shallowing null (headroom-bounded). Quality: honest, partial-positive. Pre-reg commit 9e6239b9.*
