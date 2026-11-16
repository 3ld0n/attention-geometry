# exp-179 Pre-registration
# Pythia-410m text-native Δ-window heads: positional retrieval (Task B)
# 2026-11-15, solo physics room, ~6:50 PM MST

---

## Context

exp-177 (2026-11-13) and exp-178 (2026-11-14) established that Pythia-410m's 16
text-native Δ-window heads (from exp-118) do NOT carry content-retrieval function
(Task C). With a native-tokenizer battery:

- 0/16 heads degrade Task C on W_K ablation (H_positive DEAD)
- 8/16 heads actively IMPROVE Task C on ablation — competitor structure
- Dominant competitor: L8H15 (+0.219 nats, 20/20 items)
- Reversed recency: begin > middle > end (anti-recency, −0.401 nats)

The open question: **What positive function do these heads carry?**

Candidate functions considered:
1. **Positional retrieval (Task B)** — the same function GPT-2 small's structural/
   steep-local heads carry (exp-141, exp-150 et al.); exp-179 tests this directly
2. Syntax/structural function — not tested here
3. Inhibitory gating with no positive function — unlikely given CV = 1.148

exp-179 tests hypothesis 1: do Pythia-410m's text-native Δ-window heads carry
**positional retrieval function**?

---

## Hypothesis

Population: 16 text-native Δ-window heads in Pythia-410m (from exp-118):
(L0,H3), (L1,H8), (L2,H8), (L3,H2), (L3,H7), (L3,H14), (L4,H12), (L7,H11),
(L8,H15), (L9,H15), (L12,H3), (L12,H13), (L14,H15), (L19,H15), (L21,H11), (L22,H1)

**H_positional:** ≥10/16 text-native Δ-window heads carry positional retrieval
function. Operationalization: individual W_K ablation on 20-item Task B battery;
head carries positional function when ΔP_B < −0.05 nats (ablation degrades Task B).
Item count: ≥12/20 items degraded (ΔP_B_item < 0).

**H_graded:** CV of per-head ΔP_B ≥ 0.5 (some heads contribute much more than others).

**H_competitor_null:** Among the 8 competitor heads (from exp-178: heads where ablation
improved Task C), fewer than 5/8 carry positional function (ΔP_B < −0.05). Kill on
H_competitor_null fires if ≥5/8 carry positional function — which would indicate the
competitor structure is role-switching, not pure competition.

---

## Declared prior

Given the competitor structure (ablating text-native heads improves content retrieval
in Pythia-410m), I predict these heads carry positional retrieval function — the same
role that GPT-2 small's structural/steep-local heads carry. The functional identity
switches with architecture class: AbsPE → content-retrieval for text-native Δ-window
heads; RoPE → positional retrieval for the same geometric class.

Expected: H_positional confirmed (≥10/16 heads degrade Task B). H_graded confirmed.
H_competitor_null not fired (≥5/8 competitor heads also carry positional function —
same heads serve positional, compete with content).

Confidence: moderate. The competitor structure is the main evidence. Alternative
(some other function entirely) would be more surprising.

---

## Instrument: Individual W_K ablation on Task B

**Battery:** task_b_items.json (20 items, all targets verified single-token in
GPT-NeoX tokenizer before this file was committed). Format: numbered list of 5
items in a category; probe position varies (1-5); target is the item at that
position. See `task_b_items.json`.

**Protocol:** Same W_K zeroing protocol as exp-178.
- Load Pythia-410m (float32, MPS if available)
- For each of the 16 text-native heads:
  - Baseline: log-prob of target token at the retrieval position
  - Ablation: set W_K to zero, re-run; ΔP_B = log-prob_abl − log-prob_base
  - Sham: set W_K to random Gaussian N(0, σ_K), re-run; validates that change is
    due to ablation not perturbation size
- Report: ΔP_B per head (mean over 20 items), item-level counts

**Decision criteria:**
- H_positional: mean ΔP_B < −0.05 AND item count degraded ≥12/20 for ≥10/16 heads
- H_graded: CV(per-head ΔP_B) ≥ 0.5
- H_competitor_null: <5 of the 8 competitor heads (exp-178 list) have ΔP_B < −0.05

**Kill conditions:**
- K1: Competitor structure also suppresses positional retrieval — ablation improves
  Task B for the competitor heads (ΔP_B > +0.05 for ≥5 competitor heads). Would
  indicate these heads serve some other function (neither content nor positional
  retrieval).
- K2: Near-null — <5 degraders AND <5 improvers. No clear positional function.
- K3: Mean ΔP_B baseline (sham) ≥ ablation effect for the dominant heads. Would
  indicate measurement artifact.

---

## What each outcome changes

**H_positional confirmed:** The functional identity of Δ-window heads is
architecture-conditional — AbsPE → content-retrieval; RoPE → positional-retrieval.
The same geometric class serves different computational roles depending on PE type.
This strengthens the scope limit on the three-population anatomy's functional layer
and raises a theoretical question: does PE type route the same geometric attractor
to different computations, or do RoPE and AbsPE produce geometrically similar but
functionally distinct populations?

**H_positional falsified (K1 or K2):** The geometric classification does not cleanly
predict any of the named functional roles in Pythia-410m. The competitor structure
(exp-178) is then unexplained by a positive-function story. Would call for a Task A
(entity-state tracking) battery as next experiment, or a syntax/structural probe.

---

*Pre-registration committed to attention-geometry before run.py is written.*
*Git-attestation: this commit.*
