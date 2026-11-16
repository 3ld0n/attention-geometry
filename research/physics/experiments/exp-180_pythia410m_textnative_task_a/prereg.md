# exp-180 Pre-registration
# Pythia-410m text-native Δ-window heads: entity-state tracking (Task A)
# 2026-11-16, solo physics room, ~12:30 AM MST

---

## Context

Three experiments have established the negative result for Pythia-410m's 16
text-native Δ-window heads (from exp-118):

| Experiment | Battery | H_positive result |
|---|---|---|
| exp-177 | Task C, old battery (14/20 multi-token) | 2/16 degrade |
| exp-178 | Task C, native battery (all single-token) | **0/16 degrade** (H_arch confirmed) |
| exp-179 | Task B, single-token | **1/16 degrade** (H_positional falsified) |

Additional finding from exp-178/179: 8/16 text-native heads actively **suppress**
content retrieval (ablation improves Task C), and the dominant competitor (L8H15)
suppresses both content and positional retrieval. L12H13 selectively suppresses
positional retrieval.

The open question: **What positive function do these heads carry?**

Tasks C (content-retrieval) and B (positional-retrieval) have been tested and
produced nulls. Task A (entity-state tracking / world-model coherence) is the
remaining candidate from the standard three-task battery.

---

## Hypothesis

Population: 16 text-native Δ-window heads in Pythia-410m (from exp-118):
(L0,H3), (L1,H8), (L2,H8), (L3,H2), (L3,H7), (L3,H14), (L4,H12), (L7,H11),
(L8,H15), (L9,H15), (L12,H3), (L12,H13), (L14,H15), (L19,H15), (L21,H11), (L22,H1)

**H_entity:** ≥10/16 text-native Δ-window heads carry entity-state tracking
function. Operationalization: individual W_K ablation on 20-item Task A battery;
head carries entity function when ΔP_A < −0.05 nats (ablation degrades entity
tracking) AND ≥12/20 items degraded.

**H_graded:** CV of per-head |ΔP_A| ≥ 0.5 (some heads contribute much more
than others).

**H_competitor_entity:** Among the 8 competitor heads from exp-178 (heads where
ablation improved Task C: L0H3, L2H8, L3H2, L3H7, L7H11, L8H15, L9H15, L19H15),
≥5/8 carry entity-state function (ΔP_A < −0.05). Fires if the competitor structure
serves entity tracking.

---

## Declared prior

Based on two null results (exp-178: 0/16 carry Task C function; exp-179: 1/16 carry
Task B function), I predict the entity-tracking test will also produce a null:

**H_arch_full (declared prior):** Pythia-410m text-native Δ-window heads carry no
positive function in any of the three tested categories. H_entity falsified.

Expected: H_entity dead (K2 fires or near-null pattern), H_graded confirmed
(CV high, reflecting heterogeneous near-zero/positive effects), H_competitor_entity
not fired.

Confidence: moderate-high for the null. Two independent null results with different
task designs establish the pattern. If H_entity confirms, this would be the first
positive function identified for this population in RoPE and would motivate a
revised understanding of how the geometric class maps to function across PE types.

---

## Instrument: Individual W_K ablation on Task A

**Battery:** task_a_items.json (20 items, all targets verified single-token in
GPT-NeoX tokenizer before this file was committed). Format: short passage with
named entity undergoing explicit state changes; cloze prompt at end; target is
the final state. See `task_a_items.json`.

**Protocol:** Same W_K zeroing protocol as exp-178 and exp-179.
- Load Pythia-410m (float32, MPS if available)
- For each of the 16 text-native heads:
  - Baseline: log-prob of target token at the cloze position
  - Ablation: set W_K to zero, re-run; ΔP_A = log-prob_abl − log-prob_base
  - Sham: set W_K to random Gaussian N(0, σ_K / sqrt(d_K)), re-run; validates
    that change is due to ablation not perturbation size
- Report: ΔP_A per head (mean over 20 items), item-level counts

**Decision criteria:**
- H_entity: mean ΔP_A < −0.05 AND item count degraded ≥12/20 for ≥10/16 heads
- H_graded: CV(per-head |ΔP_A|) ≥ 0.5
- H_competitor_entity: ≥5/8 competitor heads (exp-178 list) have ΔP_A < −0.05

**Kill conditions:**
- K1: Competitor structure also suppresses entity tracking — ablation improves
  Task A for ≥5/8 competitor heads (ΔP_A > +0.05). Would extend the competitor
  suppression to a third task.
- K2: Near-null — <5 degraders AND <5 improvers across all 16 heads.
- K3: Sham ≥ ablation effect for the dominant heads (|ΔP_A_sham| ≥ |ΔP_A_abl|
  for ≥10/16 heads). Would indicate measurement artifact.

---

## What each outcome changes

**H_entity confirmed:** The geometric class predicts entity-state tracking in RoPE
transformers, even when it predicts neither content-retrieval nor positional-retrieval.
A distinctive functional identity: the same geometric attractor maps to world-model
coherence in RoPE and to content-retrieval in AbsPE. Would motivate a cross-
architecture design question: does PE type route the geometric attractor to
architecturally available functions, with entity tracking as the RoPE-compatible mode?

**H_entity falsified (H_arch_full confirmed):** The three-population anatomy's
functional layer is scoped to GPT-2/AbsPE. The geometric attractor (Δ ≈ 1/4) predicts
no tested functional role in Pythia-410m. The competitor structure (8/16 heads suppress
Task C) remains functionally uncharacterized after three task types. "Text-native
Δ-window heads" is a geometric classification with no demonstrated functional
correlate in RoPE. The honest conclusion: this population's function in Pythia-410m
is unknown after three instruments, and may require a novel task design or a
mechanistic approach beyond ablation.

---

*Pre-registration committed to attention-geometry before run.py is written.*
*Git-attestation: this commit.*
