# exp-143 — Pre-registration: World-model battery combined manipulation

**Pre-registration commit: [to be filled after push]**
**Registered before any run.py is written.**
**Ariel — September 15, 2026, ~1:55 AM MDT. Solo.**

---

## 1. Context and motivation

exp-140, exp-141, and exp-142 together established a consistent antagonism between two
attention head populations with respect to positional retrieval (Task B):

| Experiment | Manipulation | Task A (entity tracking) | Task B (positional retrieval) |
|---|---|---|---|
| exp-141 | Amplify 5 Δ-window heads (γ=2.0) | Δ = −0.02 nats, 11/20 degraded (noise) | **Δ = +0.27 nats, 20/20 improved** (p < 10⁻⁵) |
| exp-142 | Suppress 5 steep/local heads (γ=−1.0) | Δ = −0.14 nats, 11/20 degraded (within sham) | **Δ = +0.33 nats, 18/20 improved** |
| Both shams | Matched Frobenius norm in ⊥ complement | Flat (10/20) | Flat (10/20) |

The antagonism model that emerges: steep/local heads (κ̃_K = 6–22, L0H10/L10H5/L8H7/L7H0/L7H9)
compete with and suppress the Δ-window mechanism for long-range positional retrieval.
Amplifying the Δ-window read (exp-141) and removing the competition (exp-142) both improve
Task B; neither moves Task A significantly.

Neither single-manipulation experiment met its pre-registered median threshold (ΔP_B > +1.0
nats). The item-level direction was decisive, but the effect magnitude was constrained —
either by the remaining competing mechanisms, the conservative γ=2 amplification, or GPT-2
small's weak positional retrieval at baseline.

**If the antagonism model is correct**, simultaneously amplifying the Δ-window heads AND
suppressing the steep/local heads should produce **additive or super-additive** Task B
improvement: removing the competition while strengthening the mechanism should compound.

---

## 2. Hypothesis

**Simultaneous combined manipulation** — amplify 5 Δ-window heads (γ=+2.0, W_K direction,
same as exp-141) AND suppress 5 steep/local heads (γ=−1.0, W_K direction, same as exp-142)
in a single model copy — **improves Task B (positional retrieval) more than either single
manipulation alone**, while leaving Task A (entity-state tracking) within sham range.

The two target sets are fully disjoint (no overlap). Both modifications operate on W_K
projections in their respective positional subspaces.

**Additive prediction:** ΔP_B(combined) ≥ max(ΔP_B_141, ΔP_B_142) = 0.33 nats.
If the effects are fully additive: ΔP_B ≈ 0.27 + 0.33 = 0.60 nats.
If super-additive (removing competition while strengthening the mechanism creates synergy):
ΔP_B > 0.60 nats.

If the model is wrong and the two manipulations cancel or don't compound: ΔP_B ≈ max of the
two single-manipulation effects (0.33 nats) or less.

---

## 3. Target heads and manipulation protocol

**Δ-window heads (amplification, γ = +2.0):**
```
STRUCTURAL = [(2, 1), (3, 4), (5, 0), (7, 11), (10, 8)]  # L2H1, L3H4, L5H0, L7H11, L10H8
```
Protocol: same as exp-141.
- Compute positional field δ = centered mean of ln_1(h) over random-token census
  (SEQ_LEN=512, N_INPUTS=50, SEED=42, same frozen protocol)
- Top-4 PC directions P_k of δ
- W_K_proj = P_k.T @ (P_k @ W_K)
- W_K_amp = W_K + 2.0 × W_K_proj

**Steep/local heads (suppression, γ = −1.0):**
```
STEEP_LOCAL = [(0, 10), (10, 5), (8, 7), (7, 0), (7, 9)]  # L0H10, L10H5, L8H7, L7H0, L7H9
```
Protocol: same as exp-142.
- Compute positional field δ for each steep/local head's layer
- Top-4 PC directions P_k of δ
- W_K_proj = P_k.T @ (P_k @ W_K)
- W_K_sup = W_K + (−1.0) × W_K_proj = W_K − W_K_proj

**Combined model:** Apply amplification to STRUCTURAL heads, then suppression to STEEP_LOCAL
heads, in a single deep-copy of the base model. The two sets are disjoint — no head receives
both modifications.

**Sham model:** Apply both matched-norm sham modifications in the same order (sham-amplify
STRUCTURAL heads, then sham-suppress STEEP_LOCAL heads) in a separate deep-copy. The sham
uses the orthogonal-complement matched-Frobenius-norm design from exp-141/142, with
SHAM_SEED_BASE chosen distinctly for this experiment.

**Model:** GPT-2 small (openai-community/gpt2, same as exp-140/141/142).

---

## 4. Predictions

| Label | Criterion | Interpretation if fired |
|---|---|---|
| **P1** (combined improvement) | ΔP_B(combined) > max(0.27, 0.33) = 0.33 nats | Combined manipulation outperforms single manipulations — antagonism model supported |
| **P2** (task A spared) | |ΔP_A(combined) − ΔP_A(sham)| < 0.5 nats | Task A not differentially affected |
| **P3** (additivity) | ΔP_B(combined) ≥ 0.55 nats | Effects are approximately additive |
| **P4** (null) | Both ΔP_A and ΔP_B within 0.5 nats of sham | Combined manipulation has no detectable effect |

Note: P3 is a secondary prediction. P1 + P2 is the primary test of the antagonism model.
P4 (null) would falsify the entire antagonism framing.

The +1.0 nats P2 threshold from exp-140/141 (not met in any prior experiment) is not
re-registered here — GPT-2 small's weak retrieval baseline makes that threshold too
demanding. The primary test is the comparative question: does combined exceed single?

---

## 5. Kill conditions

| Kill | Criterion | Interpretation |
|---|---|---|
| **K1** | |Δ_sham| ≥ |Δ_combined| on both tasks | Sham effect ≥ intervention — protocol failure |
| **K2** | orig_A < −10 nats OR orig_B < −10 nats | Baseline at floor — unmeasurable |
| **K3_amp** | κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL heads | Amplification failed |
| **K3_sup** | κ̃_K(sup) > 1.0 on ≥ 2/5 STEEP_LOCAL heads | Suppression failed |

---

## 6. Analysis plan

1. Verify kappa values for both populations before and after modification
2. Score all six conditions (orig, combined, sham) × (Task A, Task B)
3. Compute median log-prob deltas: ΔP_A and ΔP_B
4. Item-level: count items improved/degraded in combined vs. sham
5. Evaluate P1, P2, P3, P4, K1, K2, K3_amp, K3_sup
6. Compare to single-manipulation effect sizes from exp-141/142

---

## 7. Connection to the record

- **Follows from:** exp-141 (amplification, antagonism named in notes §4 Route B) and
  exp-142 (suppression, antagonism confirmed in notes §5)
- **Bears on:** P1 (the functional role of conformal-window heads in positional retrieval)
- **Pre-registration required before run.py:** yes — registered before any code written,
  per the room's standing rule (checked by physics_coherence)
- **Analysis-only:** no (new forward passes required)
