# exp-142 — Pre-registration: World-model battery suppression (steep/local heads)

**Pre-registration commit: [to be filled after push]**
**Registered before any run.py is written.**
**Ariel — September 13, 2026, ~12:30 AM MDT. Solo.**

---

## 1. Context and motivation

exp-140 and exp-141 established a clean item-level double dissociation via
amplification of the five Δ-window heads (STRUCTURAL = L2H1, L3H4, L5H0, L7H11,
L10H8):

- κ-amplification (γ = 2.0) on Δ-window heads improved **20/20 Task B items**
  (positional retrieval; p < 10⁻⁵) while leaving **11/20 Task A** (entity-state
  tracking; p = 0.83, noise).
- The median threshold (+1.0 nats P2) was not met (ΔP_B = +0.27 nats), but the
  item-level double dissociation is decisive.

exp-137 identified a separate population of heads with very high positional read
gain (κ̃_K >> 1): the steep/local heads. These are the heads whose W_K matrices
project strongly onto the positional field δ (the centered position-mean ln_1(h)
field). Their κ̃_K values from exp-137 are:

| Head | κ̃_K (exp-137) | Type |
|---|---|---|
| L0H10 | 21.7 | local/positional |
| L10H5 | 14.2 | local/positional |
| L8H7 | 9.2 | local/positional |
| L7H0 | 7.3 | local/positional |
| L7H9 | 6.2 | local/positional |
| L11H10 | 5.2 | local/positional |

exp-141's notes §4 (Route B) proposed suppressing these heads as the dissociation
test from the opposite direction.

---

## 2. Hypothesis

Suppressing the positional read of the five highest-κ̃_K steep/local heads
(L0H10, L10H5, L8H7, L7H0, L7H9; γ = −1.0 applied to W_K_proj) **selectively
degrades positional retrieval (Task B) without affecting entity-state tracking
(Task A)**. This is Route B of the exp-141 two-direction test.

The mechanism: these heads' W_K matrices project strongly onto the positional field;
removing that projection prevents them from attending to position. If positional
retrieval depends on this population, Task B should degrade. If entity-state
tracking is independent of positional encoding in this population, Task A should
be unaffected.

---

## 3. Target heads and suppression protocol

**Target heads (5 top-κ̃_K steep/local heads from exp-137):**
```
STEEP_LOCAL = [(0, 10), (10, 5), (8, 7), (7, 0), (7, 9)]
```
L11H10 (κ̃_K = 5.2) excluded to keep the number manageable and avoid overlap
with induction heads whose role may be different.

**Suppression:**
- Same ln_1 hook as exp-141 (corrected protocol) to compute the positional field δ
- Compute top-4 PC directions P_k of δ at each head's layer
- W_K_proj = P_k.T @ (P_k @ W_K)  — the positional projection
- **Suppressed:** W_K_sup = W_K + (−1.0) × W_K_proj = W_K − W_K_proj
  (i.e., γ = −1.0, removing the positional component entirely)
- **Sham:** matched Frobenius norm in the orthogonal complement of the
  positional subspace (same sham design as exp-141)

**Tasks:** identical to exp-141 — Task A (entity-state tracking, 20 items) and
Task B (positional retrieval, 20 items).

**Model:** GPT-2 small (openai-community/gpt2, same as exp-140/141).

---

## 4. Predictions

| Prediction | Criterion |
|---|---|
| **P1** Task B degrades selectively | ΔP_B < −1.0 nats AND ΔP_B < ΔP_A − 1.5 nats |
| **P2** Task A is preserved | |ΔP_A − ΔP_A_sham| < 0.5 nats |
| **P3** null — no effect or non-selective | both within 0.5 nats of sham, OR both degrade |

P1 + P2 together constitute the double dissociation from the suppression direction:
the steep/local heads are load-bearing for positional retrieval but not for entity-state tracking.

---

## 5. Kill conditions

| Kill | Criterion | Interpretation |
|---|---|---|
| **K1** | |Δ_sham| ≥ |Δ_sup| on both tasks | Suppression failed (sham effect ≥ intervention) |
| **K2** | orig_A < −10 nats OR orig_B < −10 nats | Baseline at floor — degradation unmeasurable |
| **K3** | κ̃_K(sup) > 1.0 on ≥ 2/5 target heads | Suppression insufficient (positional component not removed) |

Note on K3: by construction (W_K_sup = W_K_perp, the projection-removed complement),
κ̃_K(sup) should be near 0 for all five heads. K3 guards against implementation
error (wrong tensor, wrong layer index, etc.).

---

## 6. Analysis and outcome table

After the run:
1. Compute κ̃_K for each target head before and after suppression (baseline check)
2. Score Task A and Task B for original, suppressed, and sham models
3. Compute median log-prob and per-item improvement counts
4. Evaluate P1, P2, P3, K1, K2, K3

| Outcome | Interpretation |
|---|---|
| P1 fires + P2 fires | **Double dissociation confirmed (Route B).** Steep/local heads carry positional retrieval but not entity tracking. Combined with exp-141: Δ-window amplification improves retrieval; steep/local suppression degrades retrieval; neither moves entity tracking. |
| P1 fires + P2 does not | Task B degrades, Task A also degrades. Steep/local heads involved in both — coherence hypothesis gains ground. |
| P3 fires (null) | Suppression did not affect either task. Steep/local heads not individually necessary for retrieval (redundant with other mechanisms). |
| K1/K2/K3 fires | INCONCLUSIVE — mechanical failure. |

---

## 7. Connection to the record

- **Follows from:** exp-141 (Route B suggestion in §4); exp-137 (target head selection)
- **Bears on:** P1 (the functional role of conformal-window heads)
- **Pre-registration required before run.py:** yes — this is registered before any
  code is written, per the room's standing rule (checked by physics_coherence)
