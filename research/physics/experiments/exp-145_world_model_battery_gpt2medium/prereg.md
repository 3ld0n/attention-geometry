# exp-145 — Pre-registration: World-model battery combined manipulation, GPT-2 medium

**Pre-registration commit: (to be inserted after push)**
**Registered before run.py is written.**
**Ariel — September 17, 2026, ~5:05 AM MDT. Solo.**

---

## 1. Context and motivation

exp-141, exp-142, and exp-143 established a three-directional antagonism result in GPT-2
small: Δ-window heads and steep/local heads are antagonistic with respect to long-range
positional retrieval (Task B), while entity-state tracking (Task A) is spared. The combined
manipulation (amplify + suppress simultaneously, exp-143) produced ΔP_B = +0.71 nats,
20/20 Task B items improved — the strongest result in the series.

**The current evidence bracket:** GPT-2 small only.

This experiment extends the P1 causal test to **GPT-2 medium** (24 layers, 16 heads,
D_model=1024; 345M parameters). The question: does the antagonism pattern hold in a larger
model from the same family with the same architecture class (learned positional encoding)?

GPT-2 medium has 59 Δ-window heads (exp-118, WikiText-native census), compared to 16
text-native Δ-window heads in GPT-2 small. The κ̃_K structure has been characterized in
this session (characterize_kappa.py, results saved to `kappa_characterization.json`). Key
finding: Δ-window heads have significantly lower κ̃_K than control heads (median 2.10 vs
3.66, Mann-Whitney U p = 2.87×10⁻⁶), replicating the population structure seen in GPT-2
small. The absolute κ̃_K values are higher across the board than GPT-2 small — consistent
with a larger model's more structured positional representations.

GPT-2 medium may also be a better positional retriever at baseline than GPT-2 small (which
had Task B median logP ≈ −5.62 and never met the +1.0 nats threshold). If so, this
experiment may be the first to fire the magnitude criterion registered but never met in
exp-140–143.

---

## 2. Hypothesis

**Simultaneous combined manipulation** — amplify 5 Δ-window heads (γ=+2.0, W_K direction)
AND suppress 5 steep/local heads (γ=−1.0, W_K direction) in GPT-2 medium — **improves
Task B (positional retrieval) more than sham**, while leaving Task A (entity-state tracking)
within sham range.

The protocol is identical to exp-143 in GPT-2 small, with target heads adapted to the
GPT-2 medium κ̃_K landscape (characterize_kappa.py, same SEED=42, SEQ_LEN=512, N_INPUTS=50).

If the antagonism model generalizes beyond GPT-2 small, the same manipulation structure
should produce the same qualitative pattern: selective Task B improvement, Task A spared.

---

## 3. Target heads and manipulation protocol

**Δ-window heads for amplification (γ = +2.0):**
```
STRUCTURAL = [(7, 5), (3, 12), (9, 7), (8, 13), (7, 15)]
# L7H5, L3H12, L9H7, L8H13, L7H15
# κ̃_K: 0.257, 0.304, 0.417, 0.469, 0.556 (5 lowest among 59 wiki heads)
```
Selection criterion: 5 Δ-window heads with the lowest κ̃_K, most analogous to exp-143's
structural population. All are WikiText-native Δ-window heads (exp-118).

**Steep/local heads for suppression (γ = −1.0):**
```
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
# L4H13, L15H8, L8H7, L5H11, L11H7
# κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9 (5 highest κ̃_K, non-Δ-window)
```
Selection criterion: 5 non-Δ-window heads with the highest κ̃_K. None are WikiText-native
Δ-window heads (confirmed by characterize_kappa.py). No overlap with STRUCTURAL set.

**Protocol (identical to exp-143 except model and head lists):**
- Load GPT-2 medium (openai-community/gpt2-medium)
- For each target head's layer: compute positional field δ = centered mean of ln_1(h)
  output over N_INPUTS=50 random-token sequences of SEQ_LEN=512, SEED=42
- P_k = top-4 PC directions of δ (SVD of δ)
- Amplification: W_K_amp = W_K + 2.0 × (P_k.T @ (P_k @ W_K))
- Suppression: W_K_sup = W_K + (−1.0) × (P_k.T @ (P_k @ W_K))
- Sham: matched Frobenius norm in orthogonal complement of P_k (SHAM_SEED_BASE = 2026091701)
- Combined model: apply amplification to STRUCTURAL, then suppression to STEEP_LOCAL
- Sham model: apply matched sham modifications in same order
- Model dimensions: D_MODEL=1024, D_HEAD=64

Note: all modifications operate on the W_K subspace of c_attn weight. The two target sets
are fully disjoint — no head receives both modifications.

---

## 4. Predictions

| Label | Criterion | Interpretation if fired |
|---|---|---|
| **P1** (improvement) | ΔP_B > 0.33 nats AND ΔP_B > ΔP_A + 0.3 nats | Antagonism pattern replicates in GPT-2 medium |
| **P2** (task A spared) | \|ΔP_A − ΔP_A_sham\| < 0.5 nats | Task A not differentially affected |
| **P3** (item-level) | Task B: ≥ 60% items improved (combined vs sham); Task A: < 60% | Item-level dissociation — stronger than median-level |
| **P4_strong** | ΔP_B ≥ 1.0 nats | Magnitude threshold never met in GPT-2 small fires in GPT-2 medium |
| **P5_null** | Both |ΔP_A| and |ΔP_B| ≤ 0.1 nats | Combined manipulation has no detectable effect — would falsify antagonism model at medium scale |

P1 + P2 is the primary test. P3 (item-level) strengthens P1. P4_strong is aspirational —
not required for confirmation. P5_null (null) would disconfirm generalization.

Primary verdict: CONFIRMED if P1 + P2 fire and K1–K3 do not; DISCONFIRMED if P5_null fires.

---

## 5. Kill conditions

| Kill | Criterion | Interpretation |
|---|---|---|
| **K1** | ΔP_B(sham) ≥ ΔP_B(combined) AND ΔP_A(sham) ≥ ΔP_A(combined) | Sham ≥ intervention — protocol failure |
| **K2** | orig_A < −20 nats OR orig_B < −20 nats | Baseline at floor — unmeasurable |
| **K3_amp** | κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL heads | Amplification failed to increase positional capture |
| **K3_sup** | κ̃_K(sup) > 5.0 on ≥ 2/5 STEEP_LOCAL heads | Suppression failed (note: threshold is higher than GPT-2 small's 1.0, reflecting higher absolute κ̃_K scale in this model) |

Note on K3_sup threshold: GPT-2 small steep/local heads had κ̃_K ≥ 6.2 before suppression
and dropped to <0.45 after. GPT-2 medium steep/local targets start at κ̃_K ≥ 30. A threshold
of 5.0 post-suppression is conservative — any residual above 5.0 on ≥ 2/5 heads would
indicate suppression did not substantially reduce positional gain.

---

## 6. Analysis plan

1. Load GPT-2 medium and verify model identity
2. Run positional field computation + κ̃ verification for all target heads (before/after)
3. Build combined model and sham model
4. Score Task A (20 items) and Task B (20 items) across: original, combined, sham
5. Compute median log-prob deltas ΔP_A, ΔP_B for combined and sham
6. Item-level analysis: count improved/degraded per task
7. Evaluate P1–P5_null, K1–K3_sup
8. Compare effect size to GPT-2 small exp-143 result (ΔP_B = +0.71 nats, 20/20)

---

## 7. Connection to the record

- **Follows from:** exp-143 (GPT-2 small combined manipulation, CONFIRMED); exp-118
  (GPT-2 medium Δ-window census, 59 heads); κ̃_K characterization (this session)
- **Bears on:** P1 (the functional role of conformal-window heads in positional retrieval —
  generality across model scale within same architecture family)
- **Extends:** exp-143's evidence bracket from GPT-2 small to GPT-2 medium
- **Pre-registration required before run.py:** yes — registered before any code is written
- **Analysis-only:** no (new forward passes required; GPT-2 medium locally cached)
