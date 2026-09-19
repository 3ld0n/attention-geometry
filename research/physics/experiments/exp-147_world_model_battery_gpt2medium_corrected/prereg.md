# exp-147 — Pre-registration: World-model battery combined manipulation, GPT-2 medium (corrected targets)

**Pre-registration commit: attention-geometry de254d0 (pushed before run.py written)**
**Registered before run.py is written.**
**Ariel — September 19, 2026, ~12:30 AM MDT. Solo.**

---

## 1. Context and motivation

exp-143 (GPT-2 small) confirmed the antagonism model: amplifying random-token structural heads
(γ=+2.0) and suppressing steep/local heads (γ=−1.0) simultaneously produced ΔP_B = +0.71 nats,
20/20 Task B items improved — selective improvement of positional retrieval with Task A spared.

exp-145 attempted to replicate this in GPT-2 medium and found the direction **inverted**
(ΔP_B = −0.25 nats, 0/20 items improved). The root cause was a target-population error:
exp-145 used the WikiText-native Δ-window heads (exp-118, 59 heads, concentrated in L19–L23)
as amplification targets. The GPT-2 small result was built on **random-token structural heads**
(exp-112 equivalent) — a distinct population. The hypothesis that the two populations are
interchangeable as causal intervention targets was falsified by exp-145.

exp-146 (2026-09-18) ran the random-token census on GPT-2 medium using the frozen protocol
(N_INPUTS=50, SEQ_LEN=512, SEED=42, R²≥0.90, Δ∈[0.20,0.30]) and identified 24 structural
heads. The Jaccard with the WikiText-native population is 0.122 (9/24 overlap) — the
populations are largely distinct, with random-token structural heads concentrated in
mid-depth layers (L5–L17) versus WikiText-native heads in deep layers (L19–L23).

The 5 structural heads with the lowest κ̃_K (most purely positional) are:

| Head | Δ | R² | κ̃_K | Wiki? |
|---|---|---|---|---|
| L6H9  | 0.2116 | 0.9036 | 0.208 | No  |
| L5H14 | 0.2612 | 0.9152 | 0.220 | No  |
| L7H5  | 0.2164 | 0.9205 | 0.257 | Yes |
| L9H7  | 0.2027 | 0.9133 | 0.417 | Yes |
| L8H13 | 0.2655 | 0.9300 | 0.469 | Yes |

These are the analogues of the GPT-2 small structural heads used in exp-141/142/143 — the most
"purely positional" conformal-window heads, selected by random-token census rather than
WikiText-native census. The selection criterion is now consistent with exp-143.

The steep/local suppression targets are unchanged from exp-145 (same 5 highest-κ̃_K heads from
the kappa characterization). The protocol correction is in the amplification target only.

---

## 2. Hypothesis

**Simultaneous combined manipulation** — amplify 5 random-token structural heads (γ=+2.0,
W_K direction) AND suppress 5 steep/local heads (γ=−1.0, W_K direction) in GPT-2 medium —
**improves Task B (positional retrieval) relative to sham**, while leaving Task A (entity-state
tracking) within sham range.

The protocol is otherwise identical to exp-143 (GPT-2 small, CONFIRMED additive) and exp-145
(GPT-2 medium, INCONCLUSIVE — wrong targets), with the amplification targets corrected to the
random-token structural population from exp-146.

**If the antagonism model generalizes to GPT-2 medium when the correct target population is used:**
the same manipulation structure should produce the same qualitative pattern as exp-143 —
selective Task B improvement, Task A spared.

---

## 3. Target heads and manipulation protocol

**Structural heads (amplification, γ = +2.0):**
```
STRUCTURAL = [(6, 9), (5, 14), (7, 5), (9, 7), (8, 13)]
# L6H9, L5H14, L7H5, L9H7, L8H13
# κ̃_K: 0.208, 0.220, 0.257, 0.417, 0.469 (5 lowest-κ̃_K random-token structural heads from exp-146)
```
Selection criterion: 5 random-token structural heads (R²≥0.90, Δ∈[0.20,0.30] under frozen
random-token census) with the lowest κ̃_K — most analogous to exp-143's structural population.
Source: exp-146 results.json + kappa_characterization.json from exp-145 session.

**Steep/local heads (suppression, γ = −1.0):**
```
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
# L4H13, L15H8, L8H7, L5H11, L11H7
# κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9 (5 highest κ̃_K, non-Δ-window)
```
Unchanged from exp-145. These are the 5 non-Δ-window heads with the highest κ̃_K in the
GPT-2 medium characterization (kappa_characterization.json). None are in the STRUCTURAL set.

**Protocol (identical to exp-143 and exp-145):**
- Load GPT-2 medium (openai-community/gpt2-medium, attn_implementation="eager")
- For each target head's layer: compute positional field δ = centered mean of ln_1(h) output
  over N_INPUTS=50 random-token sequences of SEQ_LEN=512, SEED=42
- P_k = top-4 PC directions of δ (SVD of δ)
- Amplification: W_K_amp = W_K + 2.0 × (P_k.T @ (P_k @ W_K))
- Suppression: W_K_sup = W_K + (−1.0) × (P_k.T @ (P_k @ W_K))
- Sham: matched Frobenius norm in orthogonal complement of P_k (SHAM_SEED_BASE_AMP = 2026091901,
  SHAM_SEED_BASE_SUP = 2026091910; distinct from all prior experiments)
- Combined model: apply amplification to STRUCTURAL, then suppression to STEEP_LOCAL
- Sham model: apply matched sham modifications in same order
- Model dimensions: D_MODEL=1024, D_HEAD=64, N_LAYERS=24, N_HEADS=16

The two target sets are fully disjoint — no head receives both modifications.

**The one change from exp-145:** STRUCTURAL = [(6,9), (5,14), (7,5), (9,7), (8,13)]
replaces exp-145's STRUCTURAL = [(7,5), (3,12), (9,7), (8,13), (7,15)].
L3H12 and L7H15 (WikiText-native-only) are replaced by L6H9 and L5H14 (not WikiText-native).
L7H5, L9H7, L8H13 appear in both lists — these three were also in exp-145.

---

## 4. Predictions

| Label | Criterion | Interpretation if fired |
|---|---|---|
| **P1** (improvement) | ΔP_B > 0.33 nats AND ΔP_B > ΔP_A + 0.3 nats | Antagonism pattern replicates in GPT-2 medium with correct targets |
| **P2** (Task A spared) | \|ΔP_A − ΔP_A_sham\| < 0.5 nats | Task A not differentially affected |
| **P3** (item-level) | Task B: ≥ 60% items improved (combined vs sham) | Item-level dissociation — strengthens P1 |
| **P4_strong** | ΔP_B ≥ 1.0 nats | Magnitude threshold never met in any prior experiment |
| **P_null** | Both \|ΔP_A\| and \|ΔP_B\| ≤ 0.1 nats | No detectable effect — antagonism model falsified at GPT-2 medium scale even with correct targets |
| **P_direction_exp145** | ΔP_B < 0 (Task B degrades) | Direction-inverted replication of exp-145's finding — population difference is *not* the explanation |

Primary verdict: CONFIRMED if P1 + P2 fire and K1–K3 do not.
INCONCLUSIVE (wrong targets hypothesis not resolved) if P_direction_exp145 fires.
DISCONFIRMED (generalization fails) if P_null fires.

---

## 5. Kill conditions

| Kill | Criterion | Interpretation |
|---|---|---|
| **K1** | ΔP_B(sham) ≥ ΔP_B(combined) AND ΔP_A(sham) ≥ ΔP_A(combined) | Sham ≥ intervention — protocol failure |
| **K2** | orig_A < −20 nats OR orig_B < −20 nats | Baseline at floor — unmeasurable |
| **K3_amp** | κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL heads | Amplification failed |
| **K3_sup** | κ̃_K(sup) > 5.0 on ≥ 2/5 STEEP_LOCAL heads | Suppression failed |

Note: K3_sup threshold is 5.0 (not 1.0 as in exp-143) because GPT-2 medium has higher
absolute κ̃_K values across the board; the GPT-2 small threshold would trivially fire here.
Same threshold as exp-145.

---

## 6. Analysis plan

1. Load GPT-2 medium with attn_implementation="eager" (required for correct MPS forward pass)
2. Run positional field computation + κ̃ verification for all target heads (before/after)
3. Build combined model and sham model
4. Score Task A (20 items) and Task B (20 items) across: original, combined, sham
5. Compute median log-prob deltas ΔP_A, ΔP_B for combined and sham
6. Item-level analysis: count improved/degraded per task
7. Evaluate P1–P_null, K1–K3_sup
8. Compare effect size to exp-143 (GPT-2 small, CONFIRMED additive, ΔP_B = +0.71 nats)
   and to exp-145 (same model, wrong targets, ΔP_B = −0.25 nats)

---

## 7. Connection to the record

- **Follows from:** exp-143 (GPT-2 small combined manipulation, CONFIRMED additive);
  exp-145 (GPT-2 medium wrong targets, INCONCLUSIVE); exp-146 (random-token census on
  GPT-2 medium — identified the correct structural heads)
- **Bears on:** P1 (the functional role of conformal-window heads in positional retrieval —
  generality across model scale within same architecture family; specifically: whether the
  functional role is in the random-token structural population or the WikiText-native population)
- **The causal question exp-145 left open:** the world-model battery direction inverted because
  the wrong population was used. This experiment tests whether using the correct population
  recovers the antagonism effect. If it does, the structural population (not the WikiText-native
  population) is the functional carrier of long-range positional retrieval.
- **Pre-registration required before run.py:** yes — registered before any code is written,
  per the room's standing rule (checked by physics_coherence)
- **Analysis-only:** no (new forward passes required; GPT-2 medium locally cached)
