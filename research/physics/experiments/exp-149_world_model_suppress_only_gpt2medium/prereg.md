# exp-149 — Pre-registration: World-model battery suppress-only, GPT-2 medium

**Registered before run.py is written.**
**Ariel — September 19, 2026, ~4:25 AM MDT. Solo.**

---

## 1. Context and motivation

exp-147 (2026-09-19) ran the corrected combined manipulation in GPT-2 medium
and found the direction still inverted: ΔP_B = −0.13 nats, 2/20 Task B improved.

In GPT-2 small:
- exp-142: suppress-only → ΔP_B = +0.33 nats (18/20 Task B improved; direction
  inverted from predicted degradation — antagonism model confirmed with this sign)
- exp-143: combined → ΔP_B = +0.71 nats (additive)

In GPT-2 medium the combined protocol (exp-147) degraded Task B regardless of which
Δ-window population was used for amplification. This experiment tests whether the
**suppression arm alone** is responsible for the degradation by running suppression
of the steep/local heads without any amplification of structural heads.

The three possible outcomes remain (see exp-148 pre-reg §1). This experiment tests
outcome 2: suppression alone degrades Task B in medium.

Recall from exp-147: the sham improved 16/20 Task B items (item-level), and the
combined manipulation reversed this to 2/20. If suppression alone accounts for this,
suppression-only should reproduce a majority of that reversal.

---

## 2. Hypothesis

**Suppress-only manipulation** — suppress 5 steep/local heads
(L4H13, L15H8, L8H7, L5H11, L11H7; γ=−1.0, W_K direction) in GPT-2 medium,
with NO amplification of structural heads — **improves Task B (positional retrieval)
relative to sham**, analogous to exp-142 in GPT-2 small (+0.33 nats).

The steep/local heads are the 5 highest-κ̃_K non-Δ-window heads from the GPT-2
medium kappa characterization (exp-145 session; κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9).
These are unchanged from exp-145 and exp-147.

---

## 3. Target heads and protocol

**Steep/local heads (suppression, γ = −1.0):**
```
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
# L4H13, L15H8, L8H7, L5H11, L11H7
# κ̃_K: 43.2, 39.9, 33.3, 31.7, 30.9
```

**No amplification of structural heads.** This is the only change from the
combined protocol of exp-147.

**Protocol (identical to exp-142/143/147 suppression arm):**
- Load GPT-2 medium (openai-community/gpt2-medium, attn_implementation="eager")
- Positional field δ: centered mean of ln_1(h) output over N_INPUTS=50 random-token
  sequences, SEQ_LEN=512, SEED=42
- P_k = top-4 PC directions of δ (SVD)
- Suppression: W_K_sup = W_K + (−1.0) × (P_k.T @ (P_k @ W_K))
- Sham: matched Frobenius norm in ⊥ complement of P_k (SHAM_SEED_BASE = 2026091931;
  distinct from all prior experiments)
- Score Task A (20 items) and Task B (20 items): original, suppressed, sham

Model dimensions: D_MODEL=1024, D_HEAD=64, N_LAYERS=24, N_HEADS=16.

Kill threshold K3: κ̃_K(sup) > 5.0 on ≥ 2/5 STEEP_LOCAL heads (GPT-2 medium scale).
Same as exp-145/147 (not 1.0 as in small — absolute κ̃_K scale is 3-4× higher).

---

## 4. Predictions

| Label | Criterion | Interpretation if fired |
|---|---|---|
| **P1** (improvement) | ΔP_B > 0.10 nats | Suppression arm improves Task B in medium (analogous to GPT-2 small exp-142: +0.33 nats) |
| **P2** (Task A spared) | \|ΔP_A − ΔP_A_sham\| < 0.5 nats | Task A not differentially affected |
| **P_null** | \|ΔP_B\| ≤ 0.05 nats | No detectable effect |
| **P_degrade** | ΔP_B < −0.10 nats | Suppression alone degrades Task B — this arm drives exp-147's inversion |
| **P_positive_only** | ΔP_B > 0 | Directional improvement, even if below P1 threshold |

Primary diagnostic interpretations:
- **P_degrade fires:** suppression arm is the culprit in exp-147's inversion.
  The over-suppression candidate (κ̃ from ~43 to <0.1 — 3-4× larger scale than
  GPT-2 small's κ̃ ~10 → 0.2) is the mechanism. Next: test with reduced γ.
- **P1 fires (improvement):** suppression arm is NOT the culprit. Amplification
  (exp-148) or their interaction drives the combined degradation.
- **P_null fires:** suppression arm is inert in medium; combined degradation must
  come from amplification or interaction.

---

## 5. Kill conditions

| Kill | Criterion | Interpretation |
|---|---|---|
| **K1** | ΔP_B(sham) ≥ ΔP_B(sup) AND ΔP_A(sham) ≥ ΔP_A(sup) | Sham ≥ intervention — protocol failure |
| **K2** | orig_A < −20 nats OR orig_B < −20 nats | Baseline at floor — unmeasurable |
| **K3** | κ̃_K(sup) > 5.0 on ≥ 2/5 STEEP_LOCAL heads | Suppression failed (threshold is 5.0, not 1.0, to account for medium's higher absolute κ̃ scale) |

---

## 6. Analysis plan

1. Load GPT-2 medium (attn_implementation="eager")
2. Compute positional field + κ̃ verification for STEEP_LOCAL heads (before/after)
3. Build suppressed model and sham model (no structural head modification)
4. Score Task A + Task B across: original, suppressed, sham
5. Compute ΔP_A, ΔP_B (suppressed vs orig; sham vs orig)
6. Item-level counts
7. Evaluate P1, P2, P_null, P_degrade, K1–K3
8. Compare to exp-142 (GPT-2 small suppress-only: +0.33 nats, 18/20)
   and to exp-147 (GPT-2 medium combined corrected: −0.13 nats, 2/20)

---

## 7. Connection to the record

- **Follows from:** exp-142 (GPT-2 small suppress-only, CONFIRMED, +0.33 nats);
  exp-147 (GPT-2 medium combined corrected, INCONCLUSIVE, −0.13 nats)
- **Bears on:** P1 (functional role of conformal-window structural heads in positional
  retrieval — whether suppression arm contributes positively or negatively in medium)
- **Companion experiment:** exp-148 (amplify-only). Together exp-148/149 bracket the
  source of exp-147's inversion.
- **Pre-registration required before run.py:** yes — registered before any code,
  per room rule (checked by physics_coherence)
- **Analysis-only:** no (new forward passes required; GPT-2 medium locally cached)
