# exp-148 — Pre-registration: World-model battery amplify-only, GPT-2 medium

**Registered before run.py is written.**
**Ariel — September 19, 2026, ~4:25 AM MDT. Solo.**

---

## 1. Context and motivation

exp-147 (2026-09-19) ran the corrected combined manipulation in GPT-2 medium
(amplify random-token structural heads L6H9/L5H14/L7H5/L9H7/L8H13, γ=+2.0 +
suppress steep/local heads L4H13/L15H8/L8H7/L5H11/L11H7, γ=−1.0) and found
the direction **still inverted**: ΔP_B = −0.13 nats, 2/20 Task B improved.

In GPT-2 small:
- exp-141: amplify-only → ΔP_B = +0.27 nats (20/20 Task B improved)
- exp-142: suppress-only → ΔP_B = +0.33 nats (18/20 Task B improved, direction inverted
  from prediction, but positive — antagonism model confirmed)
- exp-143: combined → ΔP_B = +0.71 nats (20/20 Task B improved, additive)

In GPT-2 medium, both combined runs (exp-145 wrong targets, exp-147 corrected targets)
degraded Task B. The diagnostic question is: **which arm drives the degradation?**

Three mutually exclusive outcomes:
1. **Amplification alone degrades** — amplify-only gives ΔP_B < 0 in medium
2. **Suppression alone degrades** — suppress-only gives ΔP_B < 0 in medium
3. **Neither arm alone degrades** — the combined protocol has a conflict; single arms
   are positive individually but their combination is negative

This experiment tests outcome 1 by running amplification only.

A notable anomaly in exp-147: the sham improved 16/20 Task B items (ΔP_B_sham ≈ flat
median, but item-level positive). The combined manipulation reversed this to 2/20 —
the manipulation is actively overriding something the model uses for positional retrieval,
not merely adding noise.

---

## 2. Hypothesis

**Amplify-only manipulation** — amplify 5 random-token structural heads
(L6H9, L5H14, L7H5, L9H7, L8H13; γ=+2.0, W_K direction) in GPT-2 medium,
with NO suppression of steep/local heads — **improves Task B (positional retrieval)
relative to sham**, analogous to exp-141 in GPT-2 small.

The structural heads are the 5 lowest-κ̃_K random-token structural heads from exp-146
(R²≥0.90, Δ∈[0.20,0.30] under frozen random-token census, SEED=42).

---

## 3. Target heads and protocol

**Structural heads (amplification, γ = +2.0):**
```
STRUCTURAL = [(6, 9), (5, 14), (7, 5), (9, 7), (8, 13)]
# L6H9, L5H14, L7H5, L9H7, L8H13
# κ̃_K: 0.208, 0.220, 0.257, 0.417, 0.469 (exp-146)
```

**No suppression of steep/local heads.** This is the only change from the combined
protocol of exp-147.

**Protocol (identical to exp-141/143/147 amplification arm):**
- Load GPT-2 medium (openai-community/gpt2-medium, attn_implementation="eager")
- Positional field δ: centered mean of ln_1(h) output over N_INPUTS=50 random-token
  sequences, SEQ_LEN=512, SEED=42
- P_k = top-4 PC directions of δ (SVD)
- Amplification: W_K_amp = W_K + 2.0 × (P_k.T @ (P_k @ W_K))
- Sham: matched Frobenius norm in ⊥ complement of P_k (SHAM_SEED_BASE = 2026091921;
  distinct from all prior experiments: 2026091200, 2026091201, 2026091301, 2026091501,
  2026091510, 2026091701, 2026091710, 2026091901, 2026091910)
- Score Task A (20 items) and Task B (20 items): original, amplified, sham

Model dimensions: D_MODEL=1024, D_HEAD=64, N_LAYERS=24, N_HEADS=16.

---

## 4. Predictions

| Label | Criterion | Interpretation if fired |
|---|---|---|
| **P1** (improvement) | ΔP_B > 0.10 nats | Amplification arm improves Task B in GPT-2 medium (analogous to GPT-2 small exp-141: +0.27 nats) |
| **P2** (Task A spared) | \|ΔP_A − ΔP_A_sham\| < 0.5 nats | Task A not differentially affected |
| **P_null** | \|ΔP_B\| ≤ 0.05 nats | No detectable effect |
| **P_degrade** | ΔP_B < −0.10 nats | Amplification alone degrades Task B — this arm drives exp-147's inversion |
| **P_positive_only** | ΔP_B > 0 | Directional improvement, even if below P1 threshold |

Primary diagnostic interpretations:
- **P_degrade fires:** amplification arm is the culprit in exp-147's inversion.
  Suppression arm may still be positive (as in GPT-2 small); combined interaction
  negative because amplification dominates in medium.
- **P1 fires (improvement):** amplification arm is NOT the culprit. Suppression
  (exp-149) or their interaction drives the combined degradation.
- **P_null fires:** amplification arm is inert in medium — neither helping nor hurting;
  suppression arm (exp-149) or interaction is the driver.

---

## 5. Kill conditions

| Kill | Criterion | Interpretation |
|---|---|---|
| **K1** | ΔP_B(sham) ≥ ΔP_B(amp) AND ΔP_A(sham) ≥ ΔP_A(amp) | Sham ≥ intervention — protocol failure |
| **K2** | orig_A < −20 nats OR orig_B < −20 nats | Baseline at floor — unmeasurable |
| **K3** | κ̃_K(amp) < 0.5 on ≥ 2/5 STRUCTURAL heads after amplification | Amplification failed |

---

## 6. Analysis plan

1. Load GPT-2 medium (attn_implementation="eager")
2. Compute positional field + κ̃ verification for STRUCTURAL heads (before/after)
3. Build amplified model and sham model (no steep/local modification)
4. Score Task A + Task B across: original, amplified, sham
5. Compute ΔP_A, ΔP_B (combined vs orig; sham vs orig)
6. Item-level counts
7. Evaluate P1, P2, P_null, P_degrade, K1–K3
8. Compare to exp-141 (GPT-2 small amplify-only: +0.27 nats, 20/20)
   and to exp-147 (GPT-2 medium combined, corrected: −0.13 nats, 2/20)

---

## 7. Connection to the record

- **Follows from:** exp-141 (GPT-2 small amplify-only, CONFIRMED, +0.27 nats);
  exp-147 (GPT-2 medium combined corrected, INCONCLUSIVE, −0.13 nats);
  exp-146 (random-token census, structural head identification)
- **Bears on:** P1 (functional role of conformal-window structural heads in positional
  retrieval — whether amplification arm contributes positively or negatively in medium)
- **Companion experiment:** exp-149 (suppress-only, same diagnostic purpose for the
  suppression arm). Together exp-148/149 bracket the source of exp-147's inversion.
- **Pre-registration required before run.py:** yes — registered before any code,
  per room rule (checked by physics_coherence)
- **Analysis-only:** no (new forward passes required; GPT-2 medium locally cached)
