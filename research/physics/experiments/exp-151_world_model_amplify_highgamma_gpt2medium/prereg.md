# exp-151 — Pre-registration
## World-model battery: higher-γ amplification-only in GPT-2 medium

**Date:** 2026-09-22 (pre-registration; run.py does not exist yet)
**Registered by:** Ariel. Solo physics room session, ~12:25 AM MDT.
**Model:** openai-community/gpt2-medium

---

## 1. Background and motivation

exp-148 (2026-09-19) tested amplification-only at γ=+2.0 on the top-5 lowest-κ̃_K
random-token structural heads from exp-146 (L6H9, L5H14, L7H5, L9H7, L8H13) and
found **NULL**: ΔP_B = +0.01 nats (P_null fires). In GPT-2 small, the same protocol
at γ=+2.0 produced CONFIRMED improvement: ΔP_B = +0.27 nats, 20/20 items (exp-141).

Two hypotheses could explain the null:
- **(H_thresh)** The amplification signal exists in medium but is below detection at
  γ=+2.0. The structural heads in medium encode weaker or differently-distributed
  positional signal, requiring a larger gain to surface the effect.
- **(H_arch)** The structural heads in GPT-2 medium do not play the positional
  retrieval role that the analogous heads play in GPT-2 small. The architectural
  difference is real, not a calibration issue.

If H_thresh is true, testing at γ=+5 should produce positive ΔP_B.
If H_arch is true, γ=+5 will remain null or degrade Task B.

Amplification formula: κ̃_after ≈ (1+γ)² × κ̃_before.
- γ=+2: (1+2)² = 9×  amplification factor
- γ=+5: (1+5)² = 36× amplification factor

At γ=+5, the structural head L6H9 (κ̃=0.208) would reach κ̃_amp ≈ 7.5 — a
regime that is strongly amplified and unambiguous. If the positional retrieval
mechanism is there at all, it should be detectable at 36× gain.

The suppression arm has been tested (exp-149, exp-150) and consistently
degraded Task B, supporting the relay interpretation for the steep/local heads.
This experiment focuses on the amplification arm alone.

---

## 2. Hypothesis

**H:** The Δ-window structural heads in GPT-2 medium support positional retrieval
(Task B) but require a larger amplification gain to surface the effect than in
GPT-2 small. Amplifying the top-5 lowest-κ̃_K structural heads at γ=+5.0 (36×
amplification, vs 9× in exp-148) will produce measurable Task B improvement.

Target heads (from exp-146 random-token census):
```
STRUCTURAL = [(6, 9), (5, 14), (7, 5), (9, 7), (8, 13)]
# L6H9, L5H14, L7H5, L9H7, L8H13
# κ̃_K before: 0.208, 0.220, 0.257, 0.417, 0.469
# κ̃_K predicted after (γ=+5): ≈ 7.5, 7.9, 9.3, 15.0, 16.9
```

---

## 3. Predictions

| ID | Statement | Threshold |
|---|---|---|
| **P1** | ΔP_B > 0.10 nats (Task B improved vs original) | > 0.10 |
| **P2** | Task A approximately preserved: \|ΔP_A − ΔP_A_sham\| < 0.5 nats | < 0.5 |
| **P_null** | \|ΔP_B\| ≤ 0.05 nats | No detectable effect |
| **P_degrade** | ΔP_B < −0.10 nats | High-gain amplification damages Task B |
| **P_positive** | ΔP_B > 0 | Directional improvement even if below P1 threshold |
| **K3_check** | κ̃_after ≥ 30 × κ̃_before on ≥ 4/5 heads | Amplification achieved (≥30× regime) |

Primary interpretations:
- **P1 fires:** H_thresh confirmed — the signal exists in medium but required
  higher gain. The amplification arm contributes positively at γ=+5.
- **P_null fires:** H_arch supported — the architectural difference is real;
  the structural heads in medium are not encoders of the same positional retrieval
  circuit, regardless of γ level.
- **P_degrade fires:** High-gain amplification overloads or disrupts the
  retrieval mechanism — interesting constraint on what κ̃ amplification means
  functionally in medium.

---

## 4. Kill conditions

| ID | Condition | Interpretation if fired |
|---|---|---|
| **K1** | \|ΔP_B_sham\| ≥ \|ΔP_B_amp\| AND \|ΔP_A_sham\| ≥ \|ΔP_A_amp\| | Sham ≥ intervention — result is noise |
| **K2** | Median log P(correct) < −20 on Task A or Task B (original) | Task floor — uninformative |
| **K3** | κ̃_after < 20 × κ̃_before on ≥ 2/5 STRUCTURAL heads | Amplification failed — regime not achieved |

---

## 5. Protocol

- Model: openai-community/gpt2-medium (attn_implementation="eager"; MPS)
- Same 5 structural heads as exp-148: L6H9, L5H14, L7H5, L9H7, L8H13
- **γ_amp = +5.0** (change from exp-148's γ=+2.0; all other protocol identical)
- No suppression of steep/local heads (amplification arm only)
- Census RNG seed: 42 (same as prior experiments for reproducibility)
- Positional field δ: centered mean of ln_1(h) output over N_INPUTS=50 random-token
  sequences, SEQ_LEN=512, SEED=42
- P_k = top-4 PC directions of δ (SVD)
- Amplification: W_K_amp = W_K + γ × (P_k.T @ (P_k @ W_K))
- Sham: matched-norm perturbation in ⊥ complement of P_k
  SHAM_SEED_BASE = 2026092201 (distinct from all prior experiments)
- Three evaluations: original model, amplified model, sham model
- Same Task A (entity-state tracking, 20 items) and Task B (positional retrieval,
  20 items) as exp-140 through exp-150

---

## 6. Analysis plan

1. Load GPT-2 medium (attn_implementation="eager")
2. Compute positional field δ and κ̃_K verification for STRUCTURAL heads (before/after)
3. Build amplified model (γ=+5.0) and sham model
4. Score Task A + Task B: original, amplified, sham
5. Compute ΔP_A, ΔP_B (manipulation vs orig; sham vs orig)
6. Item-level counts
7. Evaluate P1, P2, P_null, P_degrade, K1–K3, K3_check
8. Compare to exp-141 (GPT-2 small amplify-only, γ=+2.0: +0.27 nats)
   and exp-148 (GPT-2 medium amplify-only, γ=+2.0: +0.01 nats, NULL)

---

## 7. Connection to the record

- **Follows from:** exp-148 (GPT-2 medium amplify-only, γ=+2.0, NULL);
  exp-150 (reduced-γ suppress-only, INCONCLUSIVE — relay account strengthened)
- **Bears on:** P1 (functional role of Δ-window structural heads in GPT-2medium)
- **If P_null fires again:** architecture account confirmed; amplification route
  closed for medium at this population; next step is causal ablation of the
  steep/local heads (relay confirmation), or functional characterization of
  medium's structural heads by a different method
- **Pre-registration required before run.py:** yes — this file committed and pushed
  before any code is written, per room rule (checked by physics_coherence)
- **Analysis-only:** no (new forward passes; GPT-2 medium locally cached)

---

*This pre-registration is committed and pushed before run.py is written.*
*The ordering is attested by the git commit timestamp.*
