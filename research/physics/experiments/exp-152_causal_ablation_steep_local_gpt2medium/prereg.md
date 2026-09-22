# exp-152 — Pre-registration
## Causal ablation of steep/local heads in GPT-2 medium (relay confirmation)

**Date:** 2026-09-22 (pre-registration; run.py does not exist yet)
**Registered by:** Ariel. Solo physics room session, ~8:17 AM MDT.
**Model:** openai-community/gpt2-medium

---

## 1. Background and motivation

exp-149 (suppress-only, γ=−1.0) and exp-150 (suppress-only, γ=−0.86) established that
suppressing the 5 steep/local heads in GPT-2 medium consistently degrades Task B:
- exp-149: ΔP_B = −0.10 nats, 2/20 items improved (vs sham 13/20)
- exp-150: ΔP_B = −0.08 nats, 2/20 items improved (both at different suppression magnitudes)

The over-suppression hypothesis (that γ=−1.0 was too aggressive) was tested in exp-150 and
weakened: the degradation magnitude is independent of the suppression level across the
[20×, 600×] relative-κ̃ reduction range. This implies these heads behave as **load-bearing
relays** for Task B: reducing their κ̃ (but keeping W_K non-zero) degrades positional retrieval
regardless of suppression depth.

The suppression paradigm (W_K_sup = W_K + γ × W_K_proj, with γ < 0) modifies W_K in the
positional projection direction while preserving the rest of the weight. A cleaner test of
the relay account is **full causal ablation**: zero W_K entirely for those heads. This:
1. Makes the head's key vectors identically zero for all inputs
2. Forces attention scores to zero for all (query, key) pairs
3. Makes softmax output a uniform distribution over the causal window
4. Removes the positional concentration (steep, local pattern) while preserving the value
   pathway (W_V still active; head still writes to residual stream via uniform averaging)

If these heads relay positional information through their concentrated attention pattern,
zeroing W_K destroys that relay mechanism. The relay account predicts Task B degrades.

**Amplification arm is closed (exp-151):** γ=+5.0 on the Δ-window structural heads returned
ΔP_B = +0.01 nats (identical to exp-148 at γ=+2.0). This is not the experiment's focus.
No structural head manipulation in exp-152.

---

## 2. Hypothesis

**H_relay:** The positional concentration pattern of the 5 steep/local heads is load-bearing
for Task B. Zeroing W_K (removing the concentration) will degrade Task B in GPT-2 medium.

Target heads (unchanged from exp-147/149/150):
```
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
# L4H13, L15H8, L8H7, L5H11, L11H7
# κ̃_K before (from exp-150 prereg): 43.2, 39.9, 33.3, 31.7, 30.9
```

After ablation (W_K = 0): κ̃_after = 0 for all 5 heads (keys are zero vectors;
the head cannot project any positional or semantic content into its attention scores).

No manipulation of Δ-window structural heads (amplification route is closed — exp-151).

---

## 3. Predictions

| ID | Statement | Threshold |
|---|---|---|
| **P1** | ΔP_B < −0.10 nats (Task B degrades vs original) | < −0.10 |
| **P2** | Task A approximately preserved: \|ΔP_A − ΔP_A_sham\| < 0.5 nats | < 0.5 |
| **P_relay** | P1 fires AND sham does not degrade Task B by same amount | relay confirmed: concentration pattern is load-bearing |
| **P_null** | \|ΔP_B\| ≤ 0.05 | No detectable effect of ablation |
| **P_improve** | ΔP_B > +0.10 | Ablation of steep/local heads improves Task B — competitor account |
| **K3_abl** | κ̃_after = 0 on all 5 heads | Ablation confirmed (W_K successfully zeroed) |

Primary interpretations:
- **P1 fires (Task B degrades):** relay account confirmed — the positional concentration
  of these heads is carrying information that Task B depends on. Ablating the key computation
  destroys the relay and degrades positional retrieval.
- **P_null:** relay account weakened — these heads do not contribute the positional routing
  needed for Task B, even though suppressing them degrades it (suppression vs. ablation
  distinction would need explanation — possibly the suppressed W_K still contributes noise).
- **P_improve:** competitor account supported in the ablation regime — inconsistent with
  exp-149/150 which showed consistent degradation under suppression; would require reconciliation.

---

## 4. Kill conditions

| ID | Condition | Interpretation if fired |
|---|---|---|
| **K1** | \|ΔP_B_sham\| ≥ \|ΔP_B_abl\| AND \|ΔP_A_sham\| ≥ \|ΔP_A_abl\| | Sham ≥ ablation in both tasks — result is noise or sham too disruptive |
| **K2** | Median log P(correct) < −20 on Task A or Task B (original model) | Task floor — uninformative |
| **K3** | κ̃_after > 0.01 on any of the 5 ablated heads | Ablation incomplete — W_K was not successfully zeroed |

---

## 5. Sham design

The sham tests whether **any matched-scale disruption** of those heads' W_K degrades Task B,
or whether the degradation is specific to zeroing (uniform attention). The sham applies a
perturbation of the same Frobenius norm as the original W_K, but in the ⊥ complement of
the positional field subspace:

```
delta_abl = -W_K          (ablation perturbation — zeroes the whole weight)
‖delta_abl‖_F = ‖W_K‖_F

For sham: build 4 random basis vectors in ⊥(P_k), orthogonalize;
project W_K onto that complement; scale to match ‖W_K‖_F;
add to W_K (not replacing).
W_K_sham = W_K + γ_sham × W_K_perp_proj
where γ_sham is chosen so ‖γ_sham × W_K_perp_proj‖_F = ‖W_K‖_F.
```

This gives:
- **Ablated head:** W_K = 0 → uniform attention (κ̃ = 0)
- **Sham head:** W_K = W_K + large perturbation in ⊥ complement → attention still non-uniform
  but the positional projection is preserved (sham perturbation is ⊥ to positional field);
  κ̃_sham ≈ κ̃_orig (positional component intact)

If ablated head degrades Task B but sham does not: the **zeroing of the positional concentration**
specifically is what matters — the relay is the positional routing, not any generic weighting
these heads apply.

SHAM_SEED_BASE: 2026092201 (same family as prior experiments; distinct from exp-151 since that
used the structural heads, while this uses the steep/local heads).

---

## 6. Protocol

- Model: openai-community/gpt2-medium (attn_implementation="eager"; MPS)
- Same 5 steep/local heads as exp-147/149/150: L4H13, L15H8, L8H7, L5H11, L11H7
- Census RNG seed: 42 (consistent with prior experiments)
- Positional field δ: centered mean of ln_1(h) output, N_INPUTS=50, SEQ_LEN=512, SEED=42
- P_k: top-4 PC directions of δ (SVD) — same construction as exp-148/149/150/151
- **Ablation:** W_K[layer, head] = 0 (zero matrix) for all 5 steep/local heads
- **Sham:** matched-norm perturbation in ⊥ complement of P_k (see §5)
- Three evaluations: original model, ablated model, sham model
- Same Task A (entity-state tracking, 20 items) and Task B (positional retrieval, 20 items)
  as exp-140 through exp-151

---

## 7. Analysis plan

1. Load GPT-2 medium (attn_implementation="eager")
2. For each of the 5 steep/local heads:
   a. Compute positional field δ at ln_1 output of that layer (same protocol as prior exps)
   b. Compute P_k (top-4 SVD directions of δ)
   c. Record κ̃_before (expected ≈ 43.2, 39.9, 33.3, 31.7, 30.9)
   d. Build ablated W_K = 0; verify κ̃_after = 0
   e. Build sham W_K (see §5); verify κ̃_sham ≈ κ̃_before (≥ 0.8 × original)
3. Score Task A and Task B for original, ablated, and sham models
4. Compute ΔP_A, ΔP_B (ablated vs orig; sham vs orig); item-level counts
5. Evaluate P1, P2, P_null, P_improve, P_relay, K1–K3
6. Compare to suppression results (exp-149: −0.10, exp-150: −0.08)

---

## 8. Connection to the record

- **Follows from:** exp-149 (suppress-only γ=−1.0, INCONCLUSIVE/relay pattern),
  exp-150 (suppress-only γ=−0.86, INCONCLUSIVE/relay pattern),
  exp-151 (amplify-only γ=+5.0, NULL — amplification route closed)
- **Bears on:** P1 (functional role of steep/local heads in GPT-2 medium positional retrieval)
- **Relay confirmed if:** P1 fires (ΔP_B < −0.10) and K1 does not fire
- **Relay weakened if:** P_null fires (ablation did not change Task B)
- **Novel result if:** P_improve fires (competitor account survives ablation — suppression
  and ablation would have opposite effects, requiring mechanistic explanation)
- **Pre-registration required before run.py:** yes — this file committed and pushed before
  any code is written, per room rule (checked by physics_coherence)
- **Analysis-only:** no (new forward passes; GPT-2 medium locally cached)

---

## 9. Prior suppression results for comparison

| Experiment | Protocol | γ | κ̃ reduction | ΔP_B | n_B_improved | Verdict |
|---|---|---|---|---|---|---|
| exp-149 | suppress-only | −1.0 | ~600× | −0.10 | 2/20 | INCONCLUSIVE |
| exp-150 | suppress-only | −0.86 | ~42× | −0.08 | 2/20 | INCONCLUSIVE |
| **exp-152** | **W_K ablation** | **n/a** | **∞ (W_K=0)** | **?** | **?** | **pending** |

Prediction: ablation produces ΔP_B < −0.10 (stronger degradation than suppression),
confirming the relay account and establishing that the positional concentration pattern
of these heads is the relay mechanism.

---

*This pre-registration is committed and pushed before run.py is written.*
*The ordering is attested by the git commit timestamp.*
