# exp-153 — Pre-registration
## W_V ablation of steep/local heads in GPT-2 medium (value-pathway test)

**Date:** 2026-09-22 (pre-registration; run.py does not exist yet)
**Registered by:** Ariel. Solo physics room session, ~12:17 PM MDT.
**Model:** openai-community/gpt2-medium

---

## 1. Background and motivation

The suppression-ablation dissociation established by exp-149/150/152 leaves a specific
mechanistic question open:

| Experiment | Protocol | ΔP_B | n_B_improved | Verdict |
|---|---|---|---|---|
| exp-149 | suppress-only γ=−1.0 (~600× κ̃ reduction) | −0.10 | 2/20 | INCONCLUSIVE |
| exp-150 | suppress-only γ=−0.86 (~42× κ̃ reduction) | −0.08 | 2/20 | INCONCLUSIVE |
| exp-152 | W_K ablation (W_K = 0, κ̃ = 0) | +0.051 | 10/20 | INCONCLUSIVE |

Suppression degrades Task B; W_K ablation does not. The mechanistic interpretation (from
exp-152): suppression leaves a residual non-positional W_K that actively misroutes attention
based on semantic dimensions that are wrong for Task B. Zeroing W_K entirely removes all
routing (uniform averager) — no misrouting, no contribution, Task B unchanged.

This interpretation implies the 5 steep/local heads are **genuinely neutral for Task B** when
inactive. But "inactive" in exp-152 still means the head WRITES its mean-value vector to the
residual stream (uniform attention × W_V × x = mean(W_V @ x), a non-zero contribution).
W_V was never tested.

**This experiment tests W_V = 0.** When W_V is zeroed:
- The head computes attention weights normally (steep/local positional concentration intact)
- The head writes `attention_weights @ W_V @ x = 0` to the residual stream
- The head contributes nothing to the residual stream at all (via W_O × 0 = 0)
- The positional routing pattern is preserved but carries no payload

Two outcomes:
1. **Task B degrades (ΔP_B < −0.10):** the value pathway is load-bearing for Task B; zeroing
   the write disrupts something Task B depends on.
2. **Task B unchanged (|ΔP_B| ≤ 0.10):** these heads are genuinely neutral for Task B. The
   suppression degradation (exp-149/150) was entirely a key-routing interference artifact —
   the residual W_K after suppression was actively misrouting, not the relay disruption. The
   heads write nothing Task B needs.

Outcome 2, if it obtains, closes the causal story: suppression degrades because misrouting
(residual W_K); ablation does not degrade because no misrouting (W_K=0, uniform averager);
W_V ablation does not degrade because the value payload carries nothing for Task B. The 5
steep/local heads are functionally irrelevant to Task B under any clean removal protocol —
they appear important only when their key weights are partially disrupted.

---

## 2. Hypothesis

**H_value:** The value pathway of the 5 steep/local heads is load-bearing for Task B. Zeroing
W_V (removing the value contribution while preserving the attention pattern) will degrade Task B.

**H_neutral:** These heads are genuinely neutral for Task B. Zeroing W_V, like zeroing W_K
(exp-152), will leave Task B unchanged. The suppression degradation is entirely attributable to
active key-routing misrouting by the residual non-positional W_K.

Target heads (unchanged from exp-147/149/150/152):
```
STEEP_LOCAL = [(4, 13), (15, 8), (8, 7), (5, 11), (11, 7)]
# L4H13, L15H8, L8H7, L5H11, L11H7
# κ̃_K before (from prior experiments): 43.2, 39.9, 33.3, 31.7, 30.9
```

In GPT-2 medium (HuggingFace), W_V lives in the c_attn weight matrix at the third block:
`model.transformer.h[layer].attn.c_attn.weight[:, 2*n_embd : 3*n_embd]`
Per-head (head_size = n_embd // n_head = 1024 // 16 = 64):
`W_V[head] = c_attn.weight[:, 2*n_embd + head*head_size : 2*n_embd + (head+1)*head_size]`

Ablation: set the head's W_V block to zero in-place.

---

## 3. Predictions

| ID | Statement | Threshold |
|---|---|---|
| **P1** | ΔP_B < −0.10 nats (Task B degrades vs original) | < −0.10 |
| **P2** | Task A approximately preserved: \|ΔP_A − ΔP_A_sham\| < 0.5 nats | < 0.5 |
| **P_null** | \|ΔP_B\| ≤ 0.10 | No detectable effect of V ablation |
| **P_improve** | ΔP_B > +0.10 | V ablation improves Task B |
| **K3_abl** | Ablation confirmed: head write is zero (verify via forward-pass value output) | ablation complete |

Primary interpretations:
- **P1 fires (H_value confirmed):** value pathway carries something Task B needs. These heads
  are not genuinely neutral — their write matters. Requires reconciling with exp-152 (W_K=0
  left Task B unchanged despite preserving a mean-value write).
- **P_null fires (H_neutral supported):** the heads are genuinely neutral for Task B. The
  suppression degradation story is fully explained by key-routing misrouting. Closes the causal
  arc across exp-149/150/152/153.
- **P_improve fires:** removing the value write helps Task B — these heads were actively
  interfering via their value pathway. This would be an entirely new finding.

---

## 4. Kill conditions

| ID | Condition | Interpretation if fired |
|---|---|---|
| **K1** | \|ΔP_B_sham\| ≥ \|ΔP_B_abl\| AND \|ΔP_A_sham\| ≥ \|ΔP_A_abl\| | Sham ≥ ablation in both tasks — result is noise or sham too disruptive |
| **K2** | Median log P(correct) < −20 on Task A or Task B (original model) | Task floor — uninformative |
| **K3** | Value write not zeroed on any ablated head (any forward-pass check fails) | Ablation incomplete |

---

## 5. Sham design

The sham tests whether **any matched-scale disruption of W_V** degrades Task B, or whether the
effect (if any) is specific to zeroing. The sham replaces W_V with a random matrix of the same
Frobenius norm:

```
W_V_orig: shape [n_embd, head_size] = [1024, 64]
‖W_V_orig‖_F = F_orig

For sham: sample R ~ N(0, 1) of shape [1024, 64]
W_V_sham = R × (F_orig / ‖R‖_F)   [rescaled to same norm]
```

This gives:
- **Ablated head:** W_V = 0 → head writes exactly zero to residual stream
- **Sham head:** W_V = random matrix, same Frobenius norm as original → head writes
  random-direction output at matched scale; positional pattern intact but payload scrambled

If ablated head changes Task B but sham does not: the effect is specific to complete removal,
not to scale disruption. If sham also changes Task B: any large disruption to the value
pathway has an effect (scale or direction is not the discriminator).

SHAM_SEED_BASE: 2026092253 (distinct from all prior experiments in this series).

---

## 6. Protocol

- Model: openai-community/gpt2-medium (attn_implementation="eager"; MPS)
- Same 5 steep/local heads as exp-147/149/150/152: L4H13, L15H8, L8H7, L5H11, L11H7
- Census RNG seed: 42 (consistent with prior experiments)
- **Ablation:** W_V[layer, head] = 0 (zero the c_attn V block in-place) for all 5 steep/local heads
- **Sham:** W_V[layer, head] = random matrix, Frobenius norm matched to original W_V
- Three evaluations: original model, V-ablated model, sham model
- Same Task A (entity-state tracking, 20 items) and Task B (positional retrieval, 20 items)
  as exp-140 through exp-152
- Verification: after V ablation, run a single forward pass and confirm the head's value output
  is zero for a test input (K3 check)

---

## 7. Analysis plan

1. Load GPT-2 medium (attn_implementation="eager")
2. For each of the 5 steep/local heads:
   a. Record κ̃_K (should be ≈ 43.2, 39.9, 33.3, 31.7, 30.9 — unchanged, W_K not modified)
   b. Record ‖W_V‖_F (the value norm being ablated)
   c. Zero the W_V block in c_attn.weight for that head
   d. Verify: run one forward pass, confirm head value output is zero (K3)
3. Build sham model: same random seed (SHAM_SEED_BASE); replace W_V with matched-norm random matrix for same 5 heads
4. Score Task A and Task B for original, V-ablated, and sham models
5. Compute ΔP_A, ΔP_B (ablated vs orig; sham vs orig); item-level counts
6. Evaluate P1, P2, P_null, P_improve, K1–K3
7. Compare to suppression results (exp-149: −0.10, exp-150: −0.08) and W_K ablation (exp-152: +0.051)

---

## 8. Connection to the record

- **Follows from:** exp-152 (W_K ablation INCONCLUSIVE — suppression-ablation dissociation;
  relay account not confirmed; mechanistic interpretation: residual W_K misrouting is the cause)
- **Bears on:** P1 (functional role of steep/local heads in GPT-2 medium positional retrieval)
- **Closes if P_null:** suppression degradation is entirely key-routing misrouting; W_V carries
  nothing for Task B; heads are genuinely neutral under clean removal
- **Opens if P1 fires:** value pathway is load-bearing; requires reconciliation with exp-152
- **Pre-registration required before run.py:** yes — this file committed and pushed before any
  code is written, per room rule (checked by physics_coherence)
- **Analysis-only:** no (new forward passes; GPT-2 medium locally cached)

---

## 9. Full suppression-ablation table for comparison

| Experiment | Protocol | Modification | ΔP_B | n_B_improved | Verdict |
|---|---|---|---|---|---|
| exp-149 | suppress-only γ=−1.0 | W_K partial (~600× κ̃ reduction) | −0.10 | 2/20 | INCONCLUSIVE |
| exp-150 | suppress-only γ=−0.86 | W_K partial (~42× κ̃ reduction) | −0.08 | 2/20 | INCONCLUSIVE |
| exp-152 | W_K ablation | W_K = 0 (uniform averager) | +0.051 | 10/20 | INCONCLUSIVE |
| **exp-153** | **W_V ablation** | **W_V = 0 (silent head)** | **?** | **?** | **pending** |

H_neutral predicts ΔP_B ≈ 0 for exp-153, completing a dissociation pattern:
- Partial W_K disruption → misrouting → Task B degrades
- Complete W_K removal → no routing → Task B unchanged
- Complete W_V removal → no write → Task B unchanged

H_value predicts ΔP_B < −0.10.

---

*This pre-registration is committed and pushed before run.py is written.*
*The ordering is attested by the git commit timestamp.*
