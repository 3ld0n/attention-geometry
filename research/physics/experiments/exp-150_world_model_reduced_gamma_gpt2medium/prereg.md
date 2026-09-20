# exp-150 — Pre-registration
## World-model battery: reduced-γ suppress-only in GPT-2 medium

**Date:** 2026-09-20 (pre-registration; run.py does not exist yet)
**Registered by:** Ariel. Solo session, ~3:20 AM MDT.
**Model:** openai-community/gpt2-medium

---

## Background and motivation

exp-149 (suppress-only, γ=−1.0, GPT-2 medium) found that suppression of 5 steep/local
heads degraded Task B (ΔP_B = −0.10 nats, item-level 2/20 improved vs sham 13/20).
This is the opposite of exp-142 (GPT-2 small, same protocol: ΔP_B = +0.33 nats, 18/20
improved).

The over-suppression hypothesis: γ=−1.0 reduces κ̃ by ~600× in medium (κ̃: 43 → 0.07)
vs ~50× in small (κ̃: 10 → 0.2). The suppression in medium crosses a different threshold —
near-total ablation of the positional projection scrambles the processing chain rather
than reducing competition. The hypothesis predicts that a γ calibrated for ~50×
relative reduction should recover the positive signal.

Target: κ̃_after/κ̃_before ≈ 1/50 for the steep/local heads.
Formula: W_K_sup = W_K + γ × W_K_proj, giving κ̃_after ≈ (1+γ)² × κ̃_before.
(1+γ)² = 1/50 → (1+γ) = 1/√50 ≈ 0.1414 → γ ≈ −0.859.
**Pre-registered γ: −0.86.**

Steep/local heads (unchanged from exp-147/149):
  L4H13, L15H8, L8H7, L5H11, L11H7
  κ̃ before: 43.2, 39.9, 33.3, 31.7, 30.9

No structural head amplification in this experiment (exp-148 showed amplification is
inert; exp-150 tests suppression alone).

---

## Hypothesis

**H:** The Task B degradation in exp-147/149 is caused by over-suppression at γ=−1.0.
Suppressing the same 5 steep/local heads at γ=−0.86 (targeting ~50× relative κ̃
reduction, matching exp-142's small-model regime) will produce positive ΔP_B in GPT-2
medium.

---

## Predictions

| ID | Statement | Threshold |
|---|---|---|
| P1 | ΔP_B > 0.10 nats (Task B improved vs original) | > 0.10 |
| P2 | Task A approximately preserved: \|ΔP_A − ΔP_A_sham\| < 0.5 nats | < 0.5 |
| P_check | κ̃_after/κ̃_before ∈ [1/100, 1/20] (20–100× reduction) on ≥ 4/5 heads | 4/5 in window |

P_check is the regime check: if the actual relative reduction is outside [20×, 100×] on
≥ 2/5 heads, the experiment did not achieve the targeted suppression level and the
result cannot cleanly test the over-suppression hypothesis (see K3).

---

## Kill conditions

| ID | Condition | Interpretation if fired |
|---|---|---|
| K1 | \|ΔP_B_sham\| ≥ \|ΔP_B_sup\| AND \|ΔP_A_sham\| ≥ \|ΔP_A_sup\| | Manipulation not responsible; result is noise |
| K2 | Median log P(correct) < −20 on Task A or Task B (original model) | Task floor issue; uninformative |
| K3_low | κ̃_after/κ̃_before < 1/150 on ≥ 2/5 heads (still over-suppressing like exp-149) | INCONCLUSIVE — γ_misfire; reduction too aggressive |
| K3_high | κ̃_after/κ̃_before > 1/10 on ≥ 2/5 heads (insufficient suppression) | INCONCLUSIVE — γ_misfire; reduction too weak |

If any kill fires: INCONCLUSIVE (kill fired).

---

## Verdict logic (applied after kill checks)

| Condition | Verdict |
|---|---|
| P1 and P2 | CONFIRMED — over-suppression hypothesis supported; γ calibration recovers positive signal |
| P1 fires, P2 fails | PARTIAL — directional improvement, Task A asymmetry |
| ΔP_B < −0.10 (P_degrade) | NULL — reduced suppression still degrades Task B; over-suppression hypothesis weakened |
| \|ΔP_B\| ≤ 0.05 (P_null) | NULL — suppression inert at γ=−0.86 |
| Otherwise | INCONCLUSIVE |

---

## Protocol

- Model: openai-community/gpt2-medium (attn_implementation="eager"; MPS)
- Same 5 steep/local heads as exp-149: L4H13, L15H8, L8H7, L5H11, L11H7
- Same task battery as exp-141 through exp-149 (Task A: 20 entity-state items; Task B: 20 positional-retrieval items)
- Census RNG seed: 42 (same as exp-149 for reproducibility of the positional field)
- γ_sup = −0.86 (fixed; not swept in this experiment)
- Sham: matched-norm perturbation in the ⊥ complement (same construction as exp-149)
- SHAM_SEED_BASE: 2026092001 (distinct from all prior experiments)
- Three evaluations: original model, suppressed model, sham model

---

## What a negative result teaches

If ΔP_B < 0 again at γ=−0.86 (with κ̃ reduction confirmed in [20×, 100×]):
the over-suppression hypothesis is weakened or falsified. The degradation may not be
about suppression magnitude — it may be architectural (medium's steep/local heads
serve a genuinely different functional role than small's, not just a scale difference).
This would suggest a different experimental strategy: (a) identify the functional role
of the 5 steep/local heads in medium (causal ablation, not κ̃ modification), or
(b) select different target heads in medium using the random-token census results
from exp-146.

---

*This pre-registration is committed and pushed before run.py is written.*
*The ordering is attested by the git commit timestamp.*
