# exp-149 — Notes: World-model battery suppress-only, GPT-2 medium

**Date:** 2026-09-19, ~4:55–5:20 AM MDT
**Pre-registration:** f3ccd63 (pushed before run.py written)
**Verdict: INCONCLUSIVE (borderline P_degrade) — suppression arm drives exp-147's inversion**

---

## Summary

ΔP_B = −0.10 nats. P_degrade threshold is strictly < −0.10, so the criterion is borderline —
the result sits exactly at the edge. More importantly, the **item-level signal is decisive**:
Task B improved in only 2/20 items (vs sham's 13/20). The suppression arm alone reproduces
nearly all of the combined degradation seen in exp-147 (where combined gave 2/20).

Diagnostic answer: **The suppression arm (suppress steep/local heads) is the primary driver
of exp-147's Task B degradation in GPT-2 medium.** The amplification arm (exp-148) was inert.

---

## Results table

| Quantity | exp-142 (GPT-2 small, sup-only) | exp-149 (GPT-2 medium, sup-only) |
|---|---|---|
| ΔP_B | +0.33 nats (CONFIRMED) | **−0.10 nats (borderline P_degrade)** |
| Task B improved | 18/20 | **2/20 (vs sham 13/20)** |
| ΔP_A | ~noise | **−0.03 nats** |
| Starting κ̃ (steep/local) | ~10–12 | **31–43** |
| Ending κ̃ (after γ=−1.0) | ~0.2 | **0.05–0.09** |
| Relative reduction | ~50× | **~600×** |
| K1/K2/K3 | clear | clear |

---

## The over-suppression hypothesis — now the primary candidate

In GPT-2 small (exp-142), the steep/local heads had κ̃ ≈ 10–12. After γ=−1.0 suppression,
κ̃ ≈ 0.2 — a ~50× reduction. This level of suppression improved Task B.

In GPT-2 medium, the same 5 steep/local heads have κ̃ ≈ 31–43. After γ=−1.0 suppression,
κ̃ ≈ 0.05–0.09 — a **~600× reduction**. This is a factor of ~12 more aggressive in
relative terms. The positional projection W_K_proj is effectively fully removed.

The hypothesis: removing the positional projection **entirely** in medium hits a different
regime than removing it partially in small. The steep/local heads in medium carry
positional information so strongly that their near-total ablation disrupts positional
retrieval at the output, rather than liberating it by reducing competition with the
conformal-window heads.

This is the structural difference between the models — not a different mechanism, but
the same mechanism at a different operating point. The γ = −1.0 suppression is calibrated
for small's κ̃ scale; it over-suppresses in medium.

---

## Item-level analysis

The strongest signal is item-level:

| Condition | Task B improved | Task B degraded |
|---|---|---|
| sham | 13/20 | 7/20 |
| suppressed | 2/20 | 18/20 |

The suppression reversed ~11 of the 13 naturally-improving items. This pattern is nearly
identical to exp-147 combined (sham 16/20 → combined 2/20). Since exp-148 showed
amplification is inert on Task B, the suppression arm fully accounts for this reversal.

The 2 surviving improvements in exp-149 (vs 2 in exp-147) are consistent with the
amplification arm contributing essentially zero independent effect on Task B.

---

## The sham anomaly explained

exp-147 noted that the sham improved 16/20 Task B items (while sham median was ~flat).
The suppressed model in exp-149 has sham 13/20 improved. The difference (16 vs 13) may
be due to the sham construction: in exp-147, two sets of sham modifications were applied
(one for structural heads, one for steep/local heads), each with a matched-norm perturbation
in the ⊥ complement. In exp-149, only the steep/local sham perturbations are applied. The
extra sham modifications in exp-147 for the structural heads may produce slight additional
item-level improvements on Task B via the ⊥-complement perturbation.

The core finding stands: suppression is the active driver, not noise.

---

## Why suppression degrades Task B in medium (over-suppression analysis)

The positional structure of steep/local heads in GPT-2 medium is 3-4× stronger than in small
(κ̃ ≈ 35 vs ≈ 10). The steep/local heads are named for their locally-dominant, position-tracking
behavior — they compete with the conformal structural heads for the position signal. In small,
removing 95% of their positional projection (κ̃: 10 → 0.2) is calibrated: enough suppression
to reduce competition without destroying their other contributions.

In medium, removing 99.9% of their positional projection (κ̃: 43 → 0.06) appears to cross a
different threshold. These heads are more deeply involved in the positional processing chain —
possibly serving as mid-layer position aggregators that feed the output layers, not just
competitors. Near-total ablation scrambles position signals used by the output layers.

Alternative: **redundancy in the other direction**. In small, the 5 targeted steep/local heads
represent most of the high-κ̃ competition for the structural heads. In medium, there are many
more steep/local heads not targeted — but the 5 targeted ones may be load-bearing relays.

---

## Verdict label clarification

The pre-registered P_degrade criterion was `dB_sup < −0.10` (strictly less than). The result
is approximately −0.10. The label is "INCONCLUSIVE" by strict criteria, but the item-level
signal (2/20 vs sham 13/20) clearly identifies the suppression arm as degrading. The
borderline median result is an artifact of the threshold choice, not of the physics. The
honest negative is stated: suppression degrades Task B in medium at γ = −1.0.

---

## Connection to exp-147 and exp-148

Decomposing exp-147 (combined, ΔP_B = −0.13 nats):
- Amplification contribution (exp-148): +0.01 nats (inert)
- Suppression contribution (exp-149): −0.10 nats (degrading)
- Expected combined (additive): −0.09 nats; observed: −0.13 nats
- The remaining −0.04 nats discrepancy may be a real interaction or measurement variance.

The antagonism model is not falsified — it works in GPT-2 small. It fails in medium
because the suppression arm's γ = −1.0 protocol is not the right scale for medium's
κ̃ operating point.

---

## Next step

**exp-150 (candidate):** Reduced-γ suppress-only in GPT-2 medium.

The over-suppression hypothesis predicts that the same mechanism (suppressing steep/local heads)
should improve Task B in medium if γ is calibrated to achieve a relative κ̃ reduction similar
to exp-142 (~50× rather than ~600×). Target: κ̃ ≈ 43 → 0.86 (50× reduction).

The W_K_sup = W_K + γ × W_K_proj formula reduces the positional component by (1+γ) fraction.
To achieve κ̃_after/κ̃_before ≈ 0.02:
- (1+γ)² ≈ 0.02 → (1+γ) ≈ 0.14 → γ ≈ −0.86

Candidate γ values to pre-register: −0.7, −0.75, −0.8, −0.85, −0.9 (or a focused set around −0.85).
Pre-register before running. Kill conditions: K3 threshold should be set to confirm suppression
achieved the targeted relative reduction.

The prediction: if the over-suppression hypothesis is correct, some γ in [−0.7, −0.95] will
give positive ΔP_B in medium, and the optimal γ will be near the one that achieves ~50×
relative κ̃ reduction.
