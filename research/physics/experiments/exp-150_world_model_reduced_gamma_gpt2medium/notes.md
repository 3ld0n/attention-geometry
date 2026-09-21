# exp-150 — Notes
## World-model battery: reduced-γ suppress-only in GPT-2 medium

**Date:** 2026-09-21 (run), 2026-09-20 (pre-registration, commit 3b44c1c)
**Verdict: INCONCLUSIVE — over-suppression hypothesis weakened**
**Pre-registration:** attention-geometry 3b44c1c (git-attested, pushed before run.py)

---

## Summary

The over-suppression hypothesis predicted: γ=−1.0 suppressed GPT-2 medium's 5 steep/local
heads by ~600×, crossing a threshold that scrambled the positional processing chain
(exp-149). Calibrating to ~50× relative reduction (matching exp-142's small-model regime)
should recover positive Task B signal.

**Result:** ΔP_B = −0.08 nats at γ=−0.86. Item-level: 2/20 Task B improved vs original
(identical to exp-149: 2/20 at γ=−1.0). The regime was correctly achieved (5/5 heads at
~42× relative κ̃ reduction, within the pre-registered [20×, 100×] window). **The
degradation is not a suppression-magnitude effect.**

---

## Regime verification

| Head  | κ̃ before | κ̃ after  | Ratio  | In [1/150, 1/10]? |
|-------|-----------|-----------|--------|-------------------|
| L4H13 | 43.20     | 1.040     | 0.0241 | ✓                |
| L15H8 | 39.95     | 0.939     | 0.0235 | ✓                |
| L8H7  | 33.24     | 0.772     | 0.0232 | ✓                |
| L5H11 | 31.70     | 0.768     | 0.0242 | ✓                |
| L11H7 | 30.91     | 0.705     | 0.0228 | ✓                |

Predicted: (1+γ)² = (0.14)² = 0.0196 (~51× reduction). Observed: ~42–44× (slightly less
aggressive than predicted — expected given SVD projection non-linearity). All 5 heads in
the pre-registered [20×, 100×] target window. P_check fires (5/5).

---

## Results

| Quantity | exp-142 small (γ=−1.0) | exp-149 medium (γ=−1.0) | exp-150 medium (γ=−0.86) |
|----------|------------------------|--------------------------|--------------------------|
| κ̃ reduction | ~50× | ~600× | ~42× |
| ΔP_B | +0.33 nats | −0.10 nats | −0.08 nats |
| Task B improved | 18/20 | 2/20 | 2/20 |
| ΔP_A | −0.01 nats | (Task A not measured) | −0.02 nats |
| Verdict | CONFIRMED | borderline P_degrade | INCONCLUSIVE |

Kill status:
- **K1: not fired** — suppression effect exceeds sham in both tasks (|ΔP_B_sup| = 0.08 > |ΔP_B_sham| = 0.02).
  The manipulation is doing something real; the degradation is not noise.
- **K2: not fired** — baseline log-probabilities in normal range (Task A median = −1.40, Task B = −4.98).
- **K3_low: not fired** — no head has ratio < 1/150 (all ~1/42).
- **K3_high: not fired** — no head has ratio > 1/10.

---

## Interpretation

**The over-suppression hypothesis is weakened, but not in the way I expected.**

The item-level Task B signal (2/20 improved) is identical between exp-149 (γ=−1.0, κ̃_after
~0.07) and exp-150 (γ=−0.86, κ̃_after ~0.77). The absolute κ̃ scale changed by roughly 10×;
the Task B outcome didn't change at all.

This is a clean negative for the "suppression-magnitude causes the inversion" account. The
degradation must come from something qualitative about removing any significant portion of
these heads' positional projection, not from the quantity removed.

**Two candidate explanations (untested):**

1. **Role difference — relay vs. competitor.** In GPT-2 small (12 layers, 5 structural heads),
   the 5 steep/local heads sit in a different structural position relative to the Δ-window heads.
   In exp-142, suppressing them freed the structural signal — they were competitors. In medium
   (24 layers, 24 structural heads), the same class of heads may relay positional information
   *to* the structural heads downstream, not compete with them. Near-total ablation of the
   positional subspace in these relay heads starves the mechanism.

2. **Population mismatch.** The 5 steep/local heads were selected in small because they had
   the highest κ̃_K and the suppression worked. In medium, the selection criterion (highest κ̃_K)
   may identify the load-bearing relays rather than the competitors. exp-146 found 24 structural
   heads in medium; the relevant intervention may be amplification of a different subset of
   those structural heads at higher γ than exp-148 tested.

**The sham pattern is interesting.** Task A sham improved 17/20 items by tiny amounts (ΔP_A_sham
= +0.01 median — noise level); the suppressed model degraded Task A (1/20 improved, ΔP_A = −0.02).
The suppression has a consistent, small negative effect on entity-state tracking even at γ=−0.86.
This is consistent with the heads being part of the tracking circuit, not just positional competitors.

---

## What's open after this experiment

1. **Identify the functional role of the 5 steep/local heads in GPT-2 medium.** Causal ablation
   (zero out the W_K W_Q contribution entirely) would distinguish "relay" from "competitor" more
   cleanly than κ̃ suppression. If ablation also degrades Task B, the relay hypothesis holds.

2. **Amplification-only in medium at higher γ.** exp-148 tested γ=+2.0 on 5 structural heads
   (Δ-window, random-token census) — NULL (ΔP_B = +0.01). But the target heads were mid-depth
   (L5–L9 structural heads). The 24 structural heads in medium from exp-146 include deeper
   heads (top-5 lowest κ̃_K: L6H9, L5H14, L7H5, L9H7, L8H13) — these may have different
   functional roles. A new amplification test on the deepest structural heads, or at higher
   γ, remains untested.

3. **Scale the evidence bracket.** P1 (κ-handle causally controls Task B performance) is confirmed
   in GPT-2 small (exp-141/142/143). Medium has now produced 5 experiments (exp-145, 147, 148,
   149, 150) — consistently inconclusive or degrading, with no positive Task B signal. The
   small-to-medium replication is not established.

---

## Artifacts

- `prereg.md` — pre-registration (3b44c1c, 2026-09-20)
- `run.py` — this experiment (γ=−0.86, steep/local heads, no amplification)
- `results.json` — full results (ΔP_B = −0.08, item-counts, kappa table)
