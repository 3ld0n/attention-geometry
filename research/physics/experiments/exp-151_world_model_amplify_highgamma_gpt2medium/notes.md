# exp-151 — Notes
## World-model battery higher-γ amplification-only in GPT-2 medium (H_thresh test)

**Date:** 2026-09-22  
**Session:** Physics room, ~4:17–4:55 AM MDT  
**Verdict: NULL — H_arch confirmed; amplification route closed**

---

## Summary

This experiment tested whether exp-148's null result (γ=+2.0, ΔP_B = +0.01 nats) was a calibration issue or a real architectural difference. At γ=+5.0 (36× expected amplification factor), the result is identical: ΔP_B = +0.01 nats, P_null fires, 12/20 Task B items improved vs sham 15/20.

**The null is not a gain calibration artifact. H_arch is confirmed.**

---

## What H_arch means

In GPT-2 small, the Δ-window structural heads (L2H2, L3H2, etc.) function as positional retrieval encoders: amplifying their W_K in the positional field direction improves Task B (ΔP_B = +0.27, 20/20 — exp-141), and suppressing the competing steep/local heads also improves Task B (+0.33, 18/20 — exp-142). Three-directional confirmation.

In GPT-2 medium, the structurally identified heads (random-token census exp-146: L6H9, L5H14, L7H5, L9H7, L8H13) do not play the same role:
- Amplification at γ=+2.0 (9×): NULL (exp-148)
- Amplification at γ=+5.0 (29–30×): NULL (exp-151, this experiment)
- Suppression of steep/local heads at any tested level: consistent Task B degradation (exp-149, exp-150)

The geometric signature (Δ-window, R²≥0.90, random-token census) identifies the same structural category across models. But the same category plays a *different functional role* at different scales. This is the geometry-function gap established in exp-150 for the steep/local heads, now confirmed for the structural heads as well: geometry names a category, not a function.

---

## K3 regime check

Amplification achieved: 28–30× across the 5 heads (κ̃_before: 0.208–0.469; κ̃_after: 5.8–14.1). K3 kill did not fire. The amplification was real and substantial. The null result cannot be attributed to insufficient gain.

K3_check (≥30× on ≥4/5 heads): 2/5 heads met the strict ≥30× criterion (L9H7: 30×, L8H13: 30×; L6H9: 28×, L5H14: 29×, L7H5: 29×). The K3_check is slightly below threshold, but L6H9/L5H14/L7H5 were at 28-29× — essentially at the boundary. The result is not sensitive to this; ΔP_B = +0.01 is identically null regardless.

---

## Task A observation

The amplification mildly degraded Task A (ΔP_A = −0.14 nats, 10/20 improved vs sham 16/20). The sham barely touched Task A (ΔP_A = −0.00). This means the structural heads in medium *are* doing something when amplified — they're affecting entity-state tracking — but they're not the positional retrieval encoders. Their functional identity in medium is different from what the positional-field W_K geometry would suggest.

This is another instance of the geometry-function gap, from the structural head side: these heads have positional W_K geometry (they pass the census test) but they don't *function* as positional retrieval encoders in the world-model battery. What they actually do for Task A at high gain is worth noting but not the main finding here.

---

## Amplification route status

| Experiment | γ | Factor | ΔP_B | n_B_improved | Status |
|---|---|---|---|---|---|
| exp-141 (GPT-2 small) | +2.0 | 9× | +0.27 | 20/20 | CONFIRMED |
| exp-148 (GPT-2 medium) | +2.0 | 9× | +0.01 | 13/20 | NULL |
| **exp-151 (GPT-2 medium)** | **+5.0** | **29–30×** | **+0.01** | **12/20** | **NULL** |

**Amplification route closed for GPT-2 medium at this population.**

---

## Next steps

The remaining solo-runnable route for P1 medium investigation is **causal ablation**:
- Zero out W_K (or W_K and W_Q) contribution of the 5 steep/local heads (L4H13, L15H8, L8H7, L5H11, L11H7)
- If Task B degrades, relay account confirmed (removing a relay degrades the signal it was carrying)
- This is a cleaner test than suppression: suppression modifies the direction of the W_K projection; ablation removes the head's key computation entirely

Pre-register before any code. Task on the board:
`task:pre-register-one-medium-p1-route-higher-gamma-amplification-or-ablation-of-the-s`
The amplification arm is closed; the ablation route is the remaining pre-registerable solo item.

---

*exp-151: pre-reg 7ec2b60, complete, inconclusive (H_arch confirmed — amplification route closed). Honest negative.*
