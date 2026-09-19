# exp-148 — Notes: World-model battery amplify-only, GPT-2 medium

**Date:** 2026-09-19, ~4:30–4:55 AM MDT
**Pre-registration:** f3ccd63 (pushed before run.py written)
**Verdict: NULL — amplification arm is inert in GPT-2 medium**

---

## Summary

ΔP_B = +0.01 nats (P_null fires). The amplification of 5 random-token structural heads
(L6H9, L5H14, L7H5, L9H7, L8H13; γ=+2.0) has essentially no effect on Task B
(positional retrieval) in GPT-2 medium. This is the opposite of what was expected from
exp-141 (GPT-2 small amplify-only: +0.27 nats, 20/20 Task B improved).

**Diagnostic answer:** Amplification is NOT the driver of exp-147's Task B degradation.

---

## Results table

| Quantity | exp-141 (GPT-2 small, amp-only) | exp-148 (GPT-2 medium, amp-only) |
|---|---|---|
| ΔP_B | +0.27 nats | **+0.01 nats (NULL)** |
| Task B improved | 20/20 | 13/20 (vs sham 6/20) |
| ΔP_A | ~0 nats | **−0.13 nats** |
| Task A improved (amp) | — | 10/20 |
| Task A improved (sham) | — | 3/20 |
| K1/K2/K3 | clear | clear |

---

## The two signals

**Task B (primary outcome):** Median flat. Item-level shows 13/20 improved vs sham's 6/20 —
there is some positive item-level signal, but the magnitude is too small to move the median
appreciably (+0.01 nats). Amplification is effectively inert on positional retrieval
in GPT-2 medium.

**Task A (unexpected finding):** Amplification slightly *degrades* Task A: ΔP_A = −0.13 nats
(10/20 items degraded in absolute terms, but 10/20 improved vs orig → net mixed; the median
summary: orig=−1.40, amp=−1.53, sham=−1.41; the sham has negligible Task A effect).
This is a new signal: in GPT-2 small, amplification left Task A within sham range (P2 confirmed
in exp-141). In medium, amplification slightly damages entity-state tracking without helping
positional retrieval. The 5 structural heads in medium may serve a different dual role than
their small counterparts.

---

## Why amplification is inert in medium (candidates)

1. **Depth mismatch.** The 5 targeted structural heads in GPT-2 medium are in mid-depth layers
   (L5–L9), while GPT-2 small's structural heads were in deep layers (L1–L2 for exp-141's
   selection). The functional role of these layers may differ — mid-depth heads may not be the
   "readers" of positional information for the output, even if they are genuinely conformal.

2. **Redundancy.** 5/24 structural heads = 21% coverage. In GPT-2 small, 5/5 = 100% of the
   random-token structural population was modified. The deep structural heads in medium
   (L19–L23, WikiText-native only) are untouched — they may be the actual functional carriers
   of positional retrieval for Task B output, and mid-depth structural amplification doesn't
   reach them.

3. **κ̃ scale difference.** GPT-2 medium structural heads start with lower κ̃ (0.21–0.47) than
   steep/local heads (30–43). The amplification doubles/triples κ̃ to 1.5–3.8. This is a large
   *relative* increase, but the absolute κ̃ post-amplification is still far below the steep/local
   heads' baseline. The positional field being amplified may not be influencing the token generation
   at the critical last-token positions for Task B.

---

## Connection to exp-147 and exp-149

- exp-148 (amplify-only): ΔP_B = +0.01. Amplification inert. Not the culprit.
- exp-149 (suppress-only): ΔP_B = −0.10 nats (borderline, item-level 2/20 vs sham 13/20).
  Suppression is the primary driver of the combined degradation.
- exp-147 (combined): ΔP_B = −0.13 nats. ≈ exp-148 (+0.01) + exp-149 (−0.10) ≈ −0.09, close
  to −0.13 (near-additive; the amplification also slightly degraded Task A, which may interact).

**Conclusion:** The antagonism model inversion in GPT-2 medium is driven primarily by the
suppression arm. The amplification arm is inert (not helpful, not harmful on Task B).

---

## Next step

exp-150 (candidate): Reduced-γ suppression in GPT-2 medium. The over-suppression hypothesis
predicts that γ = −0.9 to −0.86 (rather than −1.0) should reduce κ̃ by ~50× (analogous to
exp-142's small-model result) rather than 600×, and this relative level of suppression
should reproduce the positive Task B signal. Register before running.
