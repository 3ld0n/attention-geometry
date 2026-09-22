# exp-152 — Notes
## Causal ablation of steep/local heads in GPT-2 medium (relay confirmation test)

**Date:** 2026-09-22  
**Session:** Physics room, ~8:17–9:00 AM MDT  
**Verdict: INCONCLUSIVE — suppression-ablation dissociation; relay account (simple form) not confirmed**

---

## Summary

This experiment tested whether zeroing W_K for the 5 steep/local heads (L4H13, L15H8, L8H7,
L5H11, L11H7) degrades Task B, which would confirm the relay account (these heads relay
positional information that downstream positional retrieval depends on).

**Result: Task B did NOT degrade.** ΔP_B = +0.0514 nats (10/20 items improved vs sham 13/20).
The relay account, stated as "positional concentration of these heads is load-bearing for
Task B," is not confirmed by this experiment.

---

## Suppression-ablation dissociation — the key finding

| Experiment | Protocol | ΔP_B (exact) | n_B_improved |
|---|---|---|---|
| exp-149 | suppress-only, γ=−1.0 (~600× reduction) | −0.10 | 2/20 |
| exp-150 | suppress-only, γ=−0.86 (~42× reduction) | −0.08 | 2/20 |
| **exp-152** | **W_K ablation (W_K = 0)** | **+0.0514** | **10/20** |

Suppression consistently degrades Task B across two magnitude levels (exp-149/150).
Full ablation does NOT degrade Task B. This dissociation is the experiment's main finding.

**What suppression does:** W_K_sup = W_K + γ × W_K_proj (γ < 0) — subtracts the positional
component from W_K, leaving a non-zero W_K that captures non-positional dimensions. κ̃ drops
to near-zero, but the head still attends using its residual (semantic/non-positional) W_K.

**What ablation does:** W_K_abl = 0 — all keys are zero; attention scores are zero for all
pairs; softmax gives uniform causal attention. Head output = mean(W_V @ x) over all positions
in the causal window. No routing at all.

**Mechanistic interpretation:** The degradation under suppression is NOT caused by removing
positional concentration. It is caused by the RESIDUAL W_K (what remains after the positional
component is subtracted) creating actively misleading attention patterns for Task B.

In other words: when you suppress the positional projection, the head now attends based on its
non-positional dimensions — which happen to route to wrong tokens for the positional retrieval
task. This is suppression-induced misrouting, not relay disruption.

When W_K is fully zeroed (ablation), no such misrouting occurs: the head is silent (uniform
averaging), and Task B is unaffected.

---

## What this tells us about the relay account

The relay account (as stated in the pre-registration): the steep/local heads relay positional
information through concentrated local attention, and Task B performance depends on this relay.

**Verdict: NOT CONFIRMED in this form.**

Zeroing the key routing does not harm Task B. The heads are not load-bearing via their
positional ATTENTION PATTERN. The concentrated local attention they normally exhibit is not
what Task B needs.

**What they might be doing instead:**

1. **Value pathway load-bearing (not tested):** Even with W_K=0, these heads write
   mean(W_V @ x) to the residual stream. If this mean-value contribution matters for Task B,
   we'd see Task B degrade when W_V is also ablated. This is not tested in exp-152.

2. **Genuine neutrality:** These heads may be genuinely neutral for Task B when left alone or
   when fully ablated. The suppression-caused degradation is purely an artifact of the
   suppression mechanism creating residual W_K that misroutes.

3. **The suppression effect is an active interference pattern:** The suppressed W_K
   concentrates attention on non-positional (semantic) features of the context, which for
   positional retrieval tasks (Task B) actively misleads. This would make the suppressed heads
   neither relay nor competitor in the normal sense — they are interference generators when
   partially modified.

---

## Task A observation

ΔP_A = −0.1577 nats (ablated vs original); ΔP_A_sham = −0.0148 nats.
The ablation mildly degraded Task A (entity-state tracking) while the sham barely touched it.
P2 fires: |ΔP_A − ΔP_A_sham| = 0.143 < 0.5, but the asymmetry (abl hurts A, sham doesn't)
suggests these heads DO contribute something to Task A through their W_K routing, even if
not to Task B. Removing their W_K slightly degrades entity-state tracking.

---

## Ablation regime confirmation

All 5 heads: κ̃_after_abl = 0.000000 exactly (W_K successfully zeroed).
K3 not fired.

κ̃_before and κ̃_sham for verification:

| Head | κ̃_before | κ̃_abl | κ̃_sham |
|---|---|---|---|
| L4H13 | 43.199 | 0.000 | 20.328 |
| L15H8 | 39.949 | 0.000 | 18.867 |
| L8H7 | 33.235 | 0.000 | 15.721 |
| L5H11 | 31.698 | 0.000 | 14.940 |
| L11H7 | 30.911 | 0.000 | 14.600 |

Sham κ̃ values (14–20) are well below original (31–43) because the sham perturbation is in
⊥(P_k) — it doesn't add to the positional component but has its own projection onto the
positional field through indirect geometry. The sham was not intended to exactly preserve κ̃;
it was intended to provide a matched-norm perturbation in the non-positional complement.

---

## Next experimental route

The suppression-ablation dissociation opens a new question: does the suppressed W_K actively
misroute, or does the residual W_K introduce random noise? One test:

**W_V ablation (exp-153 candidate):** Zero W_V for those 5 heads. This would:
- Test whether the VALUE PATHWAY contribution (not the key routing) is load-bearing for Task B.
- If Task B degrades: the heads matter for Task B via their value output (mean values written
  to residual stream — possibly providing context that downstream heads read).
- If Task B does not degrade: the heads are genuinely neutral for Task B; the suppression
  effect is purely a key-routing interference artifact.

This is the natural next step. Pre-register before any code.

---

## What this contributes to P1 medium replication

The antagonism mechanism (exp-141/142/143, GPT-2 small) is NOT confirmed in GPT-2 medium
through any tested route:
- Amplification (structural heads): NULL at γ=+2.0 (exp-148), γ=+5.0 (exp-151)
- Suppression (steep/local heads): INCONCLUSIVE/degradation (exp-149, exp-150)
- Ablation (steep/local heads, W_K): INCONCLUSIVE — no relay effect on Task B (exp-152)

The current picture: in GPT-2 medium, the mechanisms that produce the antagonism in small
(structural heads encoding positional retrieval, steep/local heads competing with them) are
NOT replicated by the same structural geometric signatures. Geometry identifies a category;
function is architecture-specific.

The P1 evidence bracket remains GPT-2 small only.

---

*exp-152: pre-reg 5a15b35, complete, inconclusive. Honest negative. Suppression-ablation dissociation is the key finding.*
