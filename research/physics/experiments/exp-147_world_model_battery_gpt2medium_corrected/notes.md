# exp-147 — Notes: World-model battery GPT-2 medium, corrected targets

**Date:** September 19, 2026, ~12:40–1:40 AM MDT
**Ariel, solo physics session**
**Pre-registration:** attention-geometry de254d0 (git-attested, pushed before run.py)

---

## Result summary

**Overall verdict: INCONCLUSIVE — direction inverted**

The combined manipulation (amplify random-token structural heads + suppress steep/local heads)
degraded Task B in GPT-2 medium: ΔP_B = −0.13 nats, 2/20 items improved (vs 16/20 for sham).
Task A also degraded: ΔP_A = −0.17 nats, 2/20 improved.

| Quantity | exp-143 (GPT-2 small) | exp-145 (GPT-2 medium, wrong) | exp-147 (GPT-2 medium, correct) |
|---|---|---|---|
| ΔP_B | +0.71 nats ✓ | −0.25 nats ✗ | −0.13 nats ✗ |
| Task B items improved | 20/20 | 0/20 | 2/20 |
| ΔP_A | −0.02 nats (noise) | +0.08 nats (noise) | −0.17 nats |
| Task A items improved | 10/20 | 9/20 | 2/20 |
| Overall | CONFIRMED (additive) | INCONCLUSIVE | INCONCLUSIVE |

---

## What the manipulations did

Amplification worked correctly: all 5 structural heads amplified (κ̃_K rose 5–8× from baseline):
- L6H9: 0.208 → 1.554
- L5H14: 0.224 → 1.760
- L7H5: 0.261 → 2.023
- L9H7: 0.418 → 3.362
- L8H13: 0.469 → 3.834

Suppression worked correctly: all 5 steep/local heads suppressed to < 0.1 (from 30–43):
- L4H13: 43.2 → 0.076
- L15H8: 39.9 → 0.090
- L8H7: 33.3 → 0.049
- L5H11: 31.7 → 0.085
- L11H7: 30.9 → 0.057

Both manipulations executed correctly. K3_amp and K3_sup did not fire.

---

## What this constrains

**The population correction partially helped but did not fix the direction:**

exp-145 (WikiText-native targets) had ΔP_B = −0.25 nats.
exp-147 (random-token structural targets) had ΔP_B = −0.13 nats.
The correction moved the effect toward zero — smaller degradation — but the direction
stayed inverted. The root cause of the exp-145 failure is not *only* the wrong population.

**The antagonism model's evidence bracket is now confirmed as GPT-2 small only:**

Three experiments in GPT-2 medium (exp-145, exp-145's population correction in this session)
show the combined manipulation degrades Task B. Whatever the GPT-2 small mechanism is that
produces +0.71 nats improvement does not carry to GPT-2 medium under this protocol.

---

## Why might GPT-2 medium behave differently?

Several candidates, not yet tested:

1. **Higher absolute κ̃_K scale.** GPT-2 medium's structural heads have κ̃_K ≈ 0.2–0.5,
   and the steep/local heads have κ̃_K ≈ 30–43. In GPT-2 small, structural heads had
   κ̃_K similar, but steep/local heads had κ̃_K ≈ 6–22. The absolute scale of the
   positional mechanisms is much larger in GPT-2 medium, and γ=±2/−1 may be hitting
   different regimes. Specifically: suppressing a head from κ̃=43 to 0.05 may be *over-
   suppressing* relative to suppressing from κ̃=10 to 0.2 in GPT-2 small.

2. **More structural heads (24 vs 5) and redundancy.** In GPT-2 small, modifying 5 of 5
   structural heads is the entire population. In GPT-2 medium, modifying 5 of 24 (21%)
   leaves 19 structural heads untouched — potential redundancy. The 5 lowest-κ̃_K heads
   may not be the ones that dominate Task B performance.

3. **Different layer distribution.** GPT-2 medium's structural heads are mid-layer
   (L5–L17), while GPT-2 small's are deep (L2–L10 in the original training). The functional
   role of mid-layer vs deep positional heads may differ.

4. **Task B difficulty difference.** The sham improved 16/20 Task B items (ΔP_B ≈ +0.02
   nats), suggesting the items are near a distributional regime where small perturbations
   naturally improve some items. The combined manipulation actively reverting items that
   the sham improves indicates the manipulation is working against the retrieval mechanism.

5. **The 5-head selection criterion.** Lowest κ̃_K among structural heads = most purely
   positional (least coupled to semantic content). But Task B (positional retrieval in
   a list-recall task) may require a *different* functional subset of structural heads —
   perhaps ones that actively use positional structure in the semantic context, not the
   purest positional ones.

---

## Interesting anomaly: sham 16/20 on Task B

The sham improved 16/20 Task B items (median +0.02 nats, near flat). This means the
sham's matched-Frobenius perturbation in the orthogonal complement accidentally improved
most Task B items by a small amount. This is consistent with random perturbation in the
orthogonal complement occasionally nudging the model toward better list-position tracking
in this particular task design. It's not evidence of a real effect — the median is flat —
but it's worth noting as a design property.

The combined manipulation reversed almost all of those improvements (2/20). This suggests
the manipulation is actively overriding something the model uses for Task B, not merely
adding noise.

---

## What to try next

Before running more GPT-2 medium experiments, diagnose which manipulation drives the
degradation:

1. **Amplify-only in GPT-2 medium:** do the random-token structural heads (L6H9 etc.)
   alone produce the same degradation, or does the combined manipulation require the
   suppression?

2. **Suppress-only in GPT-2 medium:** do the steep/local heads alone produce the
   degradation? In exp-142 (GPT-2 small, suppress-only), ΔP_B = +0.33 nats.

3. **Smaller γ values:** the effect in GPT-2 small was ΔP_B = +0.27 nats at γ=+2.0
   for amplify-only. Maybe γ=1.0 or 0.5 in GPT-2 medium would be in a more tractable
   regime given the different absolute κ̃_K scale.

These would require pre-registration before running. The result here constrains the
prior: the antagonism model does not simply replicate in GPT-2 medium.

---

## Connection to the record

- **exp-143** (GPT-2 small CONFIRMED): established the antagonism model
- **exp-145** (GPT-2 medium, wrong targets): INCONCLUSIVE (direction inverted, population error)
- **exp-147** (GPT-2 medium, correct targets): INCONCLUSIVE (direction inverted, smaller magnitude)
- **P1 evidence bracket:** remains GPT-2 small only
- **Queue update needed:** add diagnostic experiments (amplify-only, suppress-only) as
  candidates; note that generalization to GPT-2 medium is an open question

### §4.3 paper note

The §4.3 paper correction (binary conditional → graded monotonic for the formation gate)
remains open (Eldon-present). exp-147 does not affect that.
