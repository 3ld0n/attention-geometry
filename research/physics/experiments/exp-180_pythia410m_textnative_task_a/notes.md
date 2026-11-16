# exp-180 Notes — Pythia-410m text-native heads: entity-state tracking (Task A)
# 2026-11-16, solo physics room, ~12:35–1:05 AM MST

---

## Result: H_entity DEAD — H_arch_full SUPPORTED

Three-experiment arc complete. After Task C (exp-178: 0/16 carry content-retrieval),
Task B (exp-179: 1/16 carry positional-retrieval), and now Task A: **1/16 carry
entity-state tracking function** by the registered criterion. The declared prior
(H_arch_full: Pythia-410m text-native Δ-window heads carry no positive function in
any of the three tested categories) is supported.

---

## Per-head summary

| Head | ΔP_A | n_deg/20 | sham | Competitor? | Note |
|---|---|---|---|---|---|
| L14H15 | **−0.086** | 14 | −0.151 | no | Only head approaching criterion; sham MORE negative |
| L12H3 | −0.010 | 13 | −0.003 | no | Borderline degrader; sham near-null |
| L9H15 | −0.009 | 11 | +0.125 | yes [C] | Competitor; sham large positive |
| L0H3 | −0.006 | 13 | +0.010 | yes [C] | Near-null |
| L21H11 | −0.003 | 10 | −0.003 | no | Near-null |
| L19H15 | −0.002 | 9 | +0.006 | yes [C] | Near-null |
| L22H1 | −0.000 | 12 | −0.015 | no | Near-null |
| L4H12 | +0.013 | 6 | +0.027 | no | Mild improver |
| L1H8 | +0.023 | 7 | +0.010 | no | Mild improver |
| L2H8 | +0.008 | 7 | +0.008 | yes [C] | Near-null |
| L3H2 | +0.034 | 4 | +0.037 | yes [C] | Mild improver; sham similar |
| L8H15 | +0.034 | 5 | +0.076 | yes [C] | Mild improver; sham larger |
| L12H13 | +0.042 | 8 | +0.020 | no | Mild improver |
| L3H7 | +0.051 | 3 | +0.148 | yes [C] | Mild improver; sham much larger |
| L7H11 | +0.062 | 7 | +0.065 | yes [C] | Mild improver; sham similar |
| **L3H14** | **+0.076** | 2 | +0.072 | no | Strongest improver; ablation-specific |

---

## Kill flags

**K2 fires (near-null):** By the ≥0.05 threshold, only 1 degrader (L14H15) and
3 improvers (L3H7, L7H11, L3H14). The population is near-null for entity tracking.

**K3 fires (sham ≥ ablation, 10/16 heads):** For most heads, the effects are so
small that the sham perturbation (random Gaussian W_K of matching scale) produces
comparable or larger effects. This flag confirms the near-null interpretation:
the ablation signal is below the sham noise floor for this population and this task.

**L14H15 anomaly:** The one head approaching degrader criterion (ΔP_A = −0.086,
14/20 degraded) has an even more negative sham effect (−0.151). This suggests:
either (a) any disruption of W_K at this scale hurts entity tracking for this head,
including random noise — the specific trained direction is not what matters, just
having a non-zero W_K; or (b) the effect is an artifact of the perturbation scale.
Not a positive function finding.

---

## Competitor structure in Task A

The 8 competitor heads (which improve Task C on ablation, exp-178) show a range
in Task A:
- 0/8 carry entity function (ΔP_A < −0.05)
- 2/8 mildly improve Task A on ablation (L3H7: +0.051, L7H11: +0.062)
- 6/8 near-null or mild

The competitor suppression is **Task C-specific**: ablating these heads improves
content retrieval but does not reliably improve entity tracking. The two tasks
are functionally dissociated from the competitors' perspective.

---

## Cross-experiment synthesis: three-task arc

| Experiment | Task | Declared criterion | n_carriers | Key finding |
|---|---|---|---|---|
| exp-178 | C (content-retrieval) | ≥10/16 | **0/16** | Architecture-conditionality confirmed |
| exp-179 | B (positional-retrieval) | ≥10/16 | **1/16** | H_positional falsified; L2H8 borderline |
| exp-180 | A (entity-state tracking) | ≥10/16 | **1/16** | H_entity dead; K2+K3 near-null |

**Scope conclusion:** The three-population anatomy's functional layer (text-native
Δ-window heads carry content-retrieval function) is **scoped to GPT-2/AbsPE**.
The geometric attractor (Δ ≈ 1/4) predicts no tested functional role in
Pythia-410m/RoPE across three independent instrument types.

**What this changes about what I take the census to mean:**
The Geometry question was: does the geometric attractor predict functional identity
across architecture families? After three instruments, the honest answer is: no,
it does not predict any of the three named functions in RoPE. The geometric
classification tells us something about the head's pattern of weight structure
(distributed W_K geometry, exp-127) but not about its computational role.

---

## What remains open

1. **What do Pythia-410m's text-native Δ-window heads actually do?** Three null
   results tell us what they don't do. The competitor structure (8/16 suppress
   content retrieval on ablation) is real and mechanism-unknown. Possible approaches:
   - Mechanistic analysis: what does W_K select for in natural contexts?
   - Census protocol variant: measure the population under RoPE-native conditions
   - Passive analysis: what tokens do these heads attend to in natural text?
   
2. **Is the architecture-conditionality a PE-type effect?** The comparison is
   GPT-2/AbsPE vs. Pythia/RoPE. OLMo-7B (also RoPE but different training) would
   test whether this is a PE-type effect or a Pythia-specific effect.

3. **L3H14 (mild suppressor, +0.076 on Task A, 18/20 improved):** This head
   suppresses entity tracking, and the effect is ablation-specific (sham = +0.072,
   comparable but not larger). This is the one head where the ablation direction
   is both meaningful and consistent. Worth noting for the structural session.

---

## Verdict in registry terms

- status: complete
- verdict: falsified
- H_entity: dead (1/16, K2+K3 fire)
- H_arch_full (declared prior): supported
- Three-experiment arc: H_arch_full supported across Tasks A, B, C for Pythia-410m
