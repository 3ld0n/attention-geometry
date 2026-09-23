# exp-156 — Structural (random-native) heads: W_K ablation content-retrieval test

**Pre-registration for exp-156.**
**This document is committed to attention-geometry BEFORE run.py is written.**

**Ariel — 2026-09-23, ~12:20 PM MDT. Solo physics room.**
**Seeded by exp-155 double dissociation (2026-09-23): text-native heads → content retrieval confirmed.**

---

## What exp-155 established

exp-155 confirmed the functional double dissociation in GPT-2 small:

| Population | Census | Functional role |
|---|---|---|
| Structural (5 random-native) | Δ-window under random tokens | Positional retrieval [exp-141/142/143] |
| Steep/local (5 heads) | High κ̃_K, steep decay | Anti-positional competition |
| Text-native (16 heads) | Δ-window under WikiText only | Content retrieval (−0.855 nats, 20/20) [exp-155] |

The census boundary (random-token vs WikiText-selective) tracks a functional boundary. exp-155 also found
that text-native ablation *improved* Task B (+0.189 nats, 16/20 improved), suggesting the two long-range
populations compete for signal in the positional retrieval task.

**Open question (seeded by exp-155):** Do structural heads participate in content retrieval (Task C)? The
symmetrical test is W_K ablation of the 5 structural heads, measuring both Task C and Task B.

---

## Hypothesis

**H_positional_selective:** The structural (random-native) heads encode positional routing in W_K (established
by exp-138: census slope = absolute-key-position drift, not relative lag). Their W_K does not carry
semantic content-matching capacity. Ablating W_K for these heads should:

1. Impair positional retrieval (Task B) — their established function.
2. Spare or release content retrieval (Task C) — with which they may compete via long-range signal
   occupancy.

This would confirm the dissociation is symmetric: text-native ablation harms content retrieval (exp-155
P1 confirmed), and structural ablation harms positional retrieval (this experiment P1).

---

## Population

**Structural (random-native) Δ-window heads in GPT-2 small:**
```
L2H1, L3H4, L5H0, L7H11, L10H8
```
Selected by random-token census (exp-007, exp-113); Δ_A ∈ [0.20, 0.30], R² ≥ 0.90 under frozen random
token inputs. These heads' long-range attention persists under random tokens — positionally driven, not
content driven.

---

## Task designs

### Task C — Content-specified long-range retrieval (primary)

The 20 items committed with exp-155 pre-registration (`task_c_items.json` in exp-155 folder). Same items,
same protocol: setup sentence introducing entity + property, ~80–150 token filler not mentioning the entity,
cue sentence ending before the target property token. Primary metric: log P(target | context).

### Task B — Positional retrieval (control; unchanged from exp-141/143/155)

Same approved battery from exp-143: 20 list-lookup tasks with ordinal position cues.

---

## Manipulations

### Ablation (W_K = 0)

Set W_K to zero for all 5 structural heads:
```
L2H1, L3H4, L5H0, L7H11, L10H8
```
With W_K = 0, attention scores are identically zero for all key positions; the softmax produces a uniform
distribution over causal positions. The head writes mean(W_V @ x) to the residual stream — no longer routing
by content or position.

### Sham

Random W_K of the same Frobenius norm as the original W_K for each head independently. Matches the scale
of disruption while destroying structure. Same protocol as exp-152/153/155.

---

## Metrics

For each item (Task B and Task C):
- `logp_orig`: log P(target | context, original model)
- `logp_abl`: log P(target | context, ablated model)
- `logp_sham`: log P(target | context, sham model)

Primary analysis:
- ΔP_C = median(logp_abl_C − logp_orig_C)
- ΔP_B = median(logp_abl_B − logp_orig_B)
- Sham controls: ΔP_C_sham, ΔP_B_sham
- Item-level: n_C_improved, n_B_improved

---

## Registered predictions and kill conditions

### P1 — Positional retrieval degrades with ablation

**Prediction:** ΔP_B < −0.10 nats.

**Justification:** exp-141 showed amplifying structural head κ̃ improved Task B (+0.27 nats, 20/20);
structural heads are causally involved in positional retrieval. W_K ablation removes the positional
routing mechanism, and positional retrieval should degrade.

### P2 — Content retrieval is spared or released

**Prediction:** |ΔP_C| < |ΔP_B| in absolute value — specifically ΔP_C > −0.10 nats (content retrieval
not substantially impaired).

**Justification:** Structural heads' W_K encodes absolute key position drift (exp-138), not semantic
content matching. If the census → function boundary is clean, removing their W_K should not impair
content retrieval. Competition-release is also possible: structural heads occupy long-range signal
capacity, and their ablation may free text-native heads to operate more effectively (analogous to how
text-native ablation released Task B: +0.189 nats in exp-155).

---

### Kill conditions

**K1 (structural heads contribute to content retrieval):** ΔP_C < −0.10 nats.

If K1 fires, the functional boundary between structural and text-native heads is not clean at the
positional/content axis. The double dissociation from exp-155 would be asymmetric: text-native heads
are selective (content only), but structural heads are general (both). Requires reassessment of the
three-population anatomy.

**K2 (structural heads W_K-inert for both tasks):** |ΔP_B| < 0.10 AND |ΔP_C| < 0.10.

If K2 fires, W_K ablation of the structural heads leaves both tasks unaffected. This would be a
suppression-ablation dissociation analogous to exp-152 in GPT-2 medium: the gain-of-function result
(exp-141 amplification) does not imply that ablation should produce loss-of-function. The heads may
contribute to Task B positional retrieval through their attention *distribution* (shaped by W_K), but
that distribution may be robust to moderate disruption. Honest negative — would extend the
geometry-function gap to the structural population.

**K3 (ablation failure):** ‖W_K‖_F > 0.01 on ≥ 3/5 heads after zeroing.

---

## Interpretation table

| P1 fires | K1 fires | K2 fires | Interpretation |
|---|---|---|---|
| Yes | No | No | H_positional_selective confirmed: structural → positional (W_K-dependent), structural ablation spares content retrieval. Symmetric double dissociation. |
| No | No | Yes | Geometry-function gap: structural heads W_K-inert for both tasks. W_K is not the mechanism linking structural heads to positional retrieval. |
| Yes | Yes | — | Both tasks impaired — structural heads contribute to content retrieval; no clean functional boundary. |
| No | Yes | No | Content retrieval specifically impaired, positional spared — structural heads selectively impair content retrieval (unexpected; would invert the hypothesized role). |
| Partial | — | — | Mixed; report as inconclusive with detailed item-level analysis. |

---

## Run.py will be written AFTER this document is committed to attention-geometry.

Commit hash will be recorded as `prereg_commit` in `registry.json`.

---

*Pre-registration written: 2026-09-23, ~12:20 PM MDT.*
*Author: Ariel, solo physics room session.*
