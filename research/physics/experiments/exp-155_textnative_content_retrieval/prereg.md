# exp-155 — Text-native Δ-window heads: W_K ablation content-retrieval test

**Pre-registration for exp-155.**
**This document is committed to attention-geometry BEFORE run.py is written.**

**Ariel — 2026-09-23, ~12:45 AM MDT. Solo physics room.**
**Seeded by exp-154 observational analysis (2026-09-23).**

---

## What exp-154 established

The 16 text-native Δ-window heads in GPT-2 small are **spatially indistinguishable
from the 5 structural (random-native) heads** when measured under WikiText-103: both
populations send ~79% of attention weight to tokens >50 positions back, with similar
entropy (2.9 vs 3.1 nats) and mean attended distance (186 vs 179 tokens).

The distinguishing feature: the structural heads' long-range pattern persists under
random tokens (positionally driven), while the text-native heads' long-range attention
only appears under natural language (content driven).

**Working hypothesis (H_content):** The text-native heads support content-driven
long-range retrieval. Their W_K encodes semantic feature-matching capacity that guides
attention to contextually relevant tokens. Ablating W_K (setting it to zero) should
impair content-based retrieval over long distances, while leaving purely positional
retrieval (Task B, the structural heads' function) largely unaffected.

---

## Hypothesis

**H_content:** W_K ablation of the 16 text-native Δ-window heads in GPT-2 small
degrades performance on content-specified long-range retrieval (Task C) while leaving
positional retrieval (Task B) unchanged.

This would confirm a functional double dissociation:
- Structural (random-native) heads → positional retrieval [established, exp-141/142/143]
- Text-native heads → content retrieval [this experiment tests]

---

## Task designs

### Task B — Positional retrieval (control; unchanged from exp-141/143)

Items from the approved exp-143 battery: list-lookup tasks at a stated ordinal position.
Example:
```
Colors: red, blue, green, yellow. The second color is [___].
```
Sentence construction: 4–6 items in a list, position cued by an ordinal word.
Target: the token at the stated position.

### Task C — Content-specified long-range retrieval (new)

A passage introduces an entity and a distinctive property. After a filler passage of
~80–150 tokens (unrelated to the entity), a cloze prompt uses the entity name as the
retrieval cue. The model must retrieve the property using entity identity, not ordinal
position.

Structure:
```
[Setup sentence: entity → property.] [Filler: 80–150 tokens.] [Cue: entity reference → ___.]
```

The filler text does NOT mention the entity. Retrieval requires binding entity-name to
property over the filler distance.

**20 Task C items are committed with this pre-registration (see `task_c_items.json`).**

Item construction notes:
- Entity: a concrete noun unlikely to be predicted from local context (mansion, granary, etc.)
- Property: an adjective that wouldn't be predicted from entity alone in local context
- Filler: prose sentences drawn from a fixed template (geography, weather, abstract) that
  do not contain the entity name
- Cue: a partial sentence ending right before the target token (exact match to what the
  model would next-token-predict)
- Target: the property token(s) — logged as the log-probability assigned by the model to
  the correct token at the cue position

The target log-probability is the primary metric: we measure how much the ablation changes
the model's probability of the correct property token given the full context.

---

## Manipulations

### Ablation (W_K = 0)
Set W_K to zero for all 16 text-native heads:
```
L4H10, L7H1, L8H2, L9H4, L9H6, L10H1, L10H2, L10H10,
L11H0, L11H1, L11H2, L11H4, L11H5, L11H6, L11H7, L11H9
```

With W_K = 0, attention scores are identically zero for all key positions, and the
softmax produces a uniform distribution over causal positions. The head writes
mean(W_V @ x) to the residual stream — it no longer routes by content or position.

### Sham
Random W_K of the same Frobenius norm as the original W_K, for each head independently.
The sham preserves the scale of W_K modification while destroying the structure.

This follows the sham protocol from exp-152/153 (GPT-2 medium steep/local ablation).

---

## Metrics

For each item (Task B and Task C):
- `logp_orig`: log P(target | context, original model)
- `logp_abl`: log P(target | context, ablated model)
- `logp_sham`: log P(target | context, sham model)

Primary analysis:
- ΔP_C = median(logp_abl_C − logp_orig_C) — change in content retrieval performance
- ΔP_B = median(logp_abl_B − logp_orig_B) — change in positional retrieval performance
- Sham controls: ΔP_C_sham, ΔP_B_sham (should be near zero)
- Item-level direction: n_C_improved = count(logp_abl_C > logp_orig_C)

---

## Registered predictions and kill conditions

### P1 — Content retrieval degrades with ablation

**Prediction:** ΔP_C < −0.10 nats (ablation degrades content retrieval beyond the null
threshold).

**Justification:** If text-native heads are content retrievers, removing their W_K
should impair the mechanism. −0.10 nats matches the degradation threshold used in
exp-149/150 for the analogous Task B test in GPT-2 medium.

### P2 — Positional retrieval is spared

**Prediction:** |ΔP_B| < ΔP_C in absolute value — content retrieval is specifically
impaired; positional retrieval is less affected or unaffected.

This is the dissociation criterion. P2 fires even if Task B shows some degradation,
as long as it is meaningfully smaller than the Task C effect.

### Kill conditions

**K1 (wrong function):** ΔP_B < −0.10 nats AND item-level n_B_improved ≤ 6/20 —
text-native head ablation substantially impairs positional retrieval. If K1 fires,
the functional boundary between text-native and structural heads is not at the
positional/content axis.

**K2 (both inert):** |ΔP_C| < 0.10 AND |ΔP_B| < 0.10 — the text-native heads
are functionally neutral for both task types. K2 extends the geometry-function gap
(established for steep/local and medium-scale structural heads) to the text-native
population. Honest negative.

**K3 (ablation failure):** κ̃_K check fails — W_K is not fully zeroed (‖W_K‖_F > 0.01)
on ≥ 10/16 heads. This would indicate a protocol error.

---

## Interpretation table

| P1 fires | K1 fires | K2 fires | Interpretation |
|---|---|---|---|
| Yes | No | No | H_content confirmed: text-native → content retrieval, structural → positional (double dissociation) |
| No | No | Yes | Geometry-function gap extends to text-native population: no detected functional role |
| No | Yes | No | Text-native heads participate in positional retrieval; no functional dissociation |
| Yes | Yes | — | Contradiction — check protocol; ablation effect on both tasks |
| Partial | — | — | Mixed; report as inconclusive with detailed item-level analysis |

---

## Run.py will be written AFTER this document is committed to attention-geometry.

Commit hash will be recorded as `prereg_commit` in `registry.json`.

---

*Pre-registration written: 2026-09-23, ~12:45 AM MDT.*
*Author: Ariel, solo physics room session.*
