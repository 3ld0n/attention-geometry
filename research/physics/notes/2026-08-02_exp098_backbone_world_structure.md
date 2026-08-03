# exp-098: Backbone Collapse Is World Structure, Not Vocabulary

*2026-08-02, evening physics room session*

---

## Summary

exp-098 (C-alien-realnames) tested whether the L0 backbone collapse observed in exp-097 (C-alien) was caused by the alien vocabulary — entity names like "Vex," "Nul," "Ort" having weak token embeddings due to limited pre-training exposure. The test: replace alien name pools with common English first names (Alice/Ben/Clara, Fred/Grace/Henry, Kate/Leo/Maya) while holding all world mechanics identical.

**Result: H_vocab FALSIFIED. H_backbone_absent CONFIRMED.**

n_backbone = 0 at all three seeds (1800, 1801, 1802). Real English names — with strong pre-training token statistics — do not restore L0 backbone formation. The backbone collapse is due to world structure, not vocabulary.

---

## Full results

| Seed | n_conf | n_syk_near | Δ_med | n_deep (L3–L5) | n_backbone (L0) | layer_dist |
|------|--------|-----------|-------|----------------|-----------------|------------|
| 1800 (s0) | 12/48 | 0 | 0.727 | 7 | 0 | {1:4, 2:1, 3:4, 4:3} |
| 1801 (s1) | 12/48 | 0 | 0.845 | 4 | 0 | {1:5, 2:3, 3:3, 5:1} |
| 1802 (s2) | 10/48 | 0 | 0.609 | 4 | 0 | {1:3, 2:3, 3:3, 4:1} |
| **median** | **12** | **0** | **0.727** | **4** | **0** | — |
| control (rand) | 0/48 | 0 | N/A | 0 | 0 | — |

For reference, exp-097 (C-alien, alien names):

| Metric | exp-097 values | exp-098 values |
|--------|---------------|---------------|
| n_backbone | [0,0,0] median=0 | [0,0,0] median=0 |
| Δ_med | [0.758,1.044,1.185] med=1.04 | [0.727,0.845,0.609] med=0.727 |
| n_deep | [2,3,7] med=3 | [7,4,4] med=4 |
| n_conf | [10,10,17] med=10 | [12,12,10] med=12 |

---

## What the comparison tells us

**Backbone collapse (n_backbone=0): vocabulary-inert.** The absence of L0 backbone heads is identical in both experiments. The English names Alice/Ben/Clara are among the most frequent named-entity tokens in GPT-NeoX's training corpus. If weak token embeddings were the cause, we would see partial or full recovery with these names. None occurred. The backbone collapse is structural: something about the alien world's mechanics prevents L0 from developing the backbone function.

**UV arrest: vocabulary has a marginal effect.** Δ_med dropped from ~1.04 (exp-097 median) to ~0.727 (exp-098 median). This is real — English names with rich semantic associations seem to pull the model slightly closer to the IR fixed point. But the effect is modest (0.3 reduction in Δ_med, still 4× above C-NAT's ~0.17) and does not reach the criterion for H_uv_reduced (< 0.5). UV arrest is primarily due to world structure.

**n_deep: marginal improvement.** Median n_deep increased from 3 to 4. Within the variance of the series (s0 hit n_deep=7, a clear outlier; the other two seeds are at 4).

---

## What world structure features drive backbone collapse and UV arrest?

Neither is vocabulary. Both persist with English names. The remaining candidates in the C-alien world design:

1. **Limited entity count (4 types: Flurps, Blurns, Zarbs, and the implied interaction pairs).** TinyStories has a much larger cast of named entities across its corpus. Hypothesis: more entities → richer identity tracking structure → backbone re-forms. Ablation: train on C-alien with N=15–20 distinct entity names.

2. **Rigid deterministic rules (4 causal rules: rule A/B/C/D, each deterministic).** TinyStories stories have stochastic, varied causal sequences — characters surprise each other, outcomes vary. Rigid rules may suppress the need for the kind of predictive tracking backbone heads perform.

3. **Repetitive sentence template.** C-alien generates from a small pool of sentence templates. TinyStories has linguistic variety even within story arcs. The backbone in natural text may depend on diverse syntactic contexts.

4. **Semantic richness (goals, emotions, settings).** TinyStories characters have desires, emotions, settings. C-alien characters have only states (active/resting) and actions (interact). The backbone might track not just entity identity but entity *salience*, which requires richer semantic content.

**Recommended next step:** (a) C-alien-rich: same world, but increase N_ENTITY_TYPES to 15–20 names and add stochastic rule variants (each rule fires with p=0.7; alternative outcomes with p=0.3). This tests whether world *complexity* is the missing ingredient. Register prior before running.

---

## Relationship to UV arrest

The UV arrest (Δ_med >> 0.25) is now observed in both exp-097 and exp-098, with modest vocabulary sensitivity. The pre-registered framework for UV arrest is:

- C-alien represents a held world, but the RG flow from UV to IR requires something C-NAT has that C-alien lacks.
- exp-091/093/094 showed that *ordering* is necessary but not sufficient for full convergence (sentence-shuffled C-NAT also shows mild UV elevation).
- exp-097/098 show that *world-holding* without real-world semantics produces partial conformal formation (n_deep=3–4) but arrests the RG flow far from the IR fixed point.

Interpretation: the conformal fixed point at Δ=0.25 (q=4 SYK) requires not just world-holding and ordering, but *semantic richness of the world being held*. A world with 4 entities and 4 rules can be held perfectly and generate perfect narrative ordering — and still arrest the flow. The IR fixed point requires a world with enough complexity that the attention statistics genuinely sample the q=4 correlator class.

This is a new constraint on the formation hypothesis: world-holding is necessary but not sufficient. Richness of the held world is an independent variable.

---

## Series state after exp-097 + exp-098

| Corpus | n_deep (median) | n_backbone (median) | Δ_med (median) | n_conf (median) |
|--------|----------------|---------------------|---------------|----------------|
| C-NAT (full arc) | 5–7 | 7–8 | ~0.17 | 11–15 |
| C-NAT-anon (entity anon, partial) | 4 (2 seeds) | 7–8 | ~0.14 | 13–14 |
| C-NAT-half (half arc) | 3 | 7–8 | ~0.17 | 12 |
| C-alien-realnames (exp-098) | 4 | 0 | 0.727 | 12 |
| C-alien (exp-097) | 3 | 0 | 1.04 | 10 |
| C-NAT-shuf (sentence shuffle) | 2 | 6–7 | — | 8–9 |
| C-PCFG (hierarchy only) | 0 | — | — | 0 |

Two axes are now independently documented:
- **Ordering axis** (C-PCFG → C-NAT-shuf → C-NAT-half → C-NAT): drives n_deep and n_conf upward; backbone preserved; Δ_med approaches C-NAT.
- **Semantics axis** (C-alien → C-alien-realnames → C-NAT-anon → C-NAT): backbone restoration from 0 to 7–8; Δ_med reduction from 1.0+ to ~0.14; n_deep partial gain.

The entity anonymization result (exp-096, partial — s1/s2 show n_deep=4, n_backbone=7–8, Δ_med=0.14) is structurally important: removing cross-story name identity from C-NAT (full real world) costs 1–3 deep heads but preserves backbone and Δ_med. This is the mirror of exp-098: starting from the alien world, adding real names doesn't restore backbone (world structure dominates); starting from natural text, removing name identity costs a few deep heads but backbone survives (ordering/world-richness dominates).

*exp-096 verdict pending s0 completion (~10 PM MDT); s0 relaunched at 8:33 PM MDT.*
