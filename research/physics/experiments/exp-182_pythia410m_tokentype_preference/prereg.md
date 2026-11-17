# exp-182 Pre-registration — Pythia-410m text-native Δ-window heads: token-type preference via POS/NER tagging

**Date:** 2026-11-17  
**Registered before:** run.py has not been written  
**Question served:** Geometry — do text-native Δ-window heads (diffuse in exp-181, d=1.614) attend preferentially to entity/content tokens, or is their diffuse attention uniformly distributed?

---

## Background

exp-181 confirmed that Pythia-410m text-native Δ-window heads are distinctively diffuse (entropy d=1.614 vs controls) and less recency-biased (reversed from prediction). The H_entity hypothesis was noted as a scope gap: spaCy token-type tagging was not implemented in exp-181.

exp-182 is the follow-on: same 100 WikiText-103 test passages, same population, same forward-pass protocol — add spaCy POS/NER alignment to measure whether the diffuse attention lands preferentially on semantic categories.

Three-task ablation arc (exp-178/179/180) established that these heads carry no detectable function via W_K routing in content retrieval, positional retrieval, or entity-state tracking. The passive analysis (exp-181) found they are diffuse in natural text. This experiment asks: diffuse toward what?

---

## Population

16 text-native Δ-window heads in Pythia-410m (EleutherAI/pythia-410m), identified in exp-118:

```
[[0,3],[1,8],[2,8],[3,2],[3,7],[3,14],[4,12],[7,11],[8,15],[9,15],[12,3],[12,13],[14,15],[19,15],[21,11],[22,1]]
```

16 control heads: same layer distribution, one per tn head's layer, highest |Δ − 0.25| outside [0.20, 0.30] (same as exp-181 controls).

---

## Protocol

**Data:** Same 100 WikiText-103 test passages (128 GPT-NeoX tokens each, seed=42) as exp-181. Regenerate with same seed or load from exp-181 saved passage indices.

**Forward pass:** Pythia-410m forward pass, no weight modification. Capture raw attention weights (post-softmax) for the 32 target heads (16 tn + 16 ctrl) on all 100 passages.

**Token-type classification:**
1. For each passage: decode the full token sequence to a string, recording each GPT-NeoX token's character start/end offset in the decoded text.
2. Run spaCy `en_core_web_sm` on the decoded text to obtain POS tags and NER spans.
3. Align each GPT-NeoX token position to its spaCy token by maximum character overlap. If a GPT-NeoX token spans multiple spaCy tokens, assign the dominant spaCy token (largest overlap). Record alignment failure if no spaCy token overlaps.
4. Classify each aligned GPT-NeoX token position as:
   - **ENTITY**: spaCy token has `ent_type_` != "" (is part of a named entity, any NER type)
   - **CONTENT**: spaCy token POS in {NOUN, PROPN, VERB, ADJ, ADV} and NOT ENTITY
   - **FUNCTION**: all others (DET, ADP, CCONJ, SCONJ, CONJ, AUX, PART, PRON, NUM, PUNCT, SPACE, INTJ, SYM, X, _SP, unrecognized)

**Preference ratio computation:**  
For each head h and each category c ∈ {ENTITY, CONTENT, FUNCTION}:
- For each query position i, compute `attn_on_c[i]` = sum of attention weights on positions j < i where token_type[j] == c (causal attention: only positions j ≤ i; include i=0 at position 0).
- `freq[c]` = n_tokens[c] / seq_len (category's proportional frequency)
- `preference_ratio[h, c]` = mean over all i over all 100 passages of (attn_on_c[i] / (freq[c] + ε)) — normalized so 1.0 = proportional attention.

Also compute: d(tn, ctrl) for entity_preference_ratio and content_preference_ratio (Cohen's d, pooled SD).

---

## Hypotheses

**H_entity_prefer:** ≥10/16 tn heads have entity_preference_ratio > 1.10.  
*Prior: low confidence. The ablation nulls across three task types (including entity-state tracking) suggest no content-retrieval function via W_K routing. If attention patterns also show no entity preference, the geometric class is fully decoupled from semantic content in RoPE.*

**H_content_prefer:** ≥10/16 tn heads have content_preference_ratio > 1.10.  
*Prior: low confidence, same reasoning.*

**H_uniform:** ≥10/16 tn heads have both |entity_preference_ratio − 1.0| < 0.10 AND |content_preference_ratio − 1.0| < 0.10 (diffuse and proportional — no selectivity).  
*Prior: **this is the declared prior** — moderate confidence. The ablation nulls and the diffuse/reduced-recency passive signature together suggest the geometric class in RoPE is a structural residual with no semantic targeting.*

**H_population_contrast:** d(tn_entity_pref, ctrl_entity_pref) > 0.3 OR d(tn_content_pref, ctrl_content_pref) > 0.3.  
*Prior: low-moderate. If tn heads are uniform while ctrl heads are non-uniform (e.g., recency-biased controls attend less to early-sequence entities), there may still be a population contrast even if tn preference is near 1.0.*

---

## Kill conditions

**K1 (entity undersampling):** If mean ENTITY token fraction across passages < 0.03 (< 3% of tokens are entities), the entity test is underpowered. Do not retract results; flag and report entity frequency explicitly.

**K2 (alignment failure rate):** If > 20% of GPT-NeoX token positions have alignment failure (no matching spaCy token), report as a methodology limitation. Results remain valid for the successfully aligned fraction.

---

## Expected outcomes and what each would change

| Outcome | Interpretation | Changes |
|---|---|---|
| H_entity_prefer fires | Tn heads attend preferentially to entities despite no ablation-detectable function; diffuse attention is semantically targeted | Geometric class has covert entity-attending role in RoPE; passive instrument reveals what ablation battery missed |
| H_content_prefer fires | Tn heads attend to content words broadly (not specifically entities); diffuse across open-class vocabulary | Geometric class indexes broad semantic orienting, not entity-specific function |
| H_uniform fires | Tn heads' diffuse attention is genuinely proportional — no semantic targeting of any kind | Geometric class in RoPE is a pure structural/weights phenomenon behaviorally decoupled from token-type semantics |
| H_population_contrast fires | Tn and ctrl heads differ in token-type preferences even if tn is near-uniform | The geometric classification has behavioral signature at the population level even if individual tn heads are uniform |
| All null | No hypothesis fires, population indistinguishable on preference ratios | The token-type instrument is uninformative for this population; different instrument needed (e.g., syntactic depth, coreference) |

---

## Notes

This experiment is **analysis_only: false** — it requires new forward passes on Pythia-410m (same model, same passages, but attention weights are not saved from exp-181 at the per-passage token-resolution needed for category alignment). spaCy `en_core_web_sm` must be installed before run.py is written.

The H_entity scope gap from exp-181 is the direct motivation for this experiment. exp-182 is a natural continuation of exp-181 rather than an independent design.
