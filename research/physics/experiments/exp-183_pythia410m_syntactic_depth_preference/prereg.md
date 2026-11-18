# exp-183 Pre-registration — Pythia-410m text-native Δ-window heads: syntactic depth preference via spaCy dependency parse

**Date:** 2026-11-18  
**Registered before:** run.py has not been written  
**Question served:** Geometry — instrument refinement/exploration (Direction block pause applies; logged as such; exploration is free per rule 1)

---

## Background

exp-181 found Pythia-410m text-native Δ-window heads are distinctively diffuse
(entropy d=1.614 vs controls) and less recency-biased (tn recency mean 0.475 vs
ctrl 0.640, reversed from prediction). exp-182 found no semantic targeting
(K3 fires: d_entity=0.108, d_content=0.037) — the entity/content under-attention
is an architecture-wide baseline equally present in control heads.

The combined exp-181/182 picture: tn heads are diffuse *by position* (tn-specific
signature) but not semantically selective (architecture-wide baseline). What
remains open is whether the diffuseness has *syntactic structure* — does the
population preferentially attend to tokens at specific syntactic depth levels
(distance from root in the dependency tree)?

This is motivated by the absolute-key-position mechanism (exp-138: census slope
is absolute-key-position drift, not relative-lag). If the tn population's
characteristic is about where in the sequence keys sit (absolute position), the
syntactic depth distribution may be orthogonal to it, predicting H_flat.
Alternatively, if tn heads' reduced recency reflects a bias toward structurally
important (syntactically shallow) tokens that happen to cluster in earlier
sequence positions, H_shallow may fire.

This experiment closes the passive characterization arc (exp-181/182/183) for
the Pythia-410m tn population at the level of syntactic structure. After exp-183
the question "what does the diffuse attendance attend to?" will have three
answered dimensions: positional distribution, semantic category, and syntactic
depth.

---

## Population

16 text-native Δ-window heads in Pythia-410m (EleutherAI/pythia-410m), identified
in exp-118:

```
[[0,3],[1,8],[2,8],[3,2],[3,7],[3,14],[4,12],[7,11],[8,15],[9,15],[12,3],[12,13],[14,15],[19,15],[21,11],[22,1]]
```

16 control heads: same layer distribution, one per tn head's layer, highest
|Δ − 0.25| outside [0.20, 0.30], excluding tn heads (same matching as exp-181/182).

---

## Protocol

**Data:** Same 100 WikiText-103 test passages (128 GPT-NeoX tokens each,
non-overlapping from test split, same as exp-181/182) — reproducible by taking
the first 100 × 128 = 12,800 tokens of the test split after joining with spaces.

**Forward pass:** Pythia-410m, no weight modification, `output_attentions=True`.
Capture post-softmax attention weights for the 32 target heads (16 tn + 16 ctrl)
on all 100 passages.

**Syntactic depth classification:**
1. For each passage: decode the full token sequence to a string, recording each
   GPT-NeoX token's character start/end offset (same tokenizer offset_mapping
   approach as exp-182).
2. Run spaCy `en_core_web_sm` on the decoded text to obtain the dependency tree.
3. For each spaCy token: compute syntactic depth = number of edges from root
   (root has depth 0; a direct dependent of root has depth 1; etc.). Use
   `spacy.tokens.Token.dep_` and climb `token.head` until `token.head.i == token.i`
   (the root is its own head in spaCy).
4. Align each GPT-NeoX token position to its spaCy token by maximum character
   overlap (same procedure as exp-182). Record alignment failures.
5. Classify each aligned GPT-NeoX token position into one of three depth bins:
   - **SHALLOW**: depth 0–1 (root and direct dependents — syntactically prominent)
   - **MEDIUM**: depth 2–3 (second and third level dependents)
   - **DEEP**: depth 4+ (syntactically embedded — subordinate clauses, deeply
     nested phrases)

**Preference ratio computation:**  
Same formula as exp-182. For each head h and each bin b:
- For each query position i (1 ≤ i < seq_len): compute `attn_on_b[i]` = sum of
  attention weights on positions j ≤ i where depth_bin[j] == b.
- `freq[b]` = n_tokens[b] / seq_len (bin's proportional frequency in this passage)
- `preference_ratio[h, b]` = mean over all i over all 100 passages of
  (attn_on_b[i] / (freq[b] + ε)), where ε = 1e-6.

Also compute Cohen's d between tn and ctrl preference ratios for each depth bin.

---

## Hypotheses

**H_deep:** ≥10/16 tn heads have deep_preference_ratio > 1.10.  
*Prior: low confidence (~15%). A head that is diffuse and non-recency-biased might
preferentially attend to deeply embedded constructions. But the absolute-key-position
mechanism (exp-138) suggests the tn character is about sequence position, not
linguistic depth, making deep selectivity unlikely.*

**H_shallow:** ≥10/16 tn heads have shallow_preference_ratio > 1.10.  
*Prior: low-moderate confidence (~20%). If reduced recency (exp-181) reflects
preference for earlier sequence positions, and syntactically shallow tokens
(roots, subjects) cluster in early sentence positions, H_shallow might fire.
But within a 128-token window this effect would be weak and confounded.*

**H_flat:** ≥10/16 tn heads have all three depth bins within [0.85, 1.15] of
proportional (all |depth_pref_bin − 1.0| < 0.15).  
*Prior: **declared prior** — moderate confidence (~60%). By analogy with exp-182's
K3 fire (no semantic targeting signal): if the tn population lacks semantic
selectivity, it likely also lacks syntactic depth selectivity. The positional
character (absolute-key-position mechanism) is orthogonal to syntactic structure.*

**H_contrast:** d(tn, ctrl) > 0.3 for any depth bin preference ratio.  
*Prior: low confidence (~15%). exp-182's population contrast for semantic
categories was d_entity=0.108 and d_content=0.037, both well below 0.3. By
analogy, expect d < 0.3 for depth bins as well.*

---

## Kill conditions

**K1 (shallow undersampling):** mean SHALLOW token fraction across passages < 0.05
(< 5% of tokens are roots or direct dependents). Flag and report; results for
shallow bin are underpowered.

**K2 (deep undersampling):** mean DEEP token fraction across passages < 0.05
(< 5% of tokens at depth ≥ 4). Flag and report; results for deep bin are
underpowered.

**K3 (no population difference):** d(tn, ctrl) < 0.3 for all three depth bins.
This is the kill condition analogous to exp-182's K3 — if it fires, the depth
preference instrument is uninformative at the population level, closing the
syntactic depth dimension.

---

## Expected outcomes and what each would change

| Outcome | Interpretation | What changes |
|---|---|---|
| H_deep fires | Tn heads preferentially attend to syntactically embedded tokens | Syntactic depth selectivity opens a new characterization dimension; motivates coreference/discourse tracking hypothesis |
| H_shallow fires | Tn heads preferentially attend to root-adjacent tokens | Consistent with structural-discourse hypothesis; reduced recency may be mediated by root/subject preference |
| H_flat fires | Tn heads' diffuse attention is syntactically proportional | Syntactic depth dimension closes; tn population passively characterized as: diffuse, non-recency, non-semantic, non-syntactic |
| K3 fires | No population contrast in depth preference | Same interpretation as exp-182's K3: the geometric census identifies no unique structural-attending signal in the passive instrument at this grain |
| K1 or K2 fires | Protocol underpowered for shallow/deep bin | Re-bin or use continuous depth measure in a follow-on |

---

## Notes

**analysis_only: false** — requires new forward passes (raw attention weights not
saved from exp-181/182 at per-passage resolution needed for depth alignment).
spaCy `en_core_web_sm` already installed (exp-182 session).

**Relationship to exp-182:** This experiment is a direct follow-on using the same
model, passages, and population. The only change is the alignment variable
(syntactic depth bins vs token-type categories). The same character-offset
alignment procedure applies.

**Post-arc plan (noted for context, not registered):** After exp-183, the
passive characterization arc for Pythia-410m tn heads will have covered
positional distribution (exp-181), semantic category (exp-182), and syntactic
depth (exp-183). If K3 fires again, the working hypothesis shifts to: tn head
diffuseness in RoPE reflects the absolute-key-position mechanism directly — a
structural property of the census protocol, not a functional signature detectable
at the passive level in natural text. The next direction, if any, would shift to
GPT-2 small tn heads (the semantic population) or to the LiM test from the
functional characterization task.
