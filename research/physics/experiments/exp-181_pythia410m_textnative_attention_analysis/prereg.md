# exp-181 Pre-registration — Pythia-410m text-native Δ-window heads: passive attention analysis in natural text

**Date:** 2026-11-17  
**Registered before:** run.py has not been written  
**Question served:** Geometry — what does the census-identified geometric class (Δ ≈ 1/4) attend to in natural text in a RoPE transformer, after three functional nulls?

---

## Background

exp-118 identified 16 text-native Δ-window heads in Pythia-410m (WikiText-103 validation, R²≥0.90, Δ∈[0.20,0.30]). Three subsequent experiments applied the full W_K ablation battery: Task C (content retrieval, exp-178: null), Task B (positional retrieval, exp-179: null), Task A (entity-state tracking, exp-180: null). H_arch_full is supported: these heads carry none of the three tested functional roles.

The W_K ablation battery asks *do these heads carry a function via their query routing?* The passive attention analysis asks *where do these heads actually look in natural text?* This is a mechanistic instrument orthogonal to the ablation battery.

---

## Population

16 text-native Δ-window heads in Pythia-410m (EleutherAI/pythia-410m), identified in exp-118:

```
[[0,3],[1,8],[2,8],[3,2],[3,7],[3,14],[4,12],[7,11],[8,15],[9,15],[12,3],[12,13],[14,15],[19,15],[21,11],[22,1]]
```

(Format: [layer, head])

Controls: 16 non-Δ-window heads sampled from the same layer distribution (one control head per text-native head's layer, lowest absolute deviation from mean Δ in that layer, Δ outside [0.20, 0.30]).

---

## Protocol

**Data:** WikiText-103 test split (held out from exp-118 census, which used validation). Tokenize with Pythia tokenizer (GPT-NeoX). Take 100 non-overlapping passages of 128 tokens each (12,800 tokens total). Exclude passages where any token position is padding.

**Forward pass:** Standard Pythia-410m forward pass, no weight modification. Capture raw attention weights (pre-softmax logits converted to softmax probabilities) for all heads on all 100 passages.

**Measures computed per head:**

1. **Mean recency score (R):** For each sequence, compute R = Σ_j attn_weight(i→j) × (j / seq_len) averaged over all query positions i > 0 and all sequences. R ∈ [0, 1]; R = 0.5 = uniform; R > 0.5 = recency-biased; R < 0.5 = primacy-biased.

2. **Attention entropy (H):** For each sequence, compute H = -Σ_j attn_weight(i→j) × log(attn_weight(i→j)+ε) for each query position i, then average over i and sequences. High H = diffuse; low H = peaked.

3. **Self-attention fraction (SA):** Mean fraction of attention weight on the query token's own position (i→i). Some RoPE heads carry near-identity behavior.

4. **Top-attended positions:** For each head, compute mean attention received by each absolute position (averaged over query positions that can attend to it and over sequences). Report the top-5 attended positions and whether they are early-token (pos < 10), late-token (pos > seq_len - 10), or mid-sequence.

5. **Token-type preference:** Tokenize passages with spaCy (en_core_web_sm). For each token position, label as one of: ENTITY (NNP, NNPS, NE-tagged), CONTENT (NN, NNS, VB*, JJ, RB), FUNCTION (DT, IN, CC, TO, MD), PUNCT (punctuation). Compute mean attention received per token-type category, normalized by the category's frequency in the passages. token_type_preference[category] = (mean_attn[category]) / (n_tokens[category] / n_total_tokens). Value > 1 = above-average attention to this category.

---

## Hypotheses

**H_entity:** ≥8/16 text-native Δ-window heads have ENTITY token_type_preference ≥ 1.3 (above-average attention to named entities/nouns by ≥30%). *Prior: low-moderate confidence. The ablation nulls suggest no functional content-retrieval role, which makes entity-preferential attention less likely.*

**H_positional:** ≥8/16 text-native Δ-window heads have mean recency score R > mean R of the 16 control heads + 0.05. *Prior: moderate confidence. RoPE transformers often develop strong positional structure; the census identifies these heads on WikiText where recency patterns might dominate.*

**H_diffuse:** ≥12/16 text-native Δ-window heads have attention entropy H in the top-50% of all 384 heads. *Prior: low confidence. Diffuse attention would suggest the geometric class is entirely a weights phenomenon with no attention-pattern signature.*

**H_earlypeak:** ≥8/16 text-native heads have disproportionate attention on early positions (pos < 10), mean early-token attention > 2× uniform expectation. *Prior: moderate confidence given the absolute-key-position result from exp-138: earlier positions may carry disproportionate weight.*

---

## Kill conditions

**K1 — instrument insensitivity:** If the range of R values across all 384 heads is < 0.05 (all heads attend near-uniformly), the instrument is insensitive; mark INCONCLUSIVE and investigate whether the WikiText passages produce sufficient token-position variation.

**K2 — population size:** If the 16 test heads cannot be distinguished from the 16 control heads on any measure at effect size > 0.1 (Cohen's d < 0.1), the population distinction visible in the census is not reflected in natural-text attention patterns; the census is a purely statistical artifact of the census protocol, not a natural-text behavioral signature.

---

## Expected outcomes and what each would change

| Outcome | Interpretation | Changes |
|---|---|---|
| H_entity fires | Text-native Δ-window heads in RoPE carry semantic/entity function via attention pattern, not query routing; ablation battery missed it | Geometric class still predicts semantic orientation cross-architecture; ablation instrument is insufficient for RoPE |
| H_positional fires | Geometric class in RoPE indexes positional structure; "text-native" label on census is a training artifact, not a function signature | Census protocol conflates positional and semantic structure in RoPE; design implications for cross-family claims |
| H_diffuse fires | Geometric class in RoPE has no coherent attention pattern — purely a weights-level phenomenon | The census result is architecture-specific at the behavioral level; the cross-family census claim needs qualification |
| H_earlypeak fires | Early-position concentration predicted by exp-138 (absolute-key-position carrier) is visible in natural text | Mechanism link: the geometric class's key-position structure expresses as early-sequence attention in natural text |
| All null | The 16 heads are behaviorally indistinguishable from control heads on all passive measures | The census identifies a geometric population that is a pure weights-level phenomenon; needs new instrument design |
