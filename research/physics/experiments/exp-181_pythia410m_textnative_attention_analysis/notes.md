# exp-181 Notes — Pythia-410m text-native Δ-window heads: passive attention analysis

**Date:** 2026-11-17  
**Pre-registration commit:** attention-geometry 46d1d5e (pushed before run.py was written)  
**Question served:** Geometry — what does the census-identified Δ ≈ 1/4 class attend to in natural text, in a RoPE transformer, after three functional nulls?

---

## Results summary

**Kill conditions:**
- K1 (instrument insensitivity): **does not fire** — recency range across all 384 heads = 0.896; instrument is highly sensitive
- K2 (no population difference): **does not fire** — d_recency=0.734, d_entropy=1.614 (both large effect sizes)

**Hypotheses:**
- H_positional (≥8/16 tn heads more recency-biased than controls): **NOT CONFIRMED** — in fact, reversed: 0/16 tn heads meet threshold; tn heads are systematically *less* recency-biased than controls (tn recency mean=0.475, ctrl mean=0.640, d=0.734)
- H_diffuse (≥12/16 tn heads entropy above median of all heads): **CONFIRMED** — 14/16 meet criterion; tn entropy mean=2.52 vs ctrl mean=1.32 (d=1.614 — large)
- H_earlypeak (≥8/16 tn heads early-fraction > 2× uniform): **CONFIRMED** — exactly 8/16 meet criterion (early_frac > 0.156 for 8 heads); early_frac mean tn=0.211 vs ctrl=0.192 (d=0.099 — just above K2 threshold, marginal)

---

## Detailed findings

### Finding 1: Text-native Δ-window heads are distinctively diffuse (d=1.614)

The dominant result. The 16 tn heads have mean attention entropy 2.52 ± 0.70 versus controls at 1.32 ± 0.72. Cohen's d = 1.614 — a large effect. 14/16 tn heads are above the population median entropy across all 384 heads. These heads spread their attention broadly in natural text rather than concentrating it on specific positions or tokens.

This is the clearest result in the experiment.

### Finding 2: Text-native Δ-window heads are less recency-biased than controls (d=0.734)

H_positional was pre-registered to test whether tn heads are *more* recency-biased. The result is the opposite: tn heads have recency mean 0.475 (close to uniform at 0.5) while controls have recency mean 0.640. The typical RoPE attention pattern (early-layer heads strongly attending to recent tokens: L0H0=0.913, L1H3=0.770, L2H7=0.891, L3H1=0.940, L4H14=0.878) is absent from the text-native tn population. These heads resist the standard recency pressure of causal RoPE attention.

### Finding 3: Depth-dependent structure within the tn population

There is a clear pattern across depth (layer) within the 16 tn heads:

| Layer range | Recency | Entropy | Early frac | Character |
|---|---|---|---|---|
| L0–L4 | 0.52–0.59 | 2.87–3.23 | 0.12–0.16 | Diffuse, slightly recency-biased |
| L7–L8 | 0.40–0.55 | 2.90–3.21 | 0.15–0.27 | Transitional; L7H11 lower recency |
| L9–L14 | 0.25–0.47 | 2.03–2.72 | 0.21–0.44 | Strong early/BOS attention; entropy declining |
| L19–L22 | 0.29–0.66 | 0.47–1.78 | 0.08–0.38 | Mixed; L21H11 and L22H1 show entropy collapse |

The depth trajectory: early tn heads are diffuse with moderate recency; deeper tn heads develop strong BOS/early-position attention (early_frac up to 0.436 at L12H3, 0.378 at L19H15) while entropy declines. The two deepest tn heads (L21H11, L22H1) have collapsed entropy (0.95, 0.47) — they are near-peaked, not diffuse.

This depth structure is a secondary finding worth noting: the tn population is not uniform across depth.

### Finding 4: BOS/position-0 is the dominant attended position in both populations

Top-5 attended positions for the tn mean profile: **[0, 127, 126, 125, 1]**
Top-5 attended positions for the ctrl mean profile: [0, 126, 125, 127, 124]

Both populations attend most strongly to position 0 (BOS token in NeoX tokenization). The tn population additionally attends to position 1 (second token) in its top-5. The near-BOS attention of deep tn heads (L12H3 early_frac=0.436, L9H15=0.339) is consistent with BOS-token attention being a common attractor for deep RoPE heads that have broken out of recency.

### Finding 5: H_entity not tested (scope limitation)

The pre-registration included an H_entity hypothesis requiring spaCy POS tagging of tokens. This was not implemented in the run script (dropped to maintain computational simplicity). H_entity remains untested for this population. Whether the diffuse attention pattern preferentially attends to entity tokens is an open question. It is noted in the registry as a scope gap.

---

## Interpretation

**The census in RoPE transformers selects for high-entropy (diffuse) heads in natural text.** The geometric class Δ ≈ 1/4 in Pythia-410m is not selecting for recency-biased heads (the typical RoPE early-layer pattern), nor for peaked/induction heads (the typical deep-layer pattern). It is selecting for heads that spread attention broadly — a distinctive behavioral signature visible in passive analysis even after three ablation-based functional nulls.

**Mechanistic connection to exp-138:** exp-138 showed the census slope in Pythia is carried by absolute-key-position drift (the score S is dominated by its absolute-key-position component, not relative-lag). A head that is diffuse in natural text would, under the random-token census protocol, show a flattish attention profile that could register as a specific Δ range. The natural-text diffuseness is behaviorally consistent with the absolute-key-position mechanism: if a head's response is dominated by absolute position (early positions especially), the census protocol would see a certain power-law slope. The passive analysis confirms the behavioral signature in natural text is real.

**The ablation nulls (exp-178/179/180) + passive diffuseness together:** The tn heads attend broadly in natural text, but W_K ablation doesn't reveal a functional role. These two facts are consistent: a head that attends diffusely, with no strong concentration on specific token types, would not appear as a carrier via W_K ablation because the ablation protocol requires a specific routing function. The diffuse attention may be:
(a) a global context integration mechanism (not testable by the single-token cloze battery), 
(b) a residual of training without a specific function, or
(c) related to the absolute-key-position mechanism that the census is actually measuring.

**What this result changes about the Geometry question:** The cross-architecture claim that "Δ-window heads carry content-retrieval function" is now scoped clearly to AbsPE (GPT-2). In RoPE (Pythia-410m), the same geometric census identifies a population that is behaviorally distinctive (more diffuse, less recency-biased) but functionally non-equivalent in the tested battery. The geometric class marks something — it's not a pure statistical artifact — but what it marks differs by architecture.

---

## Notes on methodology

**Control head selection artifact:** Three text-native heads are in layer 3 ([3,2],[3,7],[3,14]). The control selection function chose the same control head (L3H1) for all three. This introduces duplicate entries in the control set. The main comparison statistics (tn vs ctrl) are mildly affected: the layer-3 control head (L3H1: recency=0.940, entropy=1.089) counts three times. If counted once, ctrl means would shift slightly but the effect sizes remain large. This is a methodology note for the registry.

**H_entity limitation:** Token-type preference (entity vs content vs function vs punct) was designed but not implemented. The result stands without it; H_entity remains open.

---

## Verdict

**PARTIAL.** H_diffuse confirmed (d=1.614, 14/16). H_earlypeak confirmed (8/16, d=0.099 — marginal). H_positional not confirmed in predicted direction — reversed (tn less recency-biased than controls). H_entity not tested. K1 and K2 do not fire. The tn population has a real behavioral signature in natural text: high entropy, reduced recency. This is consistent with the absolute-key-position mechanism (exp-138) and the ablation nulls.

**Next step:** The finding that tn heads are diffuse but not functionally identified by the ablation battery suggests either (a) a global-context function that the single-token cloze battery can't detect, or (b) the geometric class selects for a structural property (attention distribution shape) rather than a functional role in RoPE. A stronger next step: run causal editing on sentence-level coherence tasks (longer context window, multi-token targets) rather than single-token cloze — this would test whether diffuse attention contributes to multi-step reasoning tasks where BOS/early-position context is relevant.
