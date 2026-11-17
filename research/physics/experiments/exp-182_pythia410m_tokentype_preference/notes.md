# exp-182 — Notes: Pythia-410m text-native Δ-window heads: token-type preference via spaCy POS/NER tagging

**Date:** 2026-11-17  
**Pre-registration:** attention-geometry 9d32bdd (pushed before run.py was written — git-attested)  
**Question:** Geometry — does the diffuse attention of text-native Δ-window heads (d=1.614 in exp-181) land preferentially on entity/content tokens?

---

## Setup verification

- Entity fraction in passages: 0.195 (19.5% of tokens are NER-tagged) — K1 does not fire.
- Alignment failure rate: 0.010 (1.0% of token positions unaligned) — K2 does not fire.
- Control heads selected from exp-118 Pythia-410m census: same layer distribution, highest |Δ−0.25| outside [0.20, 0.30].

---

## Results

| Hypothesis | Result | Detail |
|---|---|---|
| H_entity_prefer (≥10/16 entity_pref > 1.10) | **NOT CONFIRMED** | 0/16 |
| H_content_prefer (≥10/16 content_pref > 1.10) | **NOT CONFIRMED** | 4/16 |
| H_uniform (≥10/16 both |pref−1.0| < 0.10) | **NOT CONFIRMED** | 0/16 |
| H_population_contrast (d > 0.3 for entity or content) | **NOT CONFIRMED** | d_entity=0.108, d_content=0.037 |
| K3 (no population difference) | **FIRES** | — |

**Declared prior: H_uniform** — not confirmed. But it failed because tn heads are systematically *below* 1.0 for entities and content, not near 1.0. The preference ratios are further from 1.0 than H_uniform allowed, in the direction of under-attending.

---

## Key findings

**1. Text-native Δ-window heads under-attend to entity/content tokens in natural text.**

Tn entity_pref mean = 0.678 ± 0.223 (expected proportional baseline: 1.0). Tn content_pref mean = 0.913 ± 0.289. By arithmetic, function_pref > 1.0: tn heads attend to function tokens (determiners, prepositions, punctuation, auxiliaries, etc.) at an above-proportional rate.

This is not an tn-specific finding:
- Ctrl entity_pref mean = 0.700 ± 0.155
- Ctrl content_pref mean = 0.922 ± 0.194
- K3 fires: d_entity = 0.108, d_content = 0.037

The pattern is a **Pythia-410m-wide effect** — the architecture as a whole under-attends to named entities and content words relative to their frequency, distributing excess attention to function/syntactic tokens. The geometric census identifies no unique token-type signature.

**2. The H_uniform declared prior was wrong in its direction assumption.**

H_uniform assumed tn heads would be near-proportional (|pref−1| < 0.10). Instead, they're systematically off-center — under-attending to semantically rich tokens. This is a finding about the form of non-uniformity, not evidence against non-uniformity.

**3. Individual head range is wide.**

Entity_pref per tn head (sorted by layer):
```
L0H3:  0.581
L1H8:  0.879
L2H8:  0.813
L3H2:  0.661, L3H7:  0.857, L3H14: 0.631
L4H12: 1.001
L7H11: 0.658
L8H15: 0.613
L9H15: 0.659
L12H3: 0.592, L12H13: 1.087
L14H15: 0.508
L19H15: 0.706
L21H11: 0.528
L22H1:  0.080  ← extreme outlier, virtually no attention on entities
```

L22H1 (entity_pref = 0.080) is a striking outlier — this deepest tn head avoids named entities almost completely. Its content_pref is also low (0.295). It is likely attending near-exclusively to BOS or syntactic structure tokens. This is worth examining if a single-head anatomy is designed.

**4. Connection to exp-181 passive signature.**

exp-181 found tn heads are diffuse (entropy d=1.614) and less recency-biased than controls. The exp-182 result adds a second layer: the diffuseness is not semantically targeted. The tn population spreads attention broadly across the sequence (exp-181) without preferentially targeting meaningful content (exp-182). The Pythia-wide function-token preference is consistent with general language-model behavior: function tokens (connectives, auxiliaries, prepositions) are high-frequency and syntactically load-bearing, and attention to them is common across architecture classes.

**5. The scope conclusion from the three-task arc is not deepened by this finding.**

The three-task ablation arc (exp-178/179/180) established that tn W_K routing carries no detectable function. The passive analysis (exp-181/182) confirms no functional attention pattern either — neither position-preferential, nor entity-preferential, nor content-preferential. The geometric census identifies a population that:
- Has a real population-level attention signature in terms of *distribution* (diffuse, reduced-recency, exp-181)
- Has no population-level attention signature in terms of *content target* (exp-182)

The geometric classification captures a property of *how* attention is distributed (broadly, not recency-biased), not *where* it goes (no semantic preference beyond the architecture-wide function-token baseline).

---

## What this changes

**For the census cross-architecture claim:**
The exp-181/182 pair now gives the most complete characterization of Pythia-410m tn heads' passive behavior: they're geometrically identified heads that attend diffusely, without recency bias, without semantic targeting. The question of what positive function they serve is still open — the passive signature is real but its functional interpretation is not yet available from the instruments applied.

**For H_uniform as a hypothesis design:**
H_uniform checked for |pref−1| < 0.10, implicitly assuming the baseline would be ~1.0. The actual baseline in Pythia-410m is well below 1.0 for entities (both tn and ctrl at ~0.68–0.70). A better-designed uniform test would check whether tn preference ratios are indistinguishable from ctrl preference ratios — which K3 and H_population_contrast both confirm: they are. The finding is not "uniform" but "population-indistinguishable."

**For future work:**
The function-token preference is a Pythia-wide finding worth noting in OVERVIEW.md if confirmed on other architectures. For the tn population specifically: the next informative instrument would need to look at something other than POS/NER categories — syntactic depth (dependency parse distance from root), coreference structure (do tn heads attend to antecedents?), or clause-boundary structure.

---

## Honest negative

The declared prior (H_uniform) did not fire in the direction predicted. All four hypotheses failed to fire. K3 fires. This is a clean honest negative at the population-contrast level: the geometric census has no additional behavioral signature beyond exp-181's diffuse/reduced-recency finding.

---

## Artifacts

- `prereg.md` — pre-registration (9d32bdd, pushed before run.py was written)
- `run.py` — experiment script
- `results.json` — full results with per-head preference ratios
- `notes.md` — this file
