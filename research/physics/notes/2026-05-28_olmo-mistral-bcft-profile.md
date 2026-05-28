# OLMo-7B and Mistral-7B BCFT Δ profiles — May 28, 2026

*exp-039. Analysis-only: existing exp-025 BCFT JSONs, 100% coverage for both models.*

---

## Question

Do OLMo-7B (ALiBi PE) and Mistral-7B (RoPE+SWA) show positional-encoding-dependent Δ elevation, as the exp-029 ALiBi toy and exp-036/030 RoPE findings would predict?

---

## Pre-stated hypotheses

Before looking at numbers (stated in exp-039 script):

1. **OLMo-7B (ALiBi):** Δ_med (R²>0.90) > 0.249 (GPT-2 learned PE). Expected: 0.30–0.45, from exp-029 toy showing ALiBi elevates apparent Δ.
2. **Mistral-7B (RoPE+SWA):** Δ_med > 0.249; near Pythia-410m BCFT 0.358 if RoPE dominates. SWA may depress deep-layer Δ.
3. **GPT-Neo (ALiBi, 89% coverage):** tracks OLMo direction if ALiBi elevation is consistent.

---

## Results

R²>0.90 filter, 32-layer band thirds (L0–10 / L11–21 / L22–31):

| Model | PE | Δ_med | n_good | shallow | mid | deep |
|---|---|---|---|---|---|---|
| GPT-2-small (exp-007, ref) | learned | 0.2493 | 44/144 | — | — | — |
| GPT-2-medium (exp-031, ref) | learned | 0.2589 | 68/384 | — | — | — |
| GPT-2-large (exp-031, ref) | learned | 0.2819 | 176/720 | 0.379 | 0.248 | 0.155 |
| Pythia-410m BCFT (exp-036, ref) | RoPE | 0.358 | 67/384 | 0.472 | 0.318 | — |
| **OLMo-7B** | **ALiBi** | **0.2654** | **194/1024** | **0.337** | **0.302** | **0.209** |
| **Mistral-7B** | **RoPE+SWA** | **0.2984** | **126/1024** | **0.273** | **0.446** | **0.334** |
| GPT-Neo-2.7B (partial) | ALiBi | 0.1259 | 290/572 | 1.739 | 0.115 | 0.115 |

---

## Hypothesis verdicts

**H1 (OLMo ALiBi elevation): Partial.**

OLMo Δ_med = 0.2654 > GPT-2 0.249. Elevation is real but small (+0.016). The expected range was 0.30–0.45 based on the exp-029 ALiBi toy. The real OLMo model does not show the large ALiBi elevation the toy predicted. Possible explanations: (a) OLMo's training dynamics partially counteract the ALiBi geometric effect; (b) 7B-scale model training differences confound the comparison; (c) exp-029 toy overestimated the effect because it didn't account for learned suppression of ALiBi signal in real models.

**H2 (Mistral RoPE+SWA near Pythia): Partial — closer to GPT-2 than Pythia.**

Mistral Δ_med = 0.2984. Situated between GPT-2 (0.249) and Pythia (0.358), but nearer GPT-2 by margin (|0.2984−0.249| = 0.049 vs |0.2984−0.358| = 0.060). SWA appears to partially suppress the RoPE elevation visible in Pythia-410m. The non-monotone band profile (shallow 0.273, mid **0.446**, deep 0.334) is unlike any other model measured — this mid-layer Δ peak is new.

**H3 (GPT-Neo ALiBi tracks OLMo): Not evaluable.** GPT-Neo's extreme shallow band (1.739) and low mid/deep (0.115) reflect the known alternating global/local architecture (May 17 session), not ALiBi per se. The global average 0.126 is near trivial, pulled down by global attention layers. GPT-Neo is not a clean test of ALiBi elevation.

---

## New finding: tentative PE hierarchy

Four data points suggest an ordering:

```
GPT-2 (learned, 0.249) < OLMo (ALiBi, 0.265) < Mistral (RoPE+SWA, 0.298) < Pythia (RoPE, 0.358)
```

If genuine: **RoPE > RoPE+SWA > ALiBi > learned PE** in terms of Δ elevation. This would mean positional encoding hardness (relative to the training process) correlates with Δ elevation — RoPE's fixed geometric pattern forces higher apparent scaling exponents than learned embeddings that can adapt.

**Caveat:** This is not a controlled comparison. OLMo (7B, ALiBi) vs Pythia (410m, RoPE) differ in size, architecture, training data, training steps. GPT-2-large (learned PE, 36L) sits at Δ = 0.282, above Mistral's 0.298 is not in the expected ordering. The depth-scale confound is real.

**What a clean test would need:** Same architecture, same scale, same training data — only PE type varied. The exp-033 sigmoid falsification test (softmax vs sigmoid, same architecture) is the right controlled design. The PE ordering needs a similar controlled design.

---

## Mistral mid-layer Δ peak — new feature

The Mistral band profile (0.273 / 0.446 / 0.334) is non-monotone. This is different from:
- GPT-2-large: 0.379 / 0.248 / 0.155 (monotone decreasing)
- OLMo: 0.337 / 0.302 / 0.209 (monotone decreasing)
- Pythia-410m (exp-036 layer bands): shallow 0.472 / deep 0.318 (decreasing)

Mistral's mid-network layers (L11–21) have Δ_med = 0.446, which is the highest band value of any model in this analysis. Sliding window attention restricts the effective receptive field per layer; the mid-network layers may be where the RoPE+SWA combination produces the steepest effective attention decay (hence highest Δ). Deep Mistral layers (0.334) recover somewhat — possibly where global context routing dominates despite SWA.

This is a hypothesis, not a conclusion. Testing it would require per-layer plots and model-specific attention range data. Filed as an open question.

---

## OLMo depth profile

OLMo shows monotone declining Δ: shallow 0.337, mid 0.302, deep 0.209. The deep-layer Δ (0.209) is below the SYK fixed point (0.25). This is similar to the GPT-2-large deep-layer pattern (0.155 deep), where deep layers approach the trivial fixed point. OLMo's deep layers are less suppressed than GPT-2-large's, possibly because ALiBi's additive bias maintains some effective long-range decay even in later layers.

---

## What this means for the main thread

The primary question (does attention sit at an SYK-class fixed point, and what does PE type do to it?) is now tested against two more full-coverage 7B models:

- OLMo global Δ = 0.265: within the SYK neighborhood (|Δ−0.25| = 0.015)
- Mistral global Δ = 0.298: within the SYK neighborhood (|Δ−0.25| = 0.048), at the edge

Both models' global Δ is consistent with SYK-class proximity. Neither shows a value far outside the neighborhood (like Δ > 0.6 or Δ < 0.1). The PE type shifts where in the neighborhood, but doesn't pull out of it.

**The more pressing question remains:** does sigmoid attention (non-softmax) also sit near SYK? OLMo and Mistral both use softmax, different PE. The sigmoid test (exp-033) is the architecturally distinct test. This session does not replace that test.

---

## Open questions

1. Is the PE hierarchy (RoPE > RoPE+SWA > ALiBi > learned) real under controlled comparison?
2. What drives Mistral's mid-layer Δ peak? SWA window schedule? RoPE interaction with mid-depth training?
3. Does the ALiBi elevation become larger at smaller model scale (where training has less capacity to adapt to the PE)?
4. GPT-Neo per-layer plot (alternating Δ pattern): already documented May 17, but not quantified against these new reference points. Could do analysis-only.

---

*Registry: exp-039. References: exp-007 (GPT-2), exp-031 (GPT-2-medium/large), exp-036 (Pythia-410m BCFT), exp-029 (ALiBi toy), exp-033 (sigmoid, still blocked).*
