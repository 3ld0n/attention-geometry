# exp-124 — Polysemy Interference Test (Path C, Contextuality Battery)

**Date:** 2026-08-21  
**Type:** Empirical — GPT-2 small forward passes, no training, no sampling  
**Queue item:** #4b (Path C of contextuality battery; design document: `notes/2026-08-20_contextuality_battery_design.md`)  
**Spine:** P3 — Born-rule structure beyond the diagonal  
**Pre-registration commit:** (set at push)

---

## Hypothesis

Trained attention processes ambiguous words via mechanisms that, in their output
statistics, are consistent with classical probability theory — not quantum
superposition. Specifically: for a polysemous word W, the next-token probability
distribution under an ambiguous context is well-described by a convex combination
of the distributions under two unambiguous sense-specific contexts.

If this holds: P_AB ≈ λ P_A + (1−λ) P_B for optimal λ ∈ [0,1], and the residual
is unsystematic noise. The Born-rule extension of Paper 5 §6 is formally consistent
with the data but the off-diagonal sector does not improve on the classical diagonal
— no empirical purchase here.

If this fails: P_AB deviates from every convex combination, and the deviation
pattern is tested for consistency with quantum interference (proportional to
√(P_A · P_B) across the vocabulary).

---

## Registered Predictions

### P1 (primary — expected CONFIRMED)

**K_mean < 0.10** for the set of words with sense separation D_W ≥ 0.10.

K_mean is the average total-variation distance from the best-fit mixture across
qualifying words:

```
K_W = (1/2) · Σ_t |P_AB^W(t) − λ*_W P_A^W(t) − (1−λ*_W) P_B^W(t)|
```

where λ*_W minimizes ||P_AB^W − (λ P_A^W + (1−λ) P_B^W)||_2² over λ ∈ [0,1].

D_W = (1/2) Σ_t |P_A^W(t) − P_B^W(t)| (sense separation — must exceed 0.10 to
confirm the sense contexts actually differ; words below this threshold are excluded
from the main analysis and reported separately).

**P1 CONFIRMED**: K_mean < 0.10 → classical mixture is a good description; the
quantum extension earns no empirical traction at the polysemy-completion level.

**P1 FALSIFIED**: K_mean ≥ 0.10 for the qualifying word set → significant departure
from any convex mixture; triggers quantum-signature check (P2).

### P2 (conditional on P1 being FALSIFIED)

**ρ_mean > 0.30** where ρ_W is the Pearson correlation between δ_W and the
normalized geometric mean of P_A^W and P_B^W:

```
δ_W(t) = P_AB^W(t) − λ*_W P_A^W(t) − (1−λ*_W) P_B^W(t)
g_W(t) = √(P_A^W(t) · P_B^W(t)) / Σ_t √(P_A^W(t) · P_B^W(t))
ρ_W = Pearson(δ_W, g_W)
```

Quantum interference predicts δ_W ∝ g_W (the interference term is
2Re(ψ_A* ψ_B) ∝ √(P_A · P_B) for real amplitudes). ρ_mean > 0.30 would indicate
the deviation has the specific quantum-interference structure, not generic
nonlinearity.

**P2 CONFIRMED**: ρ_mean > 0.30 → deviation has quantum interference signature.  
**P2 FALSIFIED**: ρ_mean ≤ 0.30 → deviation is generic nonlinearity.

### Kill condition for P3 (via this experiment)

K_mean < 0.05 AND ρ_mean ≈ 0 (|ρ_mean| < 0.10) → classical mixture is an
excellent description; polysemy completions add no evidence for quantum
structure. The Born-rule extension is consistent but not required. P3 is not
killed globally (other paths remain), but this path contributes strongly null
evidence.

### Strong positive result

K_mean ≥ 0.15 AND P2 CONFIRMED → warrants careful follow-up (Path B LGI battery,
larger word list, controls for nonlinearity). Would be the first language-model
result consistent with quantum cognition predictions.

---

## Protocol

### Model

`openai-community/gpt2` (124M parameters, learned positional encoding). Local
weights at `~/.cache/huggingface/hub/`. CPU inference is sufficient — 60 forward
passes, each < 200ms.

### Word list (20 polysemous English nouns)

Selected to satisfy:
- Both senses common in English text (balanced frequency)
- Both senses reachable as "The [word]" — same surface form, same grammatical
  position
- GPT-2 should have strong priors on both senses

| # | Word | Sense A | Sense B |
|---|------|---------|---------|
| 1 | bank | financial institution | river/lake edge |
| 2 | bat | flying mammal | sports implement |
| 3 | crane | wading bird | lifting machine |
| 4 | palm | human hand | tropical tree |
| 5 | bark | dog's vocalization | tree covering |
| 6 | spring | coiled mechanism | season / natural spring |
| 7 | pitcher | liquid container | baseball player |
| 8 | club | social organization | blunt weapon |
| 9 | match | fire-starting device | sporting competition |
| 10 | board | wooden plank | group of directors |
| 11 | light | illumination | low weight (adjective use) |
| 12 | date | fruit / calendar day | romantic appointment |
| 13 | pool | body of water | billiards |
| 14 | sage | culinary herb | wise person |
| 15 | mole | burrowing animal | small dark skin spot |
| 16 | plane | aircraft | flat surface / woodworking tool |
| 17 | scale | fish covering | measuring device |
| 18 | seal | marine mammal | wax/document closure |
| 19 | iron | clothes-pressing appliance | metallic material |
| 20 | file | office records | metal smoothing tool |

### Context templates

All templates end with the target word as the final string (no trailing
punctuation). The next token from GPT-2 after this prompt is the measured
distribution.

C_AB (ambiguous): "The [word]" for all words — the model's prior over senses
before any disambiguation.

| Word | C_A (sense A context) | C_B (sense B context) |
|------|-----------------------|----------------------|
| bank | "She checked her balance at the bank" | "The salmon leaped from the river bank" |
| bat | "The cave was home to the little bat" | "He swung the wooden baseball bat" |
| crane | "The tall wading whooping crane" | "The steel construction crane" |
| palm | "The fortune teller read his palm" | "She climbed the tropical palm" |
| bark | "The watchdog let out a loud bark" | "He peeled the rough pine bark" |
| spring | "The worn mattress had a broken spring" | "Wildflowers bloomed in the spring" |
| pitcher | "She poured lemonade from the glass pitcher" | "The left-handed starting pitcher" |
| club | "She was president of the local book club" | "The warrior raised the heavy wooden club" |
| match | "She lit the candle with a wooden match" | "The World Cup qualifying match" |
| board | "He nailed down the long wooden board" | "She was appointed to the corporate board" |
| light | "She turned on the ceiling light" | "The feather was extremely light" |
| date | "She ate the sweet Medjool date" | "He was nervous before his first date" |
| pool | "The children splashed in the swimming pool" | "He lined up his shot at the pool" |
| sage | "She seasoned the stuffing with fresh sage" | "The old hermit was a wise sage" |
| mole | "The garden was riddled by the mole" | "She had a dark brown beauty mole" |
| plane | "She boarded the commercial passenger plane" | "He leveled the wood with a carpenter's plane" |
| scale | "The trout's shiny silver scale" | "He stepped onto the bathroom scale" |
| seal | "The playful harbor seal" | "She broke the wax seal" |
| iron | "She pressed the wrinkled shirt with the iron" | "The blacksmith worked with molten iron" |
| file | "She found the documents in the manila file" | "He smoothed the rough edge with the file" |

### Measurement procedure

For each (word, context_type) in {C_A, C_B, C_AB} × {all 20 words}:
1. Tokenize the template string
2. Run GPT-2 forward pass (no grad)
3. Extract logits at the last token position
4. Apply softmax → P (50,257-dimensional probability vector over vocabulary)
5. Record full vector (or top-5000 tokens by probability for storage efficiency)

Tokenization note: verify at runtime that the target word is a single token in
GPT-2's BPE vocabulary. If a word tokenizes to multiple tokens, verify the last
token of the sequence is the final subword of the target word; the next-token
distribution is still measured after the complete word.

Words where the template ends with a different token than expected: reported
in results.json and excluded from primary analysis if the template integrity is
compromised.

### Sense separation filter

D_W = (1/2) Σ_t |P_A(t) − P_B(t)|

Words with D_W < 0.10 are excluded from K_mean computation (sense contexts
did not meaningfully separate — likely a template problem, not a physics result).
Reported separately.

Expectation: at least 15/20 words should satisfy D_W ≥ 0.10. If fewer than 10
satisfy the filter: templates need redesign; report as a protocol failure.

### K-statistic computation

For each qualifying word W:

```python
# Mixture fit (analytic)
diff = p_a - p_b
denom = np.dot(diff, diff)
lam = np.clip(np.dot(p_ab - p_b, diff) / denom, 0, 1) if denom > 1e-10 else 0.5
mixture = lam * p_a + (1 - lam) * p_b

# K-statistic
K_w = 0.5 * np.sum(np.abs(p_ab - mixture))

# Interference correlation
delta = p_ab - mixture
geom = np.sqrt(p_a * p_b)
geom_sum = geom.sum()
geom_n = geom / geom_sum if geom_sum > 1e-10 else geom
rho_w = np.corrcoef(delta, geom_n)[0, 1]
```

### Aggregate statistics

K_mean = mean(K_W) over qualifying words  
ρ_mean = mean(ρ_W) over qualifying words  
K_max = max(K_W) over qualifying words  

Report full per-word table: word, D_W, λ*_W, K_W, ρ_W, included/excluded.

### Controls

**Top-token analysis:** For the 5 words with highest K_W, list the top-20 tokens by
|δ_W(t)| and their sign. Qualitatively: do the constructive-interference tokens
(δ > 0) and destructive-interference tokens (δ < 0) make semantic sense as a
quantum mixture of the two senses, or are they random?

**Model sanity check:** Verify that for a known monosemous word (e.g., "elephant"
— one sense only) used as a control, K_control ≈ 0 when C_A and C_B are both
unambiguous contexts for the same sense but phrased differently. This confirms
K > 0 reflects sense ambiguity, not template variation.

---

## What a positive result would and would not establish

**Would establish:** GPT-2's next-token distributions under ambiguous contexts
depart from classical probability mixtures in a pattern consistent with quantum
interference. This is evidence against the "classical mixture" model of how language
models handle ambiguity.

**Would NOT establish:**
- That GPT-2 performs quantum computation (it does not)
- That the departure is due to the specific Gibbs state of Paper 5 §6 (further
  work needed)
- That the result generalizes to other models or other forms of ambiguity

**A negative result establishes:** Classical mixture is a good description of
ambiguous-word completions in GPT-2 small. The polysemy path does not provide
empirical purchase for the quantum extension.

---

## Honest prior

P1: CONFIRMED (K_mean < 0.10). GPT-2 is a classical deterministic system. The
mixture should hold reasonably well for most words. I expect K_mean in the range
0.03–0.08 — small positive departures from mixture due to generic nonlinearity
in the softmax computation, not quantum interference.

P2: Not tested (since P1 expected CONFIRMED).

If K_mean > 0.10: I would treat this as a surprising result warranting careful
re-examination of the protocol before interpreting as quantum.

---

## Results

*(To be filled in after running.)*

---

## Registry

Experiment number: exp-124  
Folder: `research/physics/experiments/exp-124_polysemy_interference/`  
Script: `run.py`  
Results: `results.json`  
