# exp-125 — Polysemy Interference Test, Length-Equated (Path C redesign)

**Date:** 2026-08-22  
**Type:** Empirical — GPT-2 small forward passes, no training, no sampling  
**Queue item:** exp-124b (redesign following context-length confound identified in exp-124)  
**Spine:** P3 — Born-rule structure beyond the diagonal  
**Pre-registration commit:** (set at push)

---

## Background — what exp-124 found and why this redesign is needed

exp-124 (2026-08-21, pre-reg 14fd5d4) ran the polysemy interference test with:
- C_A: sense-A disambiguating context, 4–9 words
- C_B: sense-B disambiguating context, 4–9 words
- C_AB: "The [word]" — 2 tokens

The results were K_mean = 0.679, rho_mean = −0.514 — both far outside the expected
range. The decisive diagnostic: monosemous control "elephant" (only one sense) had
K = 0.556, indistinguishable from the polysemous words (mean 0.679, std 0.164).

Since K should be near zero for a word with no competing senses, this is a
protocol failure, not a physics signal. The confound: C_AB at 2 tokens is far
shorter than C_A/C_B at 4–9 tokens, so C_AB has higher entropy. The mixture model
"fails" because it's comparing contexts of fundamentally different information
content, not because of quantum interference. The negative rho also follows
mechanically from the same entropy asymmetry.

This experiment fixes only the confound. Everything else — word list, K-statistic
and rho formulas, P1/P2 thresholds, statistical procedure — is unchanged. This
isolates the fix and lets the results be compared directly to exp-124.

---

## Hypothesis

Same as exp-124: For polysemous words, GPT-2 next-token distributions under
ambiguous contexts are well-described by a convex mixture of the distributions
under unambiguous sense-A and sense-B contexts.

If P_AB ≈ λ P_A + (1−λ) P_B: classical mixture holds; quantum extension earns no
empirical traction at this grain.

If P_AB deviates from every convex mixture with the quantum-interference pattern
(residual ∝ √(P_A · P_B)): non-classical signal.

---

## Registered Predictions

### P1 (primary — expected CONFIRMED after fixing the confound)

**K_mean < 0.10** for the qualifying word set (D_W ≥ 0.10).

Same formula as exp-124:
```
K_W = 0.5 · Σ_t |P_AB^W(t) − λ*_W P_A^W(t) − (1−λ*_W) P_B^W(t)|
λ*_W = argmin_λ ||P_AB − (λ P_A + (1−λ) P_B)||_2²
```

**Kill condition** (P3, partial): K_mean < 0.05 AND |ρ_mean| < 0.10 → classical
mixture is an excellent description; this path contributes strongly null evidence
for P3.

### P2 (conditional on P1 FALSIFIED)

**ρ_mean > 0.30** — quantum-interference pattern in the residual. Same formula as
exp-124.

### Diagnostic prediction (pre-stated, not a primary test)

**D_K = K_mean(polysemous) − K_control** should be small after the length-equating
fix. In exp-124, K_control = 0.556 ≈ K_mean = 0.679 (difference 0.123, no
separation). After fixing: K_control should drop toward 0 along with K_mean,
confirming the confound was the primary driver. If K_mean stays elevated while
K_control drops: genuine polysemy signal; re-examine carefully.

---

## Protocol

### The single change from exp-124

**C_AB template:** "He was thinking about the [word]"  
(6 words, neutral — same surface form for all 20 words, no sense disambiguation)

This is the ONLY change from exp-124. C_A and C_B templates are identical to
exp-124. The monosemous control is also re-run with the same C_A and C_B but the
new length-matched C_AB.

**Context length comparison:**
- C_AB (new): 6 words for all words
- C_A: 4–9 words (unchanged from exp-124; average ~7 words)
- C_B: 4–9 words (unchanged from exp-124; average ~6 words)

Note: C_A/C_B still vary in length (this was not changed). The fix addresses
the gross asymmetry (2 vs. 6–8 tokens) rather than achieving perfect length
matching. Perfect length matching would require redesigning all C_A/C_B templates;
that is deferred unless exp-125 still shows elevated K with control K near zero
(which would indicate the C_A/C_B length variation matters).

### Model

`openai-community/gpt2` (124M parameters). Local weights. CPU.

### Word list (unchanged from exp-124)

| # | Word | Sense A | Sense B |
|---|------|---------|---------|
| 1 | bank | financial institution | river edge |
| 2 | bat | flying mammal | sports implement |
| 3 | crane | wading bird | lifting machine |
| 4 | palm | human hand | tropical tree |
| 5 | bark | dog's vocalization | tree covering |
| 6 | spring | coiled mechanism | season |
| 7 | pitcher | liquid container | baseball player |
| 8 | club | social organization | blunt weapon |
| 9 | match | fire-starting device | sporting competition |
| 10 | board | wooden plank | group of directors |
| 11 | light | illumination | low weight |
| 12 | date | fruit / calendar day | romantic appointment |
| 13 | pool | body of water | billiards |
| 14 | sage | culinary herb | wise person |
| 15 | mole | burrowing animal | skin spot |
| 16 | plane | aircraft | flat surface / tool |
| 17 | scale | fish covering | measuring device |
| 18 | seal | marine mammal | wax closure |
| 19 | iron | clothes-pressing appliance | metal |
| 20 | file | office records | metal smoothing tool |

### Context templates — C_A and C_B (unchanged from exp-124)

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

### Context templates — C_AB (new, length-equated)

For all 20 words: `"He was thinking about the [word]"`

| Word | C_AB |
|------|------|
| bank | "He was thinking about the bank" |
| bat | "He was thinking about the bat" |
| crane | "He was thinking about the crane" |
| palm | "He was thinking about the palm" |
| bark | "He was thinking about the bark" |
| spring | "He was thinking about the spring" |
| pitcher | "He was thinking about the pitcher" |
| club | "He was thinking about the club" |
| match | "He was thinking about the match" |
| board | "He was thinking about the board" |
| light | "He was thinking about the light" |
| date | "He was thinking about the date" |
| pool | "He was thinking about the pool" |
| sage | "He was thinking about the sage" |
| mole | "He was thinking about the mole" |
| plane | "He was thinking about the plane" |
| scale | "He was thinking about the scale" |
| seal | "He was thinking about the seal" |
| iron | "He was thinking about the iron" |
| file | "He was thinking about the file" |

### Monosemous control

Word: "elephant" (one sense — large African/Asian mammal)

- C_A (unchanged): "She watched the large gray elephant"
- C_B (unchanged): "The African bush elephant"
- C_AB (new): "He was thinking about the elephant"

Expected: K_control < 0.10. If C_AB is now length-matched and there's only one
sense, the mixture model should hold well.

### Statistics — unchanged from exp-124

- D_W = TV(P_A, P_B); filter: D_W ≥ 0.10 to qualify
- K_W = TV(P_AB, best-fit mixture)
- ρ_W = Pearson(δ_W, normalized √(P_A · P_B))
- Aggregates: K_mean, K_std, K_max, ρ_mean over qualifying words
- K_control, ρ_control for the monosemous control

### What counts as confirming the confound was the issue

If K_mean drops substantially (say, to < 0.15) AND K_control drops to near 0
(< 0.10): the confound was the primary driver; P1 may be confirmed.

If K_mean stays elevated (> 0.20) while K_control drops substantially: the
length-equating fix worked for the control but not the polysemous words — potential
genuine signal, warrants further investigation.

If both stay elevated (K_control > 0.30): there is still a confound in the design
not yet identified (perhaps the C_A/C_B length variation itself matters; next
step would be to standardize all three contexts to the same length).

---

## Honest prior

**P1: CONFIRMED.** I expect that fixing C_AB to 6 words will substantially reduce
K for both the polysemous words and the monosemous control. K_mean likely drops
to 0.05–0.15. If K_mean ends up in [0.10, 0.20] with K_control < 0.10, it would
be worth checking whether the remaining signal is genuine or due to C_A/C_B length
variation (next step: standardize all three contexts).

My best guess: K_mean ≈ 0.06–0.12 (classical mixture, near-miss), K_control < 0.08.

---

## What this experiment does and does not establish

If P1 CONFIRMED: classical mixture holds for polysemous words in GPT-2 after
fixing the protocol. The polysemy path does not provide evidence for quantum
structure in next-token distributions. P3 receives strongly null evidence from
Path C.

If P1 CONFIRMED but K_mean ≈ 0.05–0.15 (near-boundary): report the numerical
value; do not interpret as confirming or falsifying quantum interference without
understanding the remaining deviation's source.

Does NOT establish: anything about quantum computation, the Gibbs state of Paper 5,
or other architectures.

---

## Registry

Experiment number: exp-125  
Folder: `research/physics/experiments/exp-125_polysemy_length_equated/`  
Script: `run.py`  
Results: `results.json`  
Predecessor: exp-124 (context-length confound identified there)
