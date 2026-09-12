# exp-140 — World-model battery: does κ-amplification move entity-state coherence or positional retrieval?

**Ariel — September 12, 2026, ~12 AM MDT, Cursor, solo.**
**Pre-registration committed to attention-geometry before run.py is written.**
**Analysis-only from existing weights. No new training.**

---

## 1. Background and motivation

exp-137 measured the positional read gain κ̃(R) for every attention head in GPT-2 small:
the fraction of the positional field's energy that the read map R passes, normalised by the
isotropic baseline. Δ-window heads (L2H1, L3H4, L5H0, L7H11, L10H8) suppress the positional
field by an order of magnitude relative to isotropic (κ̃_K = 0.05–0.33). Steep local/induction
heads read the positional field preferentially (κ̃_K = 5–22, Spearman ρ = 0.78 with σ_pos).

exp-138 showed that the census slope is absolute-key-position drift, not a relative-lag law.
The Δ-window heads sit *above* the κ̃_K → σ_pos regression line: their σ_pos is higher than
their low positional gain would predict (exp-138 §C). Something else sets their census slope.

**The question:** The Δ-window heads' suppressed positional read distinguishes them from local heads.
Is this suppression *constitutive* of world-model coherence (entity-state tracking), or is κ̃
primarily a cursor for positional retrieval?  Two distinguishable outcomes:

- **P1 (coherence hypothesis):** Amplifying κ̃ for Δ-window heads degrades entity-state tracking
  (they behave more like positional/local heads, losing world-model coherence), with no equivalent
  degradation of positional retrieval or with improvement of positional retrieval.

- **P2 (retrieval hypothesis):** Amplifying κ̃ for Δ-window heads improves positional retrieval
  (they now read position more strongly, better serving list-lookup), with no equivalent effect on
  entity-state tracking.

- **P3 (null):** Neither metric changes more than the sham control. The κ-edit is not load-bearing
  for either function at the tested scale.

These are mutually exclusive only at the level of which direction moves more. Both can move — the
diagnostic is the *differential* effect.

---

## 2. Intervention design

### 2a. Amplification protocol

For each of the 5 Δ-window heads {L2H1, L3H4, L5H0, L7H11, L10H8}:

1. Compute the position-mean residual stream x̄_i at layer ℓ, using the same frozen random-token
   census protocol as exp-112 (SEQ_LEN=512, N_INPUTS=50, SEED=42 → extended via the saved
   position-mean field logic).  Δ_i = x̄_i − mean_i(x̄_i)  (the positional field).

2. SVD: Δ = U S Vt. Take P_k = Vt[:4]  (shape 4 × 768, the top-4 PC directions in d_model).

3. Extract W_K_h = W_K slice for head h (shape 768 × 64).

4. Compute the positional projection component:
   W_K_proj = P_k.T @ (P_k @ W_K_h)   (shape 768 × 64)

5. Amplified key matrix:
   W_K_amp = W_K_h + γ * W_K_proj
   where γ = 2.0 (tripling the positional component: adds 2 copies of the existing projection).

6. Verify K3: recompute κ̃(W_K_amp, Δ) ≥ 0.5 on ≥ 4/5 structural heads. If the gate fires,
   increase γ to 4.0 and retry once; if it fires again, abort with INCONCLUSIVE verdict.

Only the c_attn weight slice for the key sub-matrix of the five Δ-window heads is modified.
W_Q and W_V are unchanged. c_attn bias is unchanged.

### 2b. Sham protocol

For each of the 5 Δ-window heads:

1. Compute P_perp: a rank-4 random matrix in the orthogonal complement of P_k.
   Method: sample 4 random vectors in R^768, orthogonalize against P_k and against each other
   (Gram–Schmidt), normalise.  Seed: 2026091200 + head_index.

2. Sham projection component:
   W_K_perp_proj = P_perp.T @ (P_perp @ W_K_h)  (shape 768 × 64)

3. Sham key matrix:
   W_K_sham = W_K_h + γ_sham * W_K_perp_proj
   where γ_sham is chosen so that ||ΔW_K_sham||_F = ||ΔW_K_amp||_F (matched perturbation magnitude).

4. Verify K3-sham: recompute κ̃(W_K_sham, Δ). This must NOT exceed 0.5; if it does, the sham
   protocol is flawed — report as a protocol note (does not fail the run, but flags in notes.md).

Three model states: **original**, **κ-amplified**, **sham**. The amplified and sham versions differ
only in the 5 Δ-window heads' W_K matrices.

---

## 3. Tasks

### Task A — Entity-state tracking (world-model coherence)

20 short passages (< 80 tokens each in GPT-2 tokenisation) where a named entity changes state
across multiple events, followed by a cloze prompt asking for the current state.

**Construction rules:**
- Each passage uses one entity (object or location variable), one or two state types.
- State changes are explicit surface-level updates (no inference required beyond tracking).
- The correct completion token is unambiguous and common (top-1000 GPT-2 vocabulary).
- Passages are held constant across model conditions (original / κ-amp / sham).

**Example:** "The candle was lit. Sarah blew the candle out. Then she lit the candle again.
Now the candle is [lit/unlit]."  → correct token: "lit"

**Scoring:** P_correct_A(item i) = next-token log-probability of the correct state token at the
cloze position.  Primary metric: median P_correct_A across 20 items, per condition.

### Task B — Positional retrieval (list lookup)

20 prompts where items are listed in order and the model is asked for the item at a specified position.

**Construction rules:**
- Lists of 4–6 single-word items; positions queried are interior (not the last item).
- Query is consistent with GPT-2 continuation: "The [ordinal] item is" (no instruction-style
  framing — the model continues the natural text).
- Items are common English words (top-5000 frequency), each a single GPT-2 token.
- The correct completion token is the listed item at the stated position.
- Items held constant across conditions.

**Example:** "The list: apple, banana, cherry, date, elderberry. The second item is"
→ correct token: "banana"

**Scoring:** P_correct_B(item i) = next-token log-probability of the correct item token.
Primary metric: median P_correct_B across 20 items, per condition.

---

## 4. Primary predictions

| Label | Prediction | Criterion | Condition for fire |
|---|---|---|---|
| **P1 (coherence)** | Δ-window κ-amplification degrades entity-state tracking differentially | ΔP_A(amp − original) < ΔP_B(amp − original) − threshold | Both: ΔP_A < −1.0 nats AND ΔP_A < ΔP_B − 1.5 nats |
| **P2 (retrieval)** | κ-amplification improves positional retrieval differentially | ΔP_B(amp − original) > ΔP_A(amp − original) + threshold | Both: ΔP_B > +1.0 nats AND ΔP_B > ΔP_A + 1.5 nats |
| **P3 (null)** | No differential effect beyond sham | |ΔP_A − ΔP_sham_A| < 0.5 nats AND |ΔP_B − ΔP_sham_B| < 0.5 nats | Both within sham range |

(P1 and P2 are not mutually exclusive: entity tracking can worsen AND positional retrieval can
improve. The diagnostic is the differential direction.)

Secondary analysis: per-item signed rank of the score change, Wilcoxon signed-rank test (20 items per
task); p reported but not a kill condition given n=20.

---

## 5. Kill conditions

| Gate | Condition | Consequence |
|---|---|---|
| **K1** | Sham produces a larger absolute effect than κ-amplification on both tasks | Edit specificity fails; verdict INCONCLUSIVE; protocol note filed |
| **K2** | Baseline (original model) median log-prob < −10 nats on either task | Task is too hard / not sensible for GPT-2; redesign required; INCONCLUSIVE |
| **K3** | κ̃(W_K_amp) < 0.5 on ≥ 2/5 structural heads at γ = 4.0 | Amplification protocol failed; INCONCLUSIVE; redesign needed |

---

## 6. Population and protocol fidelity

- Model: GPT-2 small (124M), `openai-community/gpt2`, bfloat16 on MPS.
- Positional field: computed fresh using exp-112's frozen random-token protocol (SEQ_LEN=512,
  N_INPUTS=50, SEED=42), consistent with exp-137.
- γ = 2.0 (first attempt); γ = 4.0 (if K3 fires at γ = 2.0).
- Tasks constructed before run.py is executed; items frozen in the pre-registration (§7 below).
- No reruns after the first complete pass.

---

## 7. Task items (frozen at registration)

**Task A — Entity-state tracking (20 items)**

Items are constructed to have a single unambiguous correct next token and to require state-tracking
rather than surface-level statistics. The correct token is the continuation word.

```
A01: "The lamp was on. Dave turned the lamp off. Then he turned the lamp on again. The lamp is now" → "on"
A02: "The jar was open. Maria closed the jar. The jar is now" → "closed"
A03: "The door was closed. Tom opened the door and walked through. The door is" → "open"
A04: "The bottle was full. He poured half of it out. Then he filled it back up. The bottle is now" → "full"
A05: "The cat was outside. Anna let the cat inside. Then she let the cat back outside. The cat is" → "outside"
A06: "The window was shut. James opened the window. The window is now" → "open"
A07: "The bag was empty. Lisa filled the bag with books. Then she emptied it again. The bag is now" → "empty"
A08: "The switch was off. He flipped the switch on, then flipped it off again. The switch is now" → "off"
A09: "The box was closed. She opened the box, took out an apple, and closed the box again. The box is now" → "closed"
A10: "The faucet was running. He turned the faucet off. The faucet is now" → "off"
A11: "The light was on. She turned it off. Then she turned it back on. The light is" → "on"
A12: "The cup was full. He drank half and then refilled it. The cup is now" → "full"
A13: "The fire was burning. The rain put the fire out. The fire is now" → "out"
A14: "The gate was open. She closed the gate behind her. The gate is now" → "closed"
A15: "The phone was off. He turned the phone on to make a call, then turned it back off. The phone is now" → "off"
A16: "The drawer was shut. She opened it, removed a pen, and shut it again. The drawer is now" → "shut"
A17: "The engine was running. He stopped the engine. The engine is now" → "off"
A18: "The curtains were open. She closed them for the night. The curtains are now" → "closed"
A19: "The stove was off. She turned it on to cook, then turned it off after eating. The stove is now" → "off"
A20: "The valve was open. The plumber closed the valve. The valve is now" → "closed"
```

**Task B — Positional retrieval (20 items)**

```
B01: "Colors: red, blue, green, yellow. The second color is" → "blue"
B02: "Fruits: apple, mango, cherry, grape. The third fruit is" → "cherry"
B03: "Animals: cat, dog, fish, bird, rabbit. The fourth animal is" → "bird"
B04: "Months: January, March, July, October. The third month is" → "July"
B05: "Planets: Mars, Venus, Jupiter, Saturn, Mercury. The second planet is" → "Venus"
B06: "Numbers: one, three, seven, twelve. The third number is" → "seven"
B07: "Names: Alice, Bob, Carol, Dan. The first name is" → "Alice"
B08: "Countries: France, Spain, Italy, Greece, Poland. The fourth country is" → "Greece"
B09: "Shapes: circle, square, triangle, oval. The second shape is" → "square"
B10: "Seasons: spring, summer, autumn, winter. The third season is" → "autumn"
B11: "Metals: gold, silver, copper, iron. The third metal is" → "copper"
B12: "Days: Monday, Wednesday, Friday, Sunday. The second day is" → "Wednesday"
B13: "Birds: eagle, robin, sparrow, hawk, dove. The third bird is" → "sparrow"
B14: "Letters: alpha, beta, gamma, delta, epsilon. The fourth letter is" → "delta"
B15: "Coins: penny, nickel, dime, quarter. The second coin is" → "nickel"
B16: "Stars: Sirius, Vega, Rigel, Altair. The first star is" → "Sirius"
B17: "Trees: oak, pine, maple, birch, cedar. The third tree is" → "maple"
B18: "Gems: ruby, sapphire, emerald, diamond. The third gem is" → "emerald"
B19: "Spices: salt, pepper, cumin, thyme. The second spice is" → "pepper"
B20: "Flowers: rose, lily, tulip, daisy, violet. The fourth flower is" → "daisy"
```

---

## 8. Verdict mapping

| Condition | Verdict |
|---|---|
| P1 fires (and not K1) | PARTIAL (coherence hypothesis supported) |
| P2 fires (and not K1) | PARTIAL (retrieval hypothesis supported) |
| Both P1 and P2 fire | CONFIRMED (bidirectional differential effect, coherence ↓ and retrieval ↑) |
| P3 fires (and not K1) | INCONCLUSIVE (null result) |
| K1 fires | INCONCLUSIVE (edit not specific) |
| K2 or K3 fires | INCONCLUSIVE (task or intervention design failure) |

No result is labelled FALSIFIED here — the hypotheses are not yet at the strength where a null
result would kill a spine claim; it would update the picture and seed a stronger test. A null is
reported as INCONCLUSIVE with the full numbers.

---

*Pre-registered 2026-09-12. Committed to attention-geometry before run.py exists.*
